"""Shared implementation for the four TCGA ablation-study scenarios.

The preprocessing protocol intentionally reproduces the protocol selected by
the researcher: each complete expression matrix is standardized independently
before the TCGA train/test split; every external validation matrix is
standardized independently.  Stacking meta-models are trained only from
out-of-fold (OOF) probabilities computed on the TCGA training subset.
"""

from __future__ import annotations

import json
import os
import platform
import re
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
import sklearn
from bayes_opt import BayesianOptimization
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

try:
    import xgboost
    from xgboost import XGBClassifier
except ImportError as exc:
    raise ImportError("Install xgboost: pip install xgboost") from exc

warnings.filterwarnings("ignore")


@dataclass(frozen=True)
class Settings:
    class_col: str = "Class"
    random_state: int = 42
    test_size: float = 0.30
    n_splits: int = 5
    n_jobs: int = 12
    bootstrap_iterations: int = 1000
    ci_level: float = 0.95
    init_points_rf: int = 10
    n_iter_rf: int = 35
    init_points_xgb: int = 10
    n_iter_xgb: int = 35
    init_points_meta: int = 5
    n_iter_meta: int = 15


RF_BOUNDS = {
    "n_estimators": (100, 600),
    "max_depth": (3, 45),
    "min_samples_split": (2, 20),
    "min_samples_leaf": (1, 10),
    "max_features": (0.03, 0.80),
    "max_leaf_nodes": (10, 500),
}

XGB_BOUNDS = {
    "n_estimators": (100, 800),
    "max_depth": (2, 10),
    "learning_rate": (0.01, 0.25),
    "subsample": (0.55, 1.00),
    "colsample_bytree": (0.05, 0.80),
    "min_child_weight": (1.0, 15.0),
    "gamma": (0.0, 5.0),
    "reg_alpha": (0.0, 5.0),
    "reg_lambda": (0.1, 10.0),
}

ID_CANDIDATES = ("SampleID", "Sample", "ID", "sample_id")


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def save_json(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, default=_json_default)


def clean_feature_names(columns) -> list[str]:
    return [re.sub(r"^X(?=\d)", "", str(column)) for column in columns]


def read_dataset(path: Path, class_col: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    data.columns = clean_feature_names(data.columns)
    data = data.drop(
        columns=[c for c in data.columns if str(c).startswith("Unnamed")],
        errors="ignore",
    )
    if class_col not in data.columns:
        raise ValueError(f"{path}: missing required column {class_col!r}")
    return data


def split_xy_ids(data: pd.DataFrame, class_col: str):
    id_col = next((c for c in ID_CANDIDATES if c in data.columns), None)
    ids = (
        data[id_col].astype(str).reset_index(drop=True)
        if id_col
        else pd.Series([f"row_{i}" for i in range(len(data))], name="SampleID")
    )
    non_features = [class_col, "Validation_dataset", *ID_CANDIDATES]
    X = data.drop(columns=[c for c in non_features if c in data.columns])
    X = X.apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        print("Warning: non-numeric/NaN expression values were replaced with 0.")
        X = X.fillna(0)
    y = data[class_col].astype(str).reset_index(drop=True)
    return X.reset_index(drop=True), y, ids


def standardize_independently(X: pd.DataFrame, name: str):
    scaler = StandardScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(X), columns=X.columns, index=X.index
    )
    print(f"{name}: independently standardized {X.shape[0]} x {X.shape[1]}")
    return scaled, scaler


def align_features(X: pd.DataFrame, features: list[str], name: str) -> pd.DataFrame:
    missing = sorted(set(features) - set(X.columns))
    if missing:
        raise ValueError(f"{name}: missing {len(missing)} genes; first: {missing[:10]}")
    return X.loc[:, features].copy()


def encode_labels(y: pd.Series, encoder: LabelEncoder, name: str) -> np.ndarray:
    unknown = sorted(set(y.unique()) - set(encoder.classes_))
    if unknown:
        raise ValueError(f"{name}: unknown classes: {unknown}")
    return encoder.transform(y)


def make_split(y: pd.Series, settings: Settings):
    indices = np.arange(len(y))
    return train_test_split(
        indices,
        test_size=settings.test_size,
        random_state=settings.random_state,
        stratify=y,
    )


def rf_params(raw: dict, settings: Settings) -> dict:
    return {
        "n_estimators": int(round(raw["n_estimators"])),
        "max_depth": int(round(raw["max_depth"])),
        "min_samples_split": int(round(raw["min_samples_split"])),
        "min_samples_leaf": int(round(raw["min_samples_leaf"])),
        "max_features": float(raw["max_features"]),
        "max_leaf_nodes": int(round(raw["max_leaf_nodes"])),
        "class_weight": "balanced",
        "bootstrap": True,
        "criterion": "gini",
        "random_state": settings.random_state,
        "n_jobs": settings.n_jobs,
    }


def xgb_params(raw: dict, n_classes: int, settings: Settings) -> dict:
    return {
        "objective": "multi:softprob",
        "num_class": n_classes,
        "n_estimators": int(round(raw["n_estimators"])),
        "max_depth": int(round(raw["max_depth"])),
        "learning_rate": float(raw["learning_rate"]),
        "subsample": float(raw["subsample"]),
        "colsample_bytree": float(raw["colsample_bytree"]),
        "min_child_weight": float(raw["min_child_weight"]),
        "gamma": float(raw["gamma"]),
        "reg_alpha": float(raw["reg_alpha"]),
        "reg_lambda": float(raw["reg_lambda"]),
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "random_state": settings.random_state,
        "n_jobs": settings.n_jobs,
    }


def fit_model(model, X, y, weighted: bool):
    if weighted:
        model.fit(
            X,
            y,
            sample_weight=compute_sample_weight(class_weight="balanced", y=y),
        )
    else:
        model.fit(X, y)
    return model


def cv_score(
    builder: Callable[[], object],
    X: pd.DataFrame,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    weighted: bool,
) -> float:
    scores = []
    for train_fold, valid_fold in folds:
        model = fit_model(
            builder(), X.iloc[train_fold], y[train_fold], weighted
        )
        scores.append(
            f1_score(
                y[valid_fold],
                model.predict(X.iloc[valid_fold]),
                average="weighted",
                zero_division=0,
            )
        )
    return float(np.mean(scores))


def optimize_base_models(
    X: pd.DataFrame,
    y: np.ndarray,
    n_classes: int,
    settings: Settings,
    out_dir: Path,
    prefix: str,
):
    folds = list(
        StratifiedKFold(
            n_splits=settings.n_splits,
            shuffle=True,
            random_state=settings.random_state,
        ).split(X, y)
    )

    rf_opt = BayesianOptimization(
        f=lambda **p: cv_score(
            lambda: RandomForestClassifier(**rf_params(p, settings)),
            X, y, folds, False,
        ),
        pbounds=RF_BOUNDS,
        random_state=settings.random_state,
        verbose=2,
    )
    rf_opt.maximize(
        init_points=settings.init_points_rf, n_iter=settings.n_iter_rf
    )

    xgb_opt = BayesianOptimization(
        f=lambda **p: cv_score(
            lambda: XGBClassifier(**xgb_params(p, n_classes, settings)),
            X, y, folds, True,
        ),
        pbounds=XGB_BOUNDS,
        random_state=settings.random_state,
        verbose=2,
    )
    xgb_opt.maximize(
        init_points=settings.init_points_xgb, n_iter=settings.n_iter_xgb
    )

    rf_best = rf_params(rf_opt.max["params"], settings)
    xgb_best = xgb_params(xgb_opt.max["params"], n_classes, settings)
    save_json(
        {"best_cv_f1_weighted": rf_opt.max["target"], **rf_best},
        out_dir / f"{prefix}_RF_best_hyperparameters.json",
    )
    save_json(
        {"best_cv_f1_weighted": xgb_opt.max["target"], **xgb_best},
        out_dir / f"{prefix}_XGB_best_hyperparameters.json",
    )
    _save_bo_history(rf_opt, out_dir / f"{prefix}_RF_BO_history.csv")
    _save_bo_history(xgb_opt, out_dir / f"{prefix}_XGB_BO_history.csv")

    rf = fit_model(RandomForestClassifier(**rf_best), X, y, False)
    xgb = fit_model(XGBClassifier(**xgb_best), X, y, True)
    return (
        {"RF": rf, "XGB": xgb},
        {"RF": rf_best, "XGB": xgb_best},
        {"RF": float(rf_opt.max["target"]), "XGB": float(xgb_opt.max["target"])},
        folds,
    )


def _save_bo_history(optimizer, path: Path) -> None:
    rows = [
        {"iteration": i + 1, "target_f1_weighted_cv": r["target"], **r["params"]}
        for i, r in enumerate(optimizer.res)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def oof_probabilities(
    X: pd.DataFrame,
    y: np.ndarray,
    params: dict[str, dict],
    folds: list[tuple[np.ndarray, np.ndarray]],
    n_classes: int,
):
    result = {name: np.zeros((len(y), n_classes)) for name in params}
    for train_fold, valid_fold in folds:
        rf = fit_model(
            RandomForestClassifier(**params["RF"]),
            X.iloc[train_fold], y[train_fold], False,
        )
        xgb = fit_model(
            XGBClassifier(**params["XGB"]),
            X.iloc[train_fold], y[train_fold], True,
        )
        result["RF"][valid_fold] = rf.predict_proba(X.iloc[valid_fold])
        result["XGB"][valid_fold] = xgb.predict_proba(X.iloc[valid_fold])
    return result


def optimize_meta_model(
    meta_X: np.ndarray,
    y: np.ndarray,
    settings: Settings,
    out_dir: Path,
):
    folds = list(
        StratifiedKFold(
            n_splits=settings.n_splits,
            shuffle=True,
            random_state=settings.random_state + 101,
        ).split(meta_X, y)
    )

    def objective(log10_c):
        scores = []
        for tr, va in folds:
            model = LogisticRegression(
                C=10.0 ** float(log10_c),
                max_iter=5000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=settings.random_state,
            )
            model.fit(meta_X[tr], y[tr])
            scores.append(
                f1_score(y[va], model.predict(meta_X[va]), average="weighted")
            )
        return float(np.mean(scores))

    optimizer = BayesianOptimization(
        f=objective,
        pbounds={"log10_c": (-4.0, 4.0)},
        random_state=settings.random_state,
        verbose=2,
    )
    optimizer.maximize(
        init_points=settings.init_points_meta, n_iter=settings.n_iter_meta
    )
    c_value = 10.0 ** float(optimizer.max["params"]["log10_c"])
    model = LogisticRegression(
        C=c_value,
        max_iter=5000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=settings.random_state,
    )
    model.fit(meta_X, y)
    _save_bo_history(optimizer, out_dir / "META_BO_history.csv")
    save_json(
        {
            "C": c_value,
            "best_cv_f1_weighted": optimizer.max["target"],
            "input": "OOF class probabilities only",
        },
        out_dir / "META_best_hyperparameters.json",
    )
    return model


def normalized_weights(values) -> np.ndarray:
    weights = np.maximum(np.asarray(values, dtype=float), 1e-12)
    return weights / weights.sum()


def weighted_average_probabilities(
    probabilities: list[np.ndarray], weights
) -> np.ndarray:
    matrices = np.stack(probabilities, axis=0)
    normalized = normalized_weights(weights)
    if matrices.shape[0] != len(normalized):
        raise ValueError("The number of probability matrices and weights differs")
    return np.tensordot(normalized, matrices, axes=(0, 0))


def average_probabilities(probabilities: list[np.ndarray]) -> np.ndarray:
    return weighted_average_probabilities(
        probabilities, np.ones(len(probabilities), dtype=float)
    )


def metric_values(y_true, y_pred) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Recall_macro": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "F1_macro": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Precision_weighted": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "Recall_weighted": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "F1_weighted": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def metrics_with_ci(y_true, y_pred, settings: Settings) -> dict:
    result = metric_values(y_true, y_pred)
    rng = np.random.default_rng(settings.random_state)
    sampled = {name: [] for name in result}
    for _ in range(settings.bootstrap_iterations):
        idx = rng.integers(0, len(y_true), len(y_true))
        values = metric_values(np.asarray(y_true)[idx], np.asarray(y_pred)[idx])
        for name, value in values.items():
            sampled[name].append(value)
    alpha = (1.0 - settings.ci_level) / 2.0
    for name, values in sampled.items():
        result[f"{name}_CI_low"] = np.quantile(values, alpha)
        result[f"{name}_CI_high"] = np.quantile(values, 1.0 - alpha)
    return result


def save_evaluation(
    y_true,
    probabilities: np.ndarray,
    sample_ids,
    encoder: LabelEncoder,
    dataset_name: str,
    model_name: str,
    out_dir: Path,
    settings: Settings,
) -> dict:
    prediction = np.argmax(probabilities, axis=1)
    frame = pd.DataFrame(
        {
            "SampleID": np.asarray(sample_ids),
            "True_class": encoder.inverse_transform(np.asarray(y_true)),
            "Predicted_class": encoder.inverse_transform(prediction),
        }
    )
    for i, class_name in enumerate(encoder.classes_):
        frame[f"Probability_{class_name}"] = probabilities[:, i]
    frame.to_csv(out_dir / f"{model_name}_predictions_{dataset_name}.csv", index=False)

    report = classification_report(
        y_true,
        prediction,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        out_dir / f"{model_name}_per_class_{dataset_name}.csv"
    )
    pd.DataFrame(
        confusion_matrix(
            y_true, prediction, labels=np.arange(len(encoder.classes_))
        ),
        index=[f"True_{c}" for c in encoder.classes_],
        columns=[f"Pred_{c}" for c in encoder.classes_],
    ).to_csv(out_dir / f"{model_name}_confusion_matrix_{dataset_name}.csv")
    return {
        "Model": model_name,
        "Dataset": dataset_name,
        **metrics_with_ci(y_true, prediction, settings),
    }


def save_environment(out_dir: Path, settings: Settings, scenario: str) -> None:
    save_json(
        {
            "scenario": scenario,
            "settings": asdict(settings),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "normalization_protocol": (
                "Independent StandardScaler fit_transform for the complete TCGA "
                "matrix before the 70/30 split and independently for each "
                "external validation matrix."
            ),
        },
        out_dir / "run_environment_and_config.json",
    )


def load_complete_data(data_path: Path, validation_files: dict[str, Path], settings):
    data = read_dataset(data_path, settings.class_col)
    X_raw, y, ids = split_xy_ids(data, settings.class_col)
    features = list(X_raw.columns)
    X, scaler = standardize_independently(X_raw, "TCGA_complete")
    validation = {}
    for name, path in validation_files.items():
        frame = read_dataset(path, settings.class_col)
        Xv_raw, yv, idv = split_xy_ids(frame, settings.class_col)
        Xv_raw = align_features(Xv_raw, features, name)
        Xv, validation_scaler = standardize_independently(Xv_raw, name)
        validation[name] = (Xv, yv, idv, validation_scaler)
    return X, y, ids, features, scaler, validation


def load_cluster_data(cluster_files: dict, settings: Settings):
    result = {}
    for name, paths in cluster_files.items():
        train_frame = read_dataset(Path(paths["train_test"]), settings.class_col)
        val_frame = read_dataset(Path(paths["validation"]), settings.class_col)
        X_raw, y, ids = split_xy_ids(train_frame, settings.class_col)
        Xv_raw, yv, idv = split_xy_ids(val_frame, settings.class_col)
        features = list(X_raw.columns)
        Xv_raw = align_features(Xv_raw, features, f"{name}_validation")
        X, scaler = standardize_independently(X_raw, f"{name}_TCGA")
        Xv, val_scaler = standardize_independently(Xv_raw, f"{name}_validation")
        result[name] = {
            "X": X, "y": y, "ids": ids, "features": features,
            "scaler": scaler, "Xv": Xv, "yv": yv, "idv": idv,
            "validation_scaler": val_scaler,
        }
    names = list(result)
    reference = result[names[0]]
    for name in names[1:]:
        current = result[name]
        if not reference["y"].equals(current["y"]):
            raise ValueError(f"{name}: TCGA class/order mismatch")
        if not reference["yv"].equals(current["yv"]):
            raise ValueError(f"{name}: validation class/order mismatch")
        # If real identifiers exist, this catches sample-order mismatches.
        if not reference["ids"].equals(current["ids"]):
            raise ValueError(f"{name}: TCGA SampleID/order mismatch")
        if not reference["idv"].equals(current["idv"]):
            raise ValueError(f"{name}: validation SampleID/order mismatch")
    return result


def run_complete(
    scenario: str,
    use_stacking: bool,
    data_path: str,
    validation_files: dict[str, str],
    output_dir: str,
    settings: Settings,
) -> None:
    started = time.perf_counter()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_environment(out, settings, scenario)
    validation_paths = {k: Path(v) for k, v in validation_files.items()}
    X, y, ids, features, scaler, validation = load_complete_data(
        Path(data_path), validation_paths, settings
    )
    train_idx, test_idx = make_split(y, settings)
    encoder = LabelEncoder().fit(y.iloc[train_idx])
    y_train = encoder.transform(y.iloc[train_idx])
    y_test = encode_labels(y.iloc[test_idx], encoder, "Test")
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    pd.DataFrame(
        {
            "SampleID": ids.iloc[np.r_[train_idx, test_idx]].values,
            "Original_index": np.r_[train_idx, test_idx],
            "Subset": ["train"] * len(train_idx) + ["test"] * len(test_idx),
            "Class": y.iloc[np.r_[train_idx, test_idx]].values,
        }
    ).to_csv(out / "train_test_split_info.csv", index=False)
    pd.Series(features, name="Gene").to_csv(out / "feature_names.csv", index=False)
    joblib.dump(encoder, out / "label_encoder.joblib")
    joblib.dump(scaler, out / "TCGA_independent_scaler.joblib")

    models, params, scores, folds = optimize_base_models(
        X_train, y_train, len(encoder.classes_), settings, out, "Complete"
    )
    for name, model in models.items():
        joblib.dump(model, out / f"{name}_model.joblib")

    if use_stacking:
        oof = oof_probabilities(
            X_train, y_train, params, folds, len(encoder.classes_)
        )
        np.savez_compressed(out / "OOF_probabilities.npz", **oof)
        meta_train = np.hstack([oof["RF"], oof["XGB"]])
        meta = optimize_meta_model(meta_train, y_train, settings, out)
        joblib.dump(meta, out / "meta_logistic_regression.joblib")
        combine = lambda matrices: meta.predict_proba(np.hstack(matrices))
        final_name = "Complete_Stacking"
    else:
        model_order = ["RF", "XGB"]
        model_weights = normalized_weights([scores[name] for name in model_order])
        combine = lambda matrices: weighted_average_probabilities(
            matrices, model_weights
        )
        final_name = "Complete_SoftVoting"
        save_json(
            {
                "models": model_order,
                "weights": dict(zip(model_order, model_weights)),
                "weighting_rule": "normalized optimized CV Weighted F1",
                "optimized_base_cv_scores": scores,
                "gene_count": len(features),
                "note": (
                    "With one complete feature space, the structural gene-count "
                    "factor cancels during normalization."
                ),
            },
            out / "soft_voting_info.json",
        )

    metrics = []
    for dataset_name, Xe, ye, ide in (
        ("Test", X_test, y_test, ids.iloc[test_idx]),
    ):
        probabilities = combine([m.predict_proba(Xe) for m in models.values()])
        metrics.append(
            save_evaluation(
                ye, probabilities, ide, encoder, dataset_name, final_name,
                out, settings,
            )
        )
    for name, (Xv, yv, idv, val_scaler) in validation.items():
        yv_enc = encode_labels(yv, encoder, name)
        probabilities = combine([m.predict_proba(Xv) for m in models.values()])
        metrics.append(
            save_evaluation(
                yv_enc, probabilities, idv, encoder, f"Validation_{name}",
                final_name, out, settings,
            )
        )
        joblib.dump(val_scaler, out / f"{name}_independent_scaler.joblib")
    pd.DataFrame(metrics).to_csv(out / "ALL_classification_metrics.csv", index=False)
    save_json(
        {"scenario": scenario, "wall_clock_seconds": time.perf_counter() - started,
         "status": "completed"},
        out / "run_summary.json",
    )


def run_clustered(
    scenario: str,
    use_stacking: bool,
    cluster_files: dict,
    output_dir: str,
    settings: Settings,
) -> None:
    started = time.perf_counter()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_environment(out, settings, scenario)
    clusters = load_cluster_data(cluster_files, settings)
    names = list(clusters)
    reference = clusters[names[0]]
    train_idx, test_idx = make_split(reference["y"], settings)
    encoder = LabelEncoder().fit(reference["y"].iloc[train_idx])
    y_train = encoder.transform(reference["y"].iloc[train_idx])
    y_test = encode_labels(reference["y"].iloc[test_idx], encoder, "Test")
    y_val = encode_labels(reference["yv"], encoder, "Validation")
    pd.DataFrame(
        {
            "SampleID": reference["ids"].iloc[np.r_[train_idx, test_idx]].values,
            "Original_index": np.r_[train_idx, test_idx],
            "Subset": ["train"] * len(train_idx) + ["test"] * len(test_idx),
            "Class": reference["y"].iloc[np.r_[train_idx, test_idx]].values,
        }
    ).to_csv(out / "common_train_test_split_info.csv", index=False)
    joblib.dump(encoder, out / "label_encoder.joblib")

    cluster_models, cluster_params, cluster_folds = {}, {}, {}
    cluster_scores, cluster_oof = {}, {}
    for name, data in clusters.items():
        cluster_out = out / name
        cluster_out.mkdir(exist_ok=True)
        pd.Series(data["features"], name="Gene").to_csv(
            cluster_out / "feature_names.csv", index=False
        )
        models, params, scores, folds = optimize_base_models(
            data["X"].iloc[train_idx],
            y_train,
            len(encoder.classes_),
            settings,
            cluster_out,
            name,
        )
        cluster_models[name], cluster_params[name], cluster_folds[name] = (
            models, params, folds
        )
        cluster_scores[name] = scores
        for model_name, model in models.items():
            joblib.dump(model, cluster_out / f"{model_name}_model.joblib")
        joblib.dump(data["scaler"], cluster_out / "TCGA_independent_scaler.joblib")
        joblib.dump(
            data["validation_scaler"],
            cluster_out / "validation_independent_scaler.joblib",
        )
        save_json(
            {"RF": scores["RF"], "XGB": scores["XGB"]},
            cluster_out / "optimized_base_cv_scores.json",
        )
        cluster_oof[name] = oof_probabilities(
            data["X"].iloc[train_idx],
            y_train, params, folds, len(encoder.classes_),
        )

    def cluster_vote(name: str, Xeval: pd.DataFrame):
        model_order = ["RF", "XGB"]
        model_weights = normalized_weights(
            [cluster_scores[name][model] for model in model_order]
        )
        return weighted_average_probabilities(
            [cluster_models[name][model].predict_proba(Xeval)
             for model in model_order],
            model_weights,
        )

    model_order = ["RF", "XGB"]
    within_cluster_weights = {
        name: normalized_weights(
            [cluster_scores[name][model] for model in model_order]
        )
        for name in names
    }
    oof_cluster_prob = {
        name: weighted_average_probabilities(
            [cluster_oof[name][model] for model in model_order],
            within_cluster_weights[name],
        )
        for name in names
    }
    local_oof_f1 = {
        name: float(
            f1_score(
                y_train,
                np.argmax(oof_cluster_prob[name], axis=1),
                average="weighted",
                zero_division=0,
            )
        )
        for name in names
    }
    gene_counts = {name: len(clusters[name]["features"]) for name in names}
    structural_quality_scores = {
        name: gene_counts[name] * local_oof_f1[name] for name in names
    }
    cluster_weights = normalized_weights(
        [structural_quality_scores[name] for name in names]
    )
    weighting_info = {
        "model_order": model_order,
        "within_cluster_model_weights": {
            name: dict(zip(model_order, within_cluster_weights[name]))
            for name in names
        },
        "within_cluster_rule": "normalized optimized CV Weighted F1",
        "cluster_gene_counts": gene_counts,
        "cluster_local_oof_f1_weighted": local_oof_f1,
        "cluster_raw_gene_count_times_oof_quality": structural_quality_scores,
        "normalized_cluster_weights": dict(zip(names, cluster_weights)),
        "across_cluster_rule": (
            "normalize(number of genes * local OOF Weighted F1)"
        ),
        "leakage_control": (
            "All model and cluster weights use TCGA training CV/OOF results only"
        ),
    }
    save_json(weighting_info, out / "weighting_info.json")

    if use_stacking:
        # Local OOF probabilities use the same CV-quality model weights as voting.
        # Across clusters, the stacking meta-model remains the ablated mechanism.
        np.savez_compressed(out / "OOF_cluster_probabilities.npz", **oof_cluster_prob)
        meta_X = np.hstack([oof_cluster_prob[name] for name in names])
        meta = optimize_meta_model(meta_X, y_train, settings, out)
        joblib.dump(meta, out / "cross_cluster_meta_logistic_regression.joblib")
        combine = lambda matrices: meta.predict_proba(np.hstack(matrices))
        final_name = "CrossCluster_Stacking"
    else:
        combine = lambda matrices: weighted_average_probabilities(
            matrices, cluster_weights
        )
        final_name = "Clustered_SoftVoting"
        save_json(
            {
                **weighting_info,
                "aggregation": "two-level weighted soft voting",
            },
            out / "soft_voting_info.json",
        )

    test_cluster_prob = [
        cluster_vote(name, clusters[name]["X"].iloc[test_idx]) for name in names
    ]
    val_cluster_prob = [
        cluster_vote(name, clusters[name]["Xv"]) for name in names
    ]
    metrics = [
        save_evaluation(
            y_test, combine(test_cluster_prob), reference["ids"].iloc[test_idx],
            encoder, "Test", final_name, out, settings,
        ),
        save_evaluation(
            y_val, combine(val_cluster_prob), reference["idv"],
            encoder, "Validation", final_name, out, settings,
        ),
    ]
    pd.DataFrame(metrics).to_csv(out / "ALL_classification_metrics.csv", index=False)
    save_json(
        {"scenario": scenario, "wall_clock_seconds": time.perf_counter() - started,
         "status": "completed"},
        out / "run_summary.json",
    )
