# ============================================================
# TCGA multiclass classification on common genes with per-dataset StandardScaler normalization and bootstrap confidence intervals
# Models: Random Forest, XGBoost
# Hyperparameter optimization: Bayesian Optimization
# Ensemble / stacking-level decision: weighted soft voting
# External validation: each validation dataset + combined validation
# ============================================================

import os
import re
import json
import joblib
import warnings
import numpy as np
import pandas as pd

from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, classification_report, confusion_matrix
)

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise ImportError("xgboost is not installed. Install it first: pip install xgboost") from exc

warnings.filterwarnings("ignore")

# ============================================================
# 1) Settings
# ============================================================

CLASS_COL = "Class"
RANDOM_STATE = 42
TEST_SIZE = 0.30
N_SPLITS = 5

BOOTSTRAP_ITERATIONS = 1000
CI_LEVEL = 0.95

DATA_PATH = os.path.join("Common_genes", "tcga_cancer_common_genes.csv")

VALIDATION_FILES = {
    "BLCA": os.path.join("Common_genes", "BLCA_validation_common_genes.csv"),
    "BRCA": os.path.join("Common_genes", "BRCA_validation_common_genes.csv"),
    "COAD": os.path.join("Common_genes", "COAD_validation_common_genes.csv"),
    "HNSC": os.path.join("Common_genes", "HNSC_validation_common_genes.csv"),
    "KIRC": os.path.join("Common_genes", "KIRC_validation_common_genes.csv"),
    "LGG": os.path.join("Common_genes", "LGG_validation_common_genes.csv"),
    "LUAD": os.path.join("Common_genes", "LUAD_validation_common_genes.csv"),
    "LUSC": os.path.join("Common_genes", "LUSC_validation_common_genes.csv"),
}

COMBINED_VALIDATION_FILE = os.path.join("Common_genes", "ALL_validation_common_genes.csv")
OUT_DIR = "Classification_Results_RF_XGB_without_normal_standardscaled_bootstrap_CI"
os.makedirs(OUT_DIR, exist_ok=True)

INIT_POINTS_RF = 10
N_ITER_RF = 35
INIT_POINTS_XGB = 10
N_ITER_XGB = 35

# ============================================================
# 2) Utility functions
# ============================================================

def clean_feature_names(columns):
    return [re.sub(r"^X(?=\d)", "", str(c)) for c in columns]


def read_dataset(path, class_col=CLASS_COL):
    data = pd.read_csv(path)
    data.columns = clean_feature_names(data.columns)
    unnamed_cols = [c for c in data.columns if str(c).startswith("Unnamed")]
    if unnamed_cols:
        data = data.drop(columns=unnamed_cols)
    if data.columns[-1] != class_col:
        raise ValueError(f"In {path}, last column must be {class_col}, but got {data.columns[-1]}")
    return data


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def split_xy(data, class_col=CLASS_COL):
    non_feature_cols = [class_col]
    for id_col in ["SampleID", "Sample", "ID", "sample_id", "Validation_dataset"]:
        if id_col in data.columns:
            non_feature_cols.append(id_col)
    X = data.drop(columns=non_feature_cols)
    y = data[class_col].astype(str)
    X = X.apply(pd.to_numeric, errors="coerce")
    if X.isna().sum().sum() > 0:
        print("Warning: NaN values found after numeric conversion. Filling NaN with 0.")
        X = X.fillna(0)
    return X, y


def standardize_expression_profiles(X, dataset_name):
    """
    Standardize gene-expression columns using sklearn.preprocessing.StandardScaler.
    Each dataset is scaled independently. Class labels are not included in X.
    """
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    print(f"{dataset_name}: StandardScaler applied to {X.shape[0]} samples and {X.shape[1]} genes.")
    return X_scaled


def bootstrap_confidence_intervals(y_true, y_pred, n_bootstrap=BOOTSTRAP_ITERATIONS,
                                   ci_level=CI_LEVEL, random_state=RANDOM_STATE):
    """
    Calculate non-parametric bootstrap confidence intervals for classification metrics.
    Resampling is performed over samples with replacement.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    rng = np.random.default_rng(random_state)

    metric_values = {
        "Accuracy": [],
        "Balanced_Accuracy": [],
        "Precision_macro": [],
        "Recall_macro": [],
        "F1_macro": [],
        "Precision_weighted": [],
        "Recall_weighted": [],
        "F1_weighted": [],
        "MCC": [],
    }

    if n == 0:
        return {f"{m}_CI_low": np.nan for m in metric_values} | {f"{m}_CI_high": np.nan for m in metric_values}

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        metric_values["Accuracy"].append(accuracy_score(yt, yp))
        metric_values["Balanced_Accuracy"].append(balanced_accuracy_score(yt, yp))
        metric_values["Precision_macro"].append(precision_score(yt, yp, average="macro", zero_division=0))
        metric_values["Recall_macro"].append(recall_score(yt, yp, average="macro", zero_division=0))
        metric_values["F1_macro"].append(f1_score(yt, yp, average="macro", zero_division=0))
        metric_values["Precision_weighted"].append(precision_score(yt, yp, average="weighted", zero_division=0))
        metric_values["Recall_weighted"].append(recall_score(yt, yp, average="weighted", zero_division=0))
        metric_values["F1_weighted"].append(f1_score(yt, yp, average="weighted", zero_division=0))
        metric_values["MCC"].append(matthews_corrcoef(yt, yp))

    alpha = 1.0 - ci_level
    low_q = 100.0 * alpha / 2.0
    high_q = 100.0 * (1.0 - alpha / 2.0)

    ci = {}
    for metric_name, values in metric_values.items():
        arr = np.asarray(values, dtype=float)
        ci[f"{metric_name}_CI_low"] = float(np.percentile(arr, low_q))
        ci[f"{metric_name}_CI_high"] = float(np.percentile(arr, high_q))

    return ci


def calculate_metrics(y_true, y_pred, dataset_name, model_name):
    return {
        "Model": model_name,
        "Dataset": dataset_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def save_report_and_cm(y_true, y_pred, target_names, model_name, dataset_name, out_dir):
    labels = np.arange(len(target_names))
    report_df = pd.DataFrame(
        classification_report(
            y_true, y_pred, labels=labels, target_names=target_names,
            output_dict=True, zero_division=0
        )
    ).transpose()
    report_df.to_csv(os.path.join(out_dir, f"{model_name}_classification_report_{dataset_name}.csv"))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"True_{cls}" for cls in target_names],
        columns=[f"Pred_{cls}" for cls in target_names]
    )
    cm_df.to_csv(os.path.join(out_dir, f"{model_name}_confusion_matrix_{dataset_name}.csv"))


def save_predictions(y_true, y_pred, y_prob, original_indices, target_names, label_encoder,
                     model_name, dataset_name, out_dir):
    pred_df = pd.DataFrame({
        "Original_index": original_indices,
        "True_class": label_encoder.inverse_transform(y_true),
        "Predicted_class": label_encoder.inverse_transform(y_pred),
    })
    for i, class_name in enumerate(target_names):
        pred_df[f"Probability_{class_name}"] = y_prob[:, i]
    pred_df.to_csv(os.path.join(out_dir, f"{model_name}_predictions_{dataset_name}.csv"), index=False)


def cv_weighted_f1(model_builder, X_train, y_train, cv, use_sample_weight=False):
    scores = []
    for train_fold_idx, valid_fold_idx in cv.split(X_train, y_train):
        X_tr = X_train.iloc[train_fold_idx]
        X_va = X_train.iloc[valid_fold_idx]
        y_tr = y_train[train_fold_idx]
        y_va = y_train[valid_fold_idx]
        model = model_builder()
        if use_sample_weight:
            sw = compute_sample_weight(class_weight="balanced", y=y_tr)
            model.fit(X_tr, y_tr, sample_weight=sw)
        else:
            model.fit(X_tr, y_tr)
        pred = model.predict(X_va)
        scores.append(f1_score(y_va, pred, average="weighted", zero_division=0))
    return float(np.mean(scores))


def align_validation_features(X_val, train_features, dataset_name):
    missing = sorted(set(train_features) - set(X_val.columns))
    extra = sorted(set(X_val.columns) - set(train_features))
    print(f"{dataset_name}: missing features = {len(missing)}, extra features = {len(extra)}")
    if missing:
        raise ValueError(f"{dataset_name} misses training features. First missing: {missing[:10]}")
    return X_val[train_features].copy()


def encode_validation_labels(y_val, label_encoder, dataset_name):
    known = set(label_encoder.classes_)
    current = set(y_val.unique())
    unknown = sorted(current - known)
    if unknown:
        raise ValueError(f"{dataset_name} contains unknown classes: {unknown}")
    return label_encoder.transform(y_val)


def evaluate_model_on_dataset(model, model_name, X_eval, y_eval_enc, original_indices,
                              dataset_name, target_names, label_encoder, out_dir):
    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval)
    metrics = calculate_metrics(y_eval_enc, y_pred, dataset_name, model_name)
    metrics.update(bootstrap_confidence_intervals(y_eval_enc, y_pred))
    save_report_and_cm(y_eval_enc, y_pred, target_names, model_name, dataset_name, out_dir)
    save_predictions(y_eval_enc, y_pred, y_prob, original_indices, target_names,
                     label_encoder, model_name, dataset_name, out_dir)
    return metrics, y_pred, y_prob


def weighted_soft_vote(probabilities_by_model, model_order, weights):
    ensemble_prob = np.zeros_like(probabilities_by_model[model_order[0]])
    for w, model_name in zip(weights, model_order):
        ensemble_prob += w * probabilities_by_model[model_name]
    ensemble_pred = np.argmax(ensemble_prob, axis=1)
    return ensemble_pred, ensemble_prob

# ============================================================
# 3) Load TCGA data and standardize expression profiles before train/test split
# ============================================================

data = read_dataset(DATA_PATH, CLASS_COL)
print("TCGA data shape:", data.shape)
print("TCGA class distribution:")
print(data[CLASS_COL].value_counts())

X, y = split_xy(data, CLASS_COL)
feature_names = list(X.columns)
pd.Series(feature_names, name="Gene").to_csv(os.path.join(OUT_DIR, "feature_names.csv"), index=False)
print("Feature matrix before standardization:", X.shape)

# Standardize TCGA expression profiles before train/test split.
# Only gene-expression columns are scaled; Class is already separated in y.
X = standardize_expression_profiles(X, "TCGA")
print("Feature matrix after standardization:", X.shape)

sample_indices = np.arange(data.shape[0])
train_idx, test_idx = train_test_split(
    sample_indices, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()
y_train = y.iloc[train_idx].copy()
y_test = y.iloc[test_idx].copy()

label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_test_enc = label_encoder.transform(y_test)
target_names = label_encoder.classes_
n_classes = len(target_names)

print("Encoded classes:")
for i, cls in enumerate(target_names):
    print(i, "->", cls)

pd.DataFrame({
    "Original_index": np.concatenate([train_idx, test_idx]),
    "Subset": ["train"] * len(train_idx) + ["test"] * len(test_idx),
    "Class": pd.concat([y_train, y_test]).values,
}).to_csv(os.path.join(OUT_DIR, "train_test_split_info.csv"), index=False)

save_json({cls: int(i) for i, cls in enumerate(target_names)}, os.path.join(OUT_DIR, "label_mapping.json"))
joblib.dump(label_encoder, os.path.join(OUT_DIR, "label_encoder.joblib"))

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# ============================================================
# 4) Random Forest optimization and fitting
# ============================================================

def rf_cv(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, max_leaf_nodes):
    def builder():
        return RandomForestClassifier(
            n_estimators=int(round(n_estimators)), max_depth=int(round(max_depth)),
            min_samples_split=int(round(min_samples_split)), min_samples_leaf=int(round(min_samples_leaf)),
            max_features=float(max_features), max_leaf_nodes=int(round(max_leaf_nodes)),
            class_weight="balanced", bootstrap=True, criterion="gini",
            random_state=RANDOM_STATE, n_jobs=-1
        )
    return cv_weighted_f1(builder, X_train, y_train_enc, cv, use_sample_weight=False)

print("Running Bayesian optimization for Random Forest...")
optimizer_rf = BayesianOptimization(
    f=rf_cv,
    pbounds={
        "n_estimators": (100, 600), "max_depth": (3, 45),
        "min_samples_split": (2, 20), "min_samples_leaf": (1, 10),
        "max_features": (0.03, 0.80), "max_leaf_nodes": (10, 500),
    },
    random_state=RANDOM_STATE, verbose=2
)
optimizer_rf.maximize(init_points=INIT_POINTS_RF, n_iter=N_ITER_RF)

pd.DataFrame([
    {"iteration": i + 1, "target_f1_weighted_cv": r["target"],
     "n_estimators": int(round(r["params"]["n_estimators"])),
     "max_depth": int(round(r["params"]["max_depth"])),
     "min_samples_split": int(round(r["params"]["min_samples_split"])),
     "min_samples_leaf": int(round(r["params"]["min_samples_leaf"])),
     "max_features": float(r["params"]["max_features"]),
     "max_leaf_nodes": int(round(r["params"]["max_leaf_nodes"]))}
    for i, r in enumerate(optimizer_rf.res)
]).to_csv(os.path.join(OUT_DIR, "RF_bayesian_optimization_history.csv"), index=False)

raw = optimizer_rf.max["params"]
best_rf_params = {
    "n_estimators": int(round(raw["n_estimators"])),
    "max_depth": int(round(raw["max_depth"])),
    "min_samples_split": int(round(raw["min_samples_split"])),
    "min_samples_leaf": int(round(raw["min_samples_leaf"])),
    "max_features": float(raw["max_features"]),
    "max_leaf_nodes": int(round(raw["max_leaf_nodes"])),
    "class_weight": "balanced", "bootstrap": True, "criterion": "gini",
    "best_cv_f1_weighted": float(optimizer_rf.max["target"]),
}
save_json(best_rf_params, os.path.join(OUT_DIR, "RF_best_hyperparameters.json"))

rf_model = RandomForestClassifier(
    n_estimators=best_rf_params["n_estimators"], max_depth=best_rf_params["max_depth"],
    min_samples_split=best_rf_params["min_samples_split"], min_samples_leaf=best_rf_params["min_samples_leaf"],
    max_features=best_rf_params["max_features"], max_leaf_nodes=best_rf_params["max_leaf_nodes"],
    class_weight="balanced", bootstrap=True, criterion="gini", random_state=RANDOM_STATE, n_jobs=-1
)
rf_model.fit(X_train, y_train_enc)

# ============================================================
# 5) XGBoost optimization and fitting
# ============================================================

def xgb_cv(n_estimators, max_depth, learning_rate, subsample, colsample_bytree,
           min_child_weight, gamma, reg_alpha, reg_lambda):
    def builder():
        return XGBClassifier(
            objective="multi:softprob", num_class=n_classes,
            n_estimators=int(round(n_estimators)), max_depth=int(round(max_depth)),
            learning_rate=float(learning_rate), subsample=float(subsample),
            colsample_bytree=float(colsample_bytree), min_child_weight=float(min_child_weight),
            gamma=float(gamma), reg_alpha=float(reg_alpha), reg_lambda=float(reg_lambda),
            eval_metric="mlogloss", tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1
        )
    return cv_weighted_f1(builder, X_train, y_train_enc, cv, use_sample_weight=True)

print("Running Bayesian optimization for XGBoost...")
optimizer_xgb = BayesianOptimization(
    f=xgb_cv,
    pbounds={
        "n_estimators": (100, 800), "max_depth": (2, 10),
        "learning_rate": (0.01, 0.25), "subsample": (0.55, 1.00),
        "colsample_bytree": (0.05, 0.80), "min_child_weight": (1.0, 15.0),
        "gamma": (0.0, 5.0), "reg_alpha": (0.0, 5.0), "reg_lambda": (0.1, 10.0),
    },
    random_state=RANDOM_STATE, verbose=2
)
optimizer_xgb.maximize(init_points=INIT_POINTS_XGB, n_iter=N_ITER_XGB)

pd.DataFrame([
    {"iteration": i + 1, "target_f1_weighted_cv": r["target"],
     "n_estimators": int(round(r["params"]["n_estimators"])),
     "max_depth": int(round(r["params"]["max_depth"])),
     "learning_rate": float(r["params"]["learning_rate"]),
     "subsample": float(r["params"]["subsample"]),
     "colsample_bytree": float(r["params"]["colsample_bytree"]),
     "min_child_weight": float(r["params"]["min_child_weight"]),
     "gamma": float(r["params"]["gamma"]),
     "reg_alpha": float(r["params"]["reg_alpha"]),
     "reg_lambda": float(r["params"]["reg_lambda"])}
    for i, r in enumerate(optimizer_xgb.res)
]).to_csv(os.path.join(OUT_DIR, "XGB_bayesian_optimization_history.csv"), index=False)

raw = optimizer_xgb.max["params"]
best_xgb_params = {
    "n_estimators": int(round(raw["n_estimators"])),
    "max_depth": int(round(raw["max_depth"])),
    "learning_rate": float(raw["learning_rate"]),
    "subsample": float(raw["subsample"]),
    "colsample_bytree": float(raw["colsample_bytree"]),
    "min_child_weight": float(raw["min_child_weight"]),
    "gamma": float(raw["gamma"]),
    "reg_alpha": float(raw["reg_alpha"]),
    "reg_lambda": float(raw["reg_lambda"]),
    "objective": "multi:softprob", "eval_metric": "mlogloss",
    "best_cv_f1_weighted": float(optimizer_xgb.max["target"]),
}
save_json(best_xgb_params, os.path.join(OUT_DIR, "XGB_best_hyperparameters.json"))

xgb_model = XGBClassifier(
    objective="multi:softprob", num_class=n_classes,
    n_estimators=best_xgb_params["n_estimators"], max_depth=best_xgb_params["max_depth"],
    learning_rate=best_xgb_params["learning_rate"], subsample=best_xgb_params["subsample"],
    colsample_bytree=best_xgb_params["colsample_bytree"],
    min_child_weight=best_xgb_params["min_child_weight"], gamma=best_xgb_params["gamma"],
    reg_alpha=best_xgb_params["reg_alpha"], reg_lambda=best_xgb_params["reg_lambda"],
    eval_metric="mlogloss", tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1
)
xgb_model.fit(X_train, y_train_enc, sample_weight=compute_sample_weight(class_weight="balanced", y=y_train_enc))

# ============================================================
# 6) Train/test evaluation and voting weights
# ============================================================

models = {"RF": rf_model, "XGB": xgb_model}
model_order = ["RF", "XGB"]
cv_scores = {"RF": float(optimizer_rf.max["target"]), "XGB": float(optimizer_xgb.max["target"])}
raw_weights = np.maximum(np.array([cv_scores[m] for m in model_order], dtype=float), 1e-8)
weights = raw_weights / raw_weights.sum()

pd.DataFrame({
    "Model": model_order,
    "CV_F1_weighted": raw_weights,
    "Soft_voting_weight": weights,
}).to_csv(os.path.join(OUT_DIR, "weighted_soft_voting_weights.csv"), index=False)

all_metrics = []
train_test_prob = {"train": {}, "test": {}}

for model_name, model in models.items():
    for dataset_name, X_eval, y_eval, idx in [
        ("Train", X_train, y_train_enc, train_idx),
        ("Test", X_test, y_test_enc, test_idx),
    ]:
        metrics, pred, prob = evaluate_model_on_dataset(
            model, model_name, X_eval, y_eval, idx, dataset_name.lower(),
            target_names, label_encoder, OUT_DIR
        )
        metrics["Dataset"] = dataset_name
        all_metrics.append(metrics)
        train_test_prob[dataset_name.lower()][model_name] = prob

for dataset_name, y_eval, idx in [("Train", y_train_enc, train_idx), ("Test", y_test_enc, test_idx)]:
    pred, prob = weighted_soft_vote(train_test_prob[dataset_name.lower()], model_order, weights)
    ens_train_test_metrics = calculate_metrics(y_eval, pred, dataset_name, "WeightedSoftVoting")
    ens_train_test_metrics.update(bootstrap_confidence_intervals(y_eval, pred))
    all_metrics.append(ens_train_test_metrics)
    save_report_and_cm(y_eval, pred, target_names, "WeightedSoftVoting", dataset_name.lower(), OUT_DIR)
    save_predictions(y_eval, pred, prob, idx, target_names, label_encoder, "WeightedSoftVoting", dataset_name.lower(), OUT_DIR)

# ============================================================
# 7) External validation: each dataset
# ============================================================

validation_metrics = []
validation_summary = []

for dataset_name, file_path in VALIDATION_FILES.items():
    print(f"Processing external validation dataset: {dataset_name}")
    val_data = read_dataset(file_path, CLASS_COL)
    print(val_data.shape)
    print(val_data[CLASS_COL].value_counts())

    X_val, y_val = split_xy(val_data, CLASS_COL)
    X_val = align_validation_features(X_val, feature_names, dataset_name)
    X_val = standardize_expression_profiles(X_val, dataset_name)
    y_val_enc = encode_validation_labels(y_val, label_encoder, dataset_name)

    prob_by_model = {}
    for model_name, model in models.items():
        metrics, pred, prob = evaluate_model_on_dataset(
            model, model_name, X_val, y_val_enc, np.arange(val_data.shape[0]),
            f"Validation_{dataset_name}", target_names, label_encoder, OUT_DIR
        )
        validation_metrics.append(metrics)
        prob_by_model[model_name] = prob

    pred, prob = weighted_soft_vote(prob_by_model, model_order, weights)
    ens_metrics = calculate_metrics(y_val_enc, pred, f"Validation_{dataset_name}", "WeightedSoftVoting")
    ens_metrics.update(bootstrap_confidence_intervals(y_val_enc, pred))
    validation_metrics.append(ens_metrics)
    save_report_and_cm(y_val_enc, pred, target_names, "WeightedSoftVoting", f"validation_{dataset_name}", OUT_DIR)
    save_predictions(y_val_enc, pred, prob, np.arange(val_data.shape[0]), target_names,
                     label_encoder, "WeightedSoftVoting", f"validation_{dataset_name}", OUT_DIR)

    validation_summary.append({
        "Dataset": dataset_name,
        "Samples": int(val_data.shape[0]),
        "Features": int(X_val.shape[1]),
        "Classes": ";".join(sorted(y_val.unique())),
        "WeightedSoftVoting_F1_weighted": float(ens_metrics["F1_weighted"]),
        "WeightedSoftVoting_Balanced_Accuracy": float(ens_metrics["Balanced_Accuracy"]),
        "WeightedSoftVoting_MCC": float(ens_metrics["MCC"]),
        "WeightedSoftVoting_F1_weighted_CI_low": float(ens_metrics["F1_weighted_CI_low"]),
        "WeightedSoftVoting_F1_weighted_CI_high": float(ens_metrics["F1_weighted_CI_high"]),
        "WeightedSoftVoting_Balanced_Accuracy_CI_low": float(ens_metrics["Balanced_Accuracy_CI_low"]),
        "WeightedSoftVoting_Balanced_Accuracy_CI_high": float(ens_metrics["Balanced_Accuracy_CI_high"]),
        "WeightedSoftVoting_MCC_CI_low": float(ens_metrics["MCC_CI_low"]),
        "WeightedSoftVoting_MCC_CI_high": float(ens_metrics["MCC_CI_high"]),
    })

# ============================================================
# 8) Combined external validation
# ============================================================

if os.path.exists(COMBINED_VALIDATION_FILE):
    combined_data = read_dataset(COMBINED_VALIDATION_FILE, CLASS_COL)
else:
    frames = []
    for dataset_name, file_path in VALIDATION_FILES.items():
        tmp = read_dataset(file_path, CLASS_COL)
        tmp["Validation_dataset"] = dataset_name
        tmp = tmp[[c for c in tmp.columns if c != CLASS_COL] + [CLASS_COL]]
        frames.append(tmp)
    combined_data = pd.concat(frames, axis=0, ignore_index=True)
    combined_data.to_csv(COMBINED_VALIDATION_FILE, index=False)

print("Combined validation:", combined_data.shape)
print(combined_data[CLASS_COL].value_counts())

X_comb, y_comb = split_xy(combined_data, CLASS_COL)
X_comb = align_validation_features(X_comb, feature_names, "ALL_VALIDATION")
X_comb = standardize_expression_profiles(X_comb, "ALL_VALIDATION")
y_comb_enc = encode_validation_labels(y_comb, label_encoder, "ALL_VALIDATION")

combined_prob_by_model = {}
for model_name, model in models.items():
    metrics, pred, prob = evaluate_model_on_dataset(
        model, model_name, X_comb, y_comb_enc, np.arange(combined_data.shape[0]),
        "Validation_ALL", target_names, label_encoder, OUT_DIR
    )
    validation_metrics.append(metrics)
    combined_prob_by_model[model_name] = prob

pred, prob = weighted_soft_vote(combined_prob_by_model, model_order, weights)
combined_ens_metrics = calculate_metrics(y_comb_enc, pred, "Validation_ALL", "WeightedSoftVoting")
combined_ens_metrics.update(bootstrap_confidence_intervals(y_comb_enc, pred))
validation_metrics.append(combined_ens_metrics)
save_report_and_cm(y_comb_enc, pred, target_names, "WeightedSoftVoting", "validation_ALL", OUT_DIR)
save_predictions(y_comb_enc, pred, prob, np.arange(combined_data.shape[0]), target_names,
                 label_encoder, "WeightedSoftVoting", "validation_ALL", OUT_DIR)

# ============================================================
# 9) Save metrics, feature importance, models, summary
# ============================================================

train_test_metrics_df = pd.DataFrame(all_metrics)
validation_metrics_df = pd.DataFrame(validation_metrics)
all_metrics_df = pd.concat([train_test_metrics_df, validation_metrics_df], axis=0, ignore_index=True)

train_test_metrics_df.to_csv(os.path.join(OUT_DIR, "train_test_classification_metrics.csv"), index=False)
validation_metrics_df.to_csv(os.path.join(OUT_DIR, "external_validation_classification_metrics.csv"), index=False)
all_metrics_df.to_csv(os.path.join(OUT_DIR, "ALL_classification_metrics.csv"), index=False)
pd.DataFrame(validation_summary).to_csv(os.path.join(OUT_DIR, "external_validation_summary_weighted_soft_voting.csv"), index=False)

pd.DataFrame({"Gene": feature_names, "Importance": rf_model.feature_importances_}).sort_values(
    "Importance", ascending=False
).to_csv(os.path.join(OUT_DIR, "RF_feature_importances.csv"), index=False)

pd.DataFrame({"Gene": feature_names, "Importance": xgb_model.feature_importances_}).sort_values(
    "Importance", ascending=False
).to_csv(os.path.join(OUT_DIR, "XGB_feature_importances.csv"), index=False)

joblib.dump(rf_model, os.path.join(OUT_DIR, "RF_model.joblib"))
joblib.dump(xgb_model, os.path.join(OUT_DIR, "XGB_model.joblib"))

ensemble_info = {
    "type": "weighted_soft_voting",
    "models": model_order,
    "weights": {m: float(w) for m, w in zip(model_order, weights)},
    "weight_source": "5-fold CV weighted F1 after Bayesian hyperparameter optimization",
}
save_json(ensemble_info, os.path.join(OUT_DIR, "weighted_soft_voting_ensemble_info.json"))

summary = {
    "dataset": DATA_PATH,
    "analysis_type": "TCGA multiclass classification on common genes with per-dataset StandardScaler normalization; RF + XGB only",
    "class_column": CLASS_COL,
    "n_samples_total": int(data.shape[0]),
    "n_features": int(X.shape[1]),
    "n_classes": int(n_classes),
    "classes": list(target_names),
    "test_size": TEST_SIZE,
    "cv": f"StratifiedKFold(n_splits={N_SPLITS}, shuffle=True, random_state={RANDOM_STATE})",
    "optimization": "Bayesian Optimization",
    "optimization_scoring": "weighted F1",
    "models": {"RF": best_rf_params, "XGB": best_xgb_params},
    "weighted_soft_voting_weights": ensemble_info["weights"],
    "validation_files": VALIDATION_FILES,
    "combined_validation_file": COMBINED_VALIDATION_FILE,
    "standardization": {
        "method": "sklearn.preprocessing.StandardScaler",
        "strategy": "Each dataset is standardized independently. TCGA is standardized before train/test split; each external validation dataset is standardized separately. Only gene-expression columns are standardized; the Class column is excluded.",
    },
    "confidence_intervals": {
        "method": "non-parametric bootstrap over samples with replacement",
        "iterations": BOOTSTRAP_ITERATIONS,
        "ci_level": CI_LEVEL,
        "metrics": ["Accuracy", "Balanced_Accuracy", "Precision_macro", "Recall_macro", "F1_macro", "Precision_weighted", "Recall_weighted", "F1_weighted", "MCC"],
    },
    "best_test_model_by_f1_weighted": all_metrics_df[all_metrics_df["Dataset"] == "Test"].sort_values(
        "F1_weighted", ascending=False
    ).iloc[0].to_dict(),
    "best_combined_validation_model_by_f1_weighted": all_metrics_df[all_metrics_df["Dataset"] == "Validation_ALL"].sort_values(
        "F1_weighted", ascending=False
    ).iloc[0].to_dict(),
}
save_json(summary, os.path.join(OUT_DIR, "modeling_summary.json"))

print("Modeling completed successfully.")
print("Results directory:", OUT_DIR)
print("Best test result by weighted F1:")
print(summary["best_test_model_by_f1_weighted"])
print("Best combined validation result by weighted F1:")
print(summary["best_combined_validation_model_by_f1_weighted"])
print(f"Bootstrap confidence intervals: {CI_LEVEL*100:.1f}% CI, {BOOTSTRAP_ITERATIONS} iterations.")
