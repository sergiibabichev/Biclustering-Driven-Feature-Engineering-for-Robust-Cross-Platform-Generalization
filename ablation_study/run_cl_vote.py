"""CL_VOTE: cluster spaces + two-level gene-count/OOF-quality soft voting."""

from ablation_common import Settings, run_clustered

CLUSTER_FILES = {
    "Cluster_1": {
        "train_test": "Cancer_Cluster_1_train_test.csv",
        "validation": "Cancer_Cluster_1_validation.csv",
    },
    "Cluster_2": {
        "train_test": "Cancer_Cluster_2_train_test.csv",
        "validation": "Cancer_Cluster_2_validation.csv",
    },
}

if __name__ == "__main__":
    run_clustered(
        scenario="CL_VOTE",
        use_stacking=False,
        cluster_files=CLUSTER_FILES,
        output_dir="results/CL_VOTE",
        settings=Settings(n_jobs=12),
    )
