"""CF_STACK: complete feature space + genuine OOF RF/XGB stacking."""

from ablation_common import Settings, run_complete

DATA_PATH = "Common_genes/tcga_cancer_common_genes.csv"
VALIDATION_FILES = {
    "BLCA": "Common_genes/BLCA_validation_common_genes.csv",
    "BRCA": "Common_genes/BRCA_validation_common_genes.csv",
    "COAD": "Common_genes/COAD_validation_common_genes.csv",
    "HNSC": "Common_genes/HNSC_validation_common_genes.csv",
    "KIRC": "Common_genes/KIRC_validation_common_genes.csv",
    "LGG": "Common_genes/LGG_validation_common_genes.csv",
    "LUAD": "Common_genes/LUAD_validation_common_genes.csv",
    "LUSC": "Common_genes/LUSC_validation_common_genes.csv",
}

if __name__ == "__main__":
    run_complete(
        scenario="CF_STACK",
        use_stacking=True,
        data_path=DATA_PATH,
        validation_files=VALIDATION_FILES,
        output_dir="results/CF_STACK",
        settings=Settings(n_jobs=12),
    )
