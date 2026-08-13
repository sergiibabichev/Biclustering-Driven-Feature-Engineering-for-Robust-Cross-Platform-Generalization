# Biclustering-Driven Feature Engineering for Robust Cross-Platform Generalization in Transcriptomic Cancer Classification

This repository contains the R and Python code used to construct, filter, cluster, and classify the transcriptomic datasets analysed in Chapter 5 of the dissertation and in the associated manuscript. The workflow combines gene-level preprocessing, Gene Ontology (GO)-based filtering, ensemble biclustering, SOTA-based consensus clustering, RF/XGBoost classification, external validation, and a four-scenario ablation study.

The expression matrices are **not redistributed in this repository** because of their size. All source data are publicly available from the repositories listed below. After downloading them, retain the filenames expected by the scripts or update the input paths at the beginning of the relevant script.

## Repository structure

```text
.
├── Data_TCGA_Cancer_formation.R
├── External_BLCA_Formation.R
├── External_BRCA_Formation.R
├── External_COAD_Formation.R
├── External_HNSC_Formation.R
├── External_KIRC_Formation.R
├── External_LGG_Formation.R
├── External_LUAD_Formation.R
├── External_LUSC_Formation.R
├── GO_filtering.R
├── BC_filtering.R
├── UN_datasets.R
├── SOTA_Spectral_Clustering.R
├── tcga_common_external_validation_rf_xgb_stand.py
├── tcga_common_external_validation_rf_xgb_with_normal_combined_only_CI.py
├── tcga_common_external_validation_rf_xgb_with_normal_combined_scaled_CI.py
└── ablation_study/
    ├── ablation_common.py
    ├── run_cf_vote.py
    ├── run_cf_stack.py
    ├── run_cl_vote.py
    ├── run_cl_stack.py
    ├── requirements.txt
    └── README_UA.md
```

## Public data sources

| Cohort | Source and download page | Raw file(s) expected by the code |
|---|---|---|
| TCGA pan-cancer | [NCI Genomic Data Commons](https://portal.gdc.cancer.gov/) (TCGA projects; gene-expression quantification and phenotype/sample metadata) | A harmonised sample-by-gene matrix named `tcga_cancer.csv` after downloading and assembling the selected TCGA projects |
| BLCA (UROMOL) | [ArrayExpress/BioStudies E-MTAB-4321](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-4321) | `UROMOL_gene_fpkm_gtf.txt` |
| BRCA (CPTAC) | [LinkedOmics CPTAC-BRCA](https://www.linkedomics.org/data_download/CPTAC-BRCA/) | `BRCA_RNAseq_gene_RSEM_coding_UQ_1500_log2_Tumor.txt` |
| COAD (CPTAC) | [LinkedOmics CPTAC-COAD](https://www.linkedomics.org/data_download/CPTAC-COAD/) | `COAD_RNAseq_gene_RSEM_coding_UQ_1500_log2_Tumor.txt` |
| HNSC (CPTAC) | [LinkedOmics CPTAC-HNSCC](https://www.linkedomics.org/data_download/CPTAC-HNSCC/) | `HNSCC_RNAseq_gene_RSEM_coding_UQ_1500_log2_Tumor.txt`; `HNSCC_RNAseq_gene_RSEM_coding_UQ_1500_log2_Normal.txt` |
| KIRC (CPTAC) | [LinkedOmics CPTAC-CCRCC](https://www.linkedomics.org/data_download/CPTAC-CCRCC/) | `CCRCC_RNAseq_gene_RSEM_coding_UQ_1500_log2_Tumor.txt`; `CCRCC_RNAseq_gene_RSEM_coding_UQ_1500_log2_Normal.txt` |
| LGG (CGGA) | [CGGA download portal](https://www.cgga.org.cn/download.jsp), dataset `mRNAseq_693` | `CGGA.mRNAseq_693.RSEM-genes.20190909.txt`; `CGGA.mRNAseq_693.clinical.20190909.txt` |
| LUAD (CPTAC) | [LinkedOmics CPTAC-LUAD](https://www.linkedomics.org/data_download/CPTAC-LUAD/) | `LUAD_RNAseq_gene_RSEM_coding_UQ_1500_log2_Tumor.txt`; `LUAD_RNAseq_gene_RSEM_coding_UQ_1500_log2_Normal.txt` |
| LUSC (CPTAC) | [LinkedOmics CPTAC-LSCC](https://www.linkedomics.org/data_download/CPTAC-LSCC/) | `LSCC_RNAseq_gene_RSEM_coding_UQ_1500_log2_Tumor.txt`; `LSCC_RNAseq_gene_RSEM_coding_UQ_1500_log2_Normal.txt` |

The data remain subject to the access conditions, citation requirements, and licences of their original repositories. No controlled-access data are included here.

## Software requirements

R (version 4.2 or later recommended) with the following packages:

```r
install.packages(c(
  "data.table", "dplyr", "ggplot2", "patchwork", "isa2", "PMA",
  "clValid", "kernlab", "Matrix", "readr", "gridExtra", "future",
  "future.apply", "BiocManager"
))
BiocManager::install(c(
  "AnnotationDbi", "org.Hs.eg.db", "clusterProfiler", "enrichplot"
))
```

Python 3.10 or later is recommended. Install the dependencies with:

```bash
python -m pip install numpy pandas scikit-learn xgboost bayesian-optimization joblib
```

The SOTA procedure and the classification/ablation experiments are computationally intensive. Adjust `n_workers` in `SOTA_Spectral_Clustering.R` and `n_jobs` in the Python configuration to the available CPU cores and memory.

## Reproduction workflow

Run all commands from the repository root unless stated otherwise. The scripts use relative paths and write their results into the current working directory.

### 1. Construct the TCGA analysis matrix

Download and assemble the selected TCGA cohorts as a sample-by-gene CSV matrix. Gene columns must use gene identifiers, and the final column must be named `Class`. Save it as:

```text
tcga_cancer.csv
```

Then run:

```bash
Rscript Data_TCGA_Cancer_formation.R
```

This removes the `normal` class for the feature-selection stage and creates `tcga_cancer_without_normal.csv` plus a class-distribution figure.

### 2. Apply statistical and GO-based filtering

```bash
Rscript GO_filtering.R
```

The script checks expression-profile normality, performs gene-wise Wilcoxon testing with multiple-testing correction, conducts BP/MF/CC enrichment, and retains the union of genes represented in the significant GO categories. Its principal output for the next stage is:

```text
TCGA_GO_filtered_data.csv
```

Additional Wilcoxon tables, GO enrichment tables, and diagnostic plots are also generated.

### 3. Apply ensemble biclustering and frequency filtering

```bash
Rscript BC_filtering.R
```

The script repeatedly applies ISA and PMD to the GO-filtered expression matrix, estimates gene participation frequencies across 50 runs, and retains genes meeting the configured hit threshold. The principal output is:

```text
tcga_BC_reduced.csv
```

The class column is excluded from the biclustering calculations and is reattached only when the reduced matrix is formed.

### 4. Prepare external validation cohorts

Place each downloaded source file beside its corresponding R script and run:

```bash
Rscript External_BLCA_Formation.R
Rscript External_BRCA_Formation.R
Rscript External_COAD_Formation.R
Rscript External_HNSC_Formation.R
Rscript External_KIRC_Formation.R
Rscript External_LGG_Formation.R
Rscript External_LUAD_Formation.R
Rscript External_LUSC_Formation.R
```

These scripts convert source-specific identifiers to Entrez Gene IDs, collapse duplicate mappings by their mean, transform the UROMOL BLCA FPKM values as `log2(FPKM + 1)`, retain the already log-transformed CPTAC values, transpose the matrices to samples by genes, and append `Class`.

Expected outputs are `UROMOL_BLCA.csv`, `CPTAC_BRCA.csv`, `CPTAC_COAD.csv`, `CPTAC_HNSC_NORMAL.csv`, `CPTAC_kirc_NORMAL.csv`, `CGGA_mRNAseq_693_LGG.csv`, `CPTAC_luad_NORMAL.csv`, and `CPTAC_lusc_NORMAL.csv`.

### 5. Harmonise gene identifiers and construct common-gene matrices

The `UN_datasets.R` script automates the common-gene harmonisation step. It:

- removes the R-generated `X` prefix from numeric gene identifiers;
- excludes `Class` and, when present, `SampleID` from the gene set;
- calculates the intersection of genes across the reduced TCGA matrix and all eight external cohorts;
- preserves the TCGA gene order;
- aligns every matrix to the same ordered feature set;
- concatenates all external cohorts into one validation matrix.

Before running it, place `tcga_BC_reduced.csv` in the repository root and rename or copy the processed external-cohort outputs to the filenames expected by the script:

| Processed output | Input name expected by `UN_datasets.R` |
|---|---|
| `UROMOL_BLCA.csv` | `BLCA_validation.csv` |
| `CPTAC_BRCA.csv` | `BRCA_validation.csv` |
| `CPTAC_COAD.csv` | `COAD_validation.csv` |
| `CPTAC_HNSC_NORMAL.csv` | `HNSC_validation.csv` |
| `CPTAC_kirc_NORMAL.csv` | `KIRC_validation.csv` |
| `CGGA_mRNAseq_693_LGG.csv` | `LGG_validation.csv` |
| `CPTAC_luad_NORMAL.csv` | `LUAD_validation.csv` |
| `CPTAC_lusc_NORMAL.csv` | `LUSC_validation.csv` |

Run:

```bash
Rscript UN_datasets.R
```

The script creates the following files in the current working directory:

```text
tcga_cancer_common_genes.csv
BLCA_validation_common_genes.csv
BRCA_validation_common_genes.csv
COAD_validation_common_genes.csv
HNSC_validation_common_genes.csv
KIRC_validation_common_genes.csv
LGG_validation_common_genes.csv
LUAD_validation_common_genes.csv
LUSC_validation_common_genes.csv
ALL_validation_common_genes.csv
```

Create a `Common_genes` directory and copy `tcga_cancer_common_genes.csv` plus the eight cohort-specific common-gene files into it for the Python classification and complete-feature ablation scripts. Keep copies of `tcga_cancer_common_genes.csv` and the combined validation matrix in the repository root for SOTA. Because `SOTA_Spectral_Clustering.R` currently requests `All_validation_common_genes.csv`, either rename the root copy of `ALL_validation_common_genes.csv` accordingly or change line 4 of the SOTA script. Filenames are case-sensitive on Linux.

### 6. Run SOTA consensus clustering

Keep the `tcga_cancer_common_genes.csv` and `All_validation_common_genes.csv` copies prepared in Step 5 in the repository root (or change their paths in the script), then run:

```bash
Rscript SOTA_Spectral_Clustering.R
```

The script standardises gene profiles, performs 50 bootstrap SOTA runs on random gene subsets, constructs a consensus matrix, applies spectral clustering, conducts cluster-wise GO enrichment, and creates cluster-specific training and validation matrices. Results are written mainly to `SOTA_results_new/`, including:

- gene-to-cluster assignments and cluster summaries;
- bootstrap cluster counts and the consensus matrix;
- GO enrichment plots;
- `TCGA_Cluster_1_train.csv`, `TCGA_Cluster_2_train.csv`, etc.;
- corresponding validation matrices.

### 7. Train RF/XGBoost models and perform external validation

The three top-level Python scripts implement related experimental variants. All expect the `Common_genes` directory described above.

- `tcga_common_external_validation_rf_xgb_stand.py`: standard workflow with per-cohort external validation and a combined validation matrix.
- `tcga_common_external_validation_rf_xgb_with_normal_combined_only_CI.py`: combined external validation with 95% bootstrap confidence intervals.
- `tcga_common_external_validation_rf_xgb_with_normal_combined_scaled_CI.py`: combined external validation standardised once as a single matrix, with 95% bootstrap confidence intervals.

Run the variant corresponding to the reported experiment, for example:

```bash
python tcga_common_external_validation_rf_xgb_with_normal_combined_scaled_CI.py
```

Each script performs a stratified 70/30 TCGA split, Bayesian hyperparameter optimisation on the training portion, final RF and XGBoost fitting, weighted soft voting, held-out TCGA evaluation, and external validation. Result directories contain metrics, confidence intervals where applicable, predictions, confusion matrices, feature importances, fitted models, hyperparameters, ensemble weights, and run summaries.

### 8. Run the ablation study

Read `ablation_study/README_UA.md` for the detailed design. The four scenarios are:

| Script | Feature representation | Combination strategy |
|---|---|---|
| `run_cf_vote.py` | Complete/common feature space | Weighted RF/XGBoost soft voting |
| `run_cf_stack.py` | Complete/common feature space | OOF stacking |
| `run_cl_vote.py` | SOTA-derived cluster spaces | Two-level weighted soft voting |
| `run_cl_stack.py` | SOTA-derived cluster spaces | OOF cross-cluster stacking |

The run files currently use paths relative to `ablation_study`. Place or link `Common_genes` and the required cluster CSV files there, or edit the path constants. From that directory, execute the scenarios separately:

```bash
cd ablation_study
python run_cf_vote.py
python run_cf_stack.py
python run_cl_vote.py
python run_cl_stack.py
```

The scripts may be run in parallel only if sufficient CPU and memory are available. Their default `n_jobs=12` setting can consume up to 48 threads when all four are launched simultaneously.

## Reproducibility and leakage control

- Random seeds are fixed in the supplied scripts where stochastic procedures are used.
- During classification, the TCGA data are split stratifiably into training and held-out test subsets.
- Bayesian hyperparameter optimisation, voting weights, and OOF stacking meta-models use only the TCGA training subset.
- External cohort labels are used for final performance evaluation, not for model fitting or hyperparameter selection.
- In the biclustering stage, `Class` is removed before ISA/PMD and reattached after gene selection.

The supplied scripts preserve the exact preprocessing and standardisation variants used in the reported experiments. Because some variants standardise a complete matrix before splitting, users wishing to evaluate a strictly deployment-oriented pipeline should fit any scaling parameters on the training subset only and apply the fitted transformation unchanged to test data.

## Data availability

The code required to reproduce the analyses is provided in this repository. The underlying transcriptomic datasets are not redistributed because of their size. They can be downloaded from the NCI Genomic Data Commons, ArrayExpress/BioStudies, LinkedOmics/CPTAC, and CGGA using the links and source filenames provided in the **Public data sources** section. Users are responsible for complying with the source repositories' terms of access and citation requirements.

## Citation

If you use this code, please cite the associated dissertation/manuscript and the archived release of this repository. Add the final publication citation and Zenodo DOI here after they become available:

```text
Author(s). Title. Journal/Institution, year. DOI: [to be added]
Code archive. Zenodo. DOI: [to be added]
```

## Licence

No software licence file is currently included. Before public release, add an explicit licence (for example, MIT, BSD-3-Clause, or GPL-3.0) that is compatible with the intended reuse conditions. The source datasets are governed separately by their original providers.
