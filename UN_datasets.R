# ============================================================
# 1) Data loading
# ============================================================

tcga_reduced <- read.csv("tcga_BC_reduced.csv", check.names = FALSE)

BLCA = read.csv("BLCA_validation.csv", check.names = FALSE)
BRCA = read.csv("BRCA_validation.csv", check.names = FALSE)
COAD = read.csv("COAD_validation.csv", check.names = FALSE)
HNSC = read.csv("HNSC_validation.csv", check.names = FALSE)
KIRC = read.csv("KIRC_validation.csv", check.names = FALSE)
LUAD = read.csv("LUAD_validation.csv", check.names = FALSE)
LGG = read.csv("LGG_validation.csv", check.names = FALSE)
LUSC = read.csv("LUSC_validation.csv", check.names = FALSE)


# ============================================================
# 2) Clean gene column names
# ============================================================

clean_gene_names <- function(data) {
  colnames(data) <- gsub("^X", "", colnames(data))
  return(data)
}

tcga_reduced <- clean_gene_names(tcga_reduced)
BLCA <- clean_gene_names(BLCA)
BRCA <- clean_gene_names(BRCA)
COAD <- clean_gene_names(COAD)
HNSC <- clean_gene_names(HNSC)
KIRC <- clean_gene_names(KIRC)
LUAD <- clean_gene_names(LUAD)
LGG <- clean_gene_names(LGG)
LUSC <- clean_gene_names(LUSC)

validation_list <- list(BLCA = BLCA, BRCA = BRCA, COAD = COAD, HNSC = HNSC,
                        KIRC = KIRC, LUAD = LUAD, LGG = LGG, LUSC = LUSC)


# ============================================================
# 3) Helper function
# ============================================================

get_gene_columns <- function(data,
                             class_col = "Class",
                             sample_col = "SampleID") {
  
  setdiff(
    colnames(data),
    c(class_col, sample_col)
  )
}

# ============================================================
# 4) Find common genes across TCGA and all validation datasets
# ============================================================

tcga_genes <- get_gene_columns(tcga_reduced)

common_genes_all <- tcga_genes

for (nm in names(validation_list)) {
  val_genes <- get_gene_columns(validation_list[[nm]])
  common_genes_all <- intersect(common_genes_all, val_genes)
}

# Preserve TCGA gene order
common_genes_all <- tcga_genes[tcga_genes %in% common_genes_all]

length(common_genes_all)

# ============================================================
# 5) Align TCGA and each validation dataset
# ============================================================

align_dataset <- function(data,
                          common_genes,
                          class_col = "Class",
                          sample_col = "SampleID") {
  
  keep_cols <- common_genes
  
  if (sample_col %in% colnames(data)) {
    result <- data[, c(sample_col, keep_cols, class_col), drop = FALSE]
  } else {
    result <- data[, c(keep_cols, class_col), drop = FALSE]
  }
  
  return(result)
}

tcga_train_final <- align_dataset(
  data = tcga_reduced,
  common_genes = common_genes_all
)

BLCA_final <- align_dataset(
  data = BLCA,
  common_genes = common_genes_all
)

BRCA_final <- align_dataset(
  data = BRCA,
  common_genes = common_genes_all
)

COAD_final <- align_dataset(
  data = COAD,
  common_genes = common_genes_all
)

HNSC_final <- align_dataset(
  data = HNSC,
  common_genes = common_genes_all
)

KIRC_final <- align_dataset(
  data = KIRC,
  common_genes = common_genes_all
)

LGG_final <- align_dataset(
  data = LGG,
  common_genes = common_genes_all
)

LUAD_final <- align_dataset(
  data = LUAD,
  common_genes = common_genes_all
)

LUSC_final <- align_dataset(
  data = LUSC,
  common_genes = common_genes_all
)

ALL_final <- rbind(BLCA_final, BRCA_final, COAD_final, HNSC_final,KIRC_final,
                   LGG_final,LUAD_final, LUSC_final)



# ============================================================
# 6) Save each aligned validation dataset separately
# ============================================================

write.csv(
  tcga_train_final,
  "tcga_cancer_common_genes.csv",
  row.names = FALSE
)

write.csv(
  BRCA_final,
  "BRCA_validation_common_genes.csv",
  row.names = FALSE
)

write.csv(
  BLCA_final,
  "BLCA_validation_common_genes.csv",
  row.names = FALSE
)

write.csv(
  COAD_final,
  "COAD_validation_common_genes.csv",
  row.names = FALSE
)

write.csv(
  HNSC_final,
  "HNSC_validation_common_genes.csv",
  row.names = FALSE
)

write.csv(
  KIRC_final,
  "KIRC_validation_common_genes.csv",
  row.names = FALSE
)

write.csv(
  LGG_final,
  "LGG_validation_common_genes.csv",
  row.names = FALSE
)

write.csv(
  LUAD_final,
  "LUAD_validation_common_genes.csv",
  row.names = FALSE
)

write.csv(
  LUSC_final,
  "LUSC_validation_common_genes.csv",
  row.names = FALSE
)

write.csv(
  ALL_final,
  "ALL_validation_common_genes.csv",
  row.names = FALSE
)


get_stats <- function(df){
  
  X <- df[, !(names(df) %in% "Class")]
  
  values <- as.numeric(as.matrix(X))
  
  data.frame(
    Min    = min(values, na.rm = TRUE),
    Q1     = quantile(values, 0.25, na.rm = TRUE),
    Median = median(values, na.rm = TRUE),
    Mean   = mean(values, na.rm = TRUE),
    Q3     = quantile(values, 0.75, na.rm = TRUE),
    Max    = max(values, na.rm = TRUE),
    SD     = sd(values, na.rm = TRUE)
  )
}

get_stats(tcga_train_final)
get_stats(LUAD_final)
get_stats(LUSC_final)
get_stats(HNSC_final)

get_stats(BRCA_final)
get_stats(BLCA_final)
get_stats(COAD_final)
get_stats(KIRC_final)
get_stats(LGG_final)


