library(data.table)
library(dplyr)
library(AnnotationDbi)
library(org.Hs.eg.db)



## ===== 2) Find RNA-seq expression file =====

expr_file <- "BRCA_RNAseq_gene_RSEM_coding_UQ_1500_log2_Tumor.txt"

## ===== 3) Load expression matrix =====

expr_raw <- fread(
  expr_file,
  data.table = FALSE,
  check.names = FALSE
)

cat("Raw expression dimensions:\n")
print(dim(expr_raw))

cat("First columns:\n")
print(colnames(expr_raw)[1:10])

cat("Preview:\n")
print(head(expr_raw[, 1:5]))

## ===== 4) Detect gene ID column =====
## Usually the first column contains Ensembl IDs.

gene_col <- colnames(expr_raw)[1]

gene_ids_raw <- expr_raw[[gene_col]]

expr_mat <- expr_raw[, -1, drop = FALSE]

## ===== 5) Convert expression values to numeric =====

expr_mat <- as.data.frame(
  lapply(expr_mat, function(x) as.numeric(as.character(x)))
)

## ===== 6) Remove Ensembl version suffix =====
## ENSG00000000003.15 -> ENSG00000000003

ensembl_ids <- sub("\\..*$", "", gene_ids_raw)

cat("Example Ensembl IDs:\n")
print(head(data.frame(raw = gene_ids_raw, clean = ensembl_ids)))

## ===== 7) Map ENSEMBL -> ENTREZID =====

gene_map <- AnnotationDbi::select(
  org.Hs.eg.db,
  keys = unique(ensembl_ids),
  columns = c("ENTREZID", "SYMBOL"),
  keytype = "ENSEMBL"
)

gene_map <- gene_map[!is.na(gene_map$ENTREZID), ]
gene_map <- gene_map[!duplicated(gene_map$ENSEMBL), ]

cat("Mapping dimensions:\n")
print(dim(gene_map))

cat("Mapping preview:\n")
print(head(gene_map))

## ===== 8) Attach ENTREZID to expression matrix =====

map_idx <- match(ensembl_ids, gene_map$ENSEMBL)

valid <- !is.na(map_idx)

expr_mat <- expr_mat[valid, , drop = FALSE]
entrez_ids <- gene_map$ENTREZID[map_idx[valid]]

cat("Expression after mapping:\n")
print(dim(expr_mat))

cat("Number of unique ENTREZ IDs:\n")
print(length(unique(entrez_ids)))

## ===== 9) Collapse duplicated ENTREZID by mean =====

expr_dt <- as.data.table(expr_mat)

expr_dt[, ENTREZID := as.character(entrez_ids)]

expr_gene <- expr_dt[
  ,
  lapply(.SD, mean, na.rm = TRUE),
  by = ENTREZID,
  .SDcols = setdiff(colnames(expr_dt), "ENTREZID")
]

expr_gene <- as.data.frame(expr_gene)

rownames(expr_gene) <- expr_gene$ENTREZID
expr_gene$ENTREZID <- NULL

cat("Gene-level expression dimensions:\n")
print(dim(expr_gene))

cat("Duplicated ENTREZ IDs after collapsing:\n")
print(sum(duplicated(rownames(expr_gene))))

## ===== 10) Check expression scale =====
## File name contains log2, so do NOT apply log2 again.

cat("Expression summary:\n")
print(summary(as.vector(as.matrix(expr_gene))))

cat("Expression range:\n")
print(range(expr_gene, na.rm = TRUE))

## ===== 11) Convert genes × samples to samples × genes =====

expr_samp <- as.data.frame(
  t(as.matrix(expr_gene))
)

cat("Sample-level expression dimensions:\n")
print(dim(expr_samp))

## ===== 12) Add Class column =====

expr_samp$Class <- "brca"

cat("Class distribution:\n")
print(table(expr_samp$Class))

## ===== 13) Save final file =====

write.csv(
  expr_samp,
  "CPTAC_BRCA.csv",
  row.names = FALSE
)

cat("Saved file: CPTAC_BRCA.csv\n")