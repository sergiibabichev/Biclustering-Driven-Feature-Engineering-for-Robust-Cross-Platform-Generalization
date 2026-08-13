library(data.table)
library(org.Hs.eg.db)
library(AnnotationDbi)

expr <- fread("UROMOL_gene_fpkm_gtf.txt")

## ===== 1) Prepare gene identifiers =====

gene_ensembl <- sub("\\..*$", "", expr[["tracking_id"]])

sample_cols <- setdiff(
  colnames(expr),
  c("tracking_id", "gene.type", "gene.status", "gene.name")
)

expr_df <- as.data.frame(expr[, ..sample_cols])

## Important: use Ensembl IDs, not gene symbols
rownames(expr_df) <- gene_ensembl

## Check duplicates
sum(duplicated(rownames(expr_df)))

## ===== 2) Numeric conversion and log2(FPKM + 1) =====

expr_df <- as.data.frame(
  lapply(expr_df, as.numeric),
  row.names = rownames(expr_df)
)

expr_log <- log2(expr_df + 1)

summary(as.vector(as.matrix(expr_log)))
range(expr_log, na.rm = TRUE)

## ===== 3) ENSEMBL -> ENTREZID =====

gene_map <- AnnotationDbi::select(
  org.Hs.eg.db,
  keys = rownames(expr_log),
  columns = c("ENTREZID"),
  keytype = "ENSEMBL"
)

gene_map <- gene_map[!is.na(gene_map$ENTREZID), ]
gene_map <- gene_map[!duplicated(gene_map$ENSEMBL), ]

expr_log_mapped <- expr_log[
  gene_map$ENSEMBL,
  ,
  drop = FALSE
]

## Do NOT assign duplicated ENTREZID as rownames here.
## Instead, add ENTREZID as a normal column.

expr_dt <- as.data.table(expr_log_mapped)

expr_dt[, ENTREZID := gene_map$ENTREZID]

## ===== 4) Collapse duplicated ENTREZID by mean =====

expr_entrez <- expr_dt[
  ,
  lapply(.SD, mean, na.rm = TRUE),
  by = ENTREZID
]

expr_entrez <- as.data.frame(expr_entrez)

rownames(expr_entrez) <- expr_entrez$ENTREZID
expr_entrez$ENTREZID <- NULL

expr_samp <- as.data.frame(t(expr_entrez))

expr_samp$Class <- 'blca'

write.csv(
  expr_samp,
  "UROMOL_BLCA.csv",
  row.names = FALSE
)
