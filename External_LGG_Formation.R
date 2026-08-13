library(data.table)
library(dplyr)
library(org.Hs.eg.db)
library(AnnotationDbi)

##################################################
# 1. Expression data
##################################################

expr <- fread("CGGA.mRNAseq_693.RSEM-genes.20190909.txt")

dim(expr)

gene_symbols <- expr$Gene

expr_df <- as.data.frame(expr)

rownames(expr_df) <- gene_symbols
expr_df$Gene <- NULL

##################################################
# 2. Clinical data
##################################################

clin <- fread("CGGA.mRNAseq_693.clinical.20190909.txt")

dim(clin)
colnames(clin)

table(clin$Grade)

##################################################
# 3. Keep only LGG
##################################################

clin_lgg <- clin[
  Grade %in% c("WHO II", "WHO III")
]

table(clin_lgg$Grade)

##################################################
# 4. Match samples
##################################################

lgg_samples <- clin_lgg$CGGA_ID

common_samples <- intersect(
  colnames(expr_df),
  lgg_samples
)

length(common_samples)

expr_df <- expr_df[, common_samples, drop = FALSE]

clin_lgg <- clin_lgg[
  match(common_samples, clin_lgg$CGGA_ID)
]

##################################################
# 5. log2 transformation
##################################################

expr_df <- as.data.frame(
  lapply(expr_df, as.numeric),
  row.names = rownames(expr_df)
)

expr_log <- log2(expr_df + 1)

summary(as.vector(as.matrix(expr_log)))

##################################################
# 6. Gene Symbol -> ENTREZID
##################################################

gene_map <- AnnotationDbi::select(
  org.Hs.eg.db,
  keys = rownames(expr_log),
  columns = c("ENTREZID"),
  keytype = "SYMBOL"
)

gene_map <- gene_map[!is.na(gene_map$ENTREZID), ]

gene_map <- gene_map[!duplicated(gene_map$SYMBOL), ]

expr_log <- expr_log[
  gene_map$SYMBOL,
  ,
  drop = FALSE
]

rownames(expr_log) <- gene_map$ENTREZID

##################################################
# 7. Collapse duplicated ENTREZID
##################################################

expr_dt <- as.data.table(
  expr_log,
  keep.rownames = "ENTREZID"
)

expr_entrez <- expr_dt[
  ,
  lapply(.SD, mean, na.rm = TRUE),
  by = ENTREZID
]

expr_entrez <- as.data.frame(expr_entrez)

rownames(expr_entrez) <- expr_entrez$ENTREZID
expr_entrez$ENTREZID <- NULL

##################################################
# 8. samples × genes
##################################################

expr_samp <- as.data.frame(
  t(as.matrix(expr_entrez))
)

##################################################
# 9. Add class
##################################################

expr_samp$Class <- "lgg"

##################################################
# 10. Save
##################################################


write.csv(
  expr_samp,
  "CGGA_mRNAseq_693_LGG.csv",
  row.names = FALSE
)
