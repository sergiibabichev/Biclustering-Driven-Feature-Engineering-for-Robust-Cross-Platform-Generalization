
# -----------------------------
# Libraries
# -----------------------------
packages <- c(
  "dplyr", "ggplot2", "reshape2",
  "limma", "clusterProfiler", "org.Hs.eg.db",
  "AnnotationDbi", "enrichplot"
)

for (p in packages) {
  if (!require(p, character.only = TRUE)) {
    install.packages(p, dependencies = TRUE)
    library(p, character.only = TRUE)
  }
}

if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

bio_packages <- c("clusterProfiler", "org.Hs.eg.db", "limma", "enrichplot", "AnnotationDbi")

for (p in bio_packages) {
  if (!require(p, character.only = TRUE)) {
    BiocManager::install(p, ask = FALSE, update = FALSE)
    library(p, character.only = TRUE)
  }
}




# Data loading
tcga_data <- read.csv('tcga_cancer_without_normal.csv')


# -----------------------------
# Create output directories
# -----------------------------
dir.create("results", showWarnings = FALSE)
dir.create("results/normality", showWarnings = FALSE)
dir.create("results/GO", showWarnings = FALSE)

# -----------------------------
# Remove 'X' from gene names
# -----------------------------
clean_gene_names <- function(data) {
  colnames(data) <- gsub("^X", "", colnames(data))
  return(data)
}

tcga_data <- clean_gene_names(tcga_data)

# Formation of the features and labels

prepare_data <- function(data) {
  
  # select Class
  class_vector <- data$Class
  
  # Remove Class
  expr <- data[, !colnames(data) %in% "Class"]
  
  # Convert to numeric
  expr <- as.data.frame(lapply(expr, as.numeric))
  
  # Remove NA
  expr <- expr[complete.cases(expr), ]
  class_vector <- class_vector[complete.cases(expr)]
  
  # Convert to matrix
  expr <- as.matrix(expr)
  
  return(list(expr = expr, class = class_vector))
}

tcga <- prepare_data(tcga_data)

tcga_expr <- tcga$expr
tcga_class <- tcga$class


normality_analysis <- function(expr_matrix, dataset_name) {

  results <- data.frame(
    Gene = colnames(expr_matrix),
    W = as.numeric(rep(NA, ncol(expr_matrix))),
    p_value = as.numeric(rep(NA, ncol(expr_matrix)))
  )
  
  for (i in seq_len(ncol(expr_matrix))) {
    
    x <- expr_matrix[, i]
    
    if(length(x) > 5000){
      x <- sample(x, 5000)
    }
    
    if (length(x) >= 3 && length(x) <= 5000) {
      test <- shapiro.test(x)
      results$W[i] <- test$statistic
      results$p_value[i] <- test$p.value
    }
  }
  
  results$Normality <- ifelse(
    results$p_value > 0.05,
    "Normal",
    "Not Normal"
  )
  
  results$p_value <- as.numeric(results$p_value)
  results$W <- as.numeric(results$W)
  
  # -----------------------------
  # Plots
  # -----------------------------
  
  p_hist <- ggplot(results, aes(x = p_value)) +
    geom_histogram(bins = 50, fill = "green", color = "blue") +
    theme_minimal(base_size = 13) +
    labs(
      title = paste("Shapiro p-values:", dataset_name),
      x = "p-value",
      y = "Count"
    )
  
  p_bar <- ggplot(results, aes(x = Normality, fill = Normality)) +
    geom_bar() +
    theme_minimal(base_size = 13) +
    labs(
      title = paste("Normality:", dataset_name),
      x = "",
      y = "Count"
    )
  
  # Save individual plots
  ggsave(paste0("results_", dataset_name, "_pvalues.png"),
         p_hist, width = 8, height = 5, dpi = 300)
  
  ggsave(paste0("results_", dataset_name, "_normality_bar.png"),
         p_bar, width = 6, height = 4, dpi = 300)
  
  return(list(
    results = results,
    hist = p_hist,
    bar = p_bar
  ))
}


tcga_out <- normality_analysis(tcga_expr, "TCGA")


library(patchwork)

combined_plot <- 
  (tcga_out$bar | tcga_out$hist)


ggsave(
  "combined_normality_TCGA.png",
  combined_plot,
  width = 10,
  height = 6,
  dpi = 300
)

# Wilcoxon test

run_wilcoxon <- function(expr_matrix, class_vector, dataset_name) {
  
  # Clean Entrez ID column names
  gene_names <- colnames(expr_matrix)
  gene_names <- gsub("^X", "", gene_names)
  
  class_vector <- droplevels(factor(class_vector))
  groups <- levels(class_vector)
  
  if (length(groups) < 2) {
    stop("At least two classes are required for Wilcoxon testing.")
  }
  
  class_pairs <- combn(groups, 2, simplify = FALSE)
  
  results_list <- vector(
    "list",
    length = ncol(expr_matrix) * length(class_pairs)
  )
  
  result_index <- 1
  
  for (i in seq_len(ncol(expr_matrix))) {
    
    x <- expr_matrix[, i]
    
    for (class_pair in class_pairs) {
      
      class_1 <- class_pair[1]
      class_2 <- class_pair[2]
      
      group_1 <- x[class_vector == class_1]
      group_2 <- x[class_vector == class_2]
      
      valid_1 <- is.finite(group_1)
      valid_2 <- is.finite(group_2)
      
      group_1 <- group_1[valid_1]
      group_2 <- group_2[valid_2]
      
      if (length(group_1) > 0 &&
          length(group_2) > 0 &&
          length(unique(c(group_1, group_2))) > 1) {
        
        test_result <- wilcox.test(
          group_1,
          group_2,
          alternative = "two.sided",
          exact = FALSE
        )
        
        p_value <- test_result$p.value
        
      } else {
        
        p_value <- 1
      }
      
      results_list[[result_index]] <- data.frame(
        Gene = gene_names[i],
        Class_1 = class_1,
        Class_2 = class_2,
        p_value = p_value,
        stringsAsFactors = FALSE
      )
      
      result_index <- result_index + 1
    }
  }
  
  pairwise_results <- do.call(rbind, results_list)
  
  # BH correction across all gene-by-class-pair comparisons
  pairwise_results$adj_p <- p.adjust(
    pairwise_results$p_value,
    method = "BH"
  )
  
  significant_pairwise <- pairwise_results[
    pairwise_results$adj_p < 0.05,
  ]
  
  # A gene is retained if it differs significantly in at least one
  # pairwise class comparison
  significant_genes <- unique(significant_pairwise$Gene)
  
  gene_summary <- data.frame(
    Gene = gene_names,
    min_p_value = vapply(
      gene_names,
      function(gene) {
        min(pairwise_results$p_value[
          pairwise_results$Gene == gene
        ])
      },
      numeric(1)
    ),
    min_adj_p = vapply(
      gene_names,
      function(gene) {
        min(pairwise_results$adj_p[
          pairwise_results$Gene == gene
        ])
      },
      numeric(1)
    ),
    significant = gene_names %in% significant_genes,
    stringsAsFactors = FALSE
  )
  
  write.csv(
    pairwise_results,
    paste0("wilcox_all_pairwise_", dataset_name, ".csv"),
    row.names = FALSE
  )
  
  write.csv(
    significant_pairwise,
    paste0("wilcox_significant_pairwise_", dataset_name, ".csv"),
    row.names = FALSE
  )
  
  write.csv(
    gene_summary,
    paste0("wilcox_gene_summary_", dataset_name, ".csv"),
    row.names = FALSE
  )
  
  significant_gene_table <- gene_summary[
    gene_summary$significant,
    c("Gene", "min_p_value", "min_adj_p")
  ]
  
  write.csv(
    significant_gene_table,
    paste0("wilcox_significant_", dataset_name, ".csv"),
    row.names = FALSE
  )
  
  message(
    "Wilcoxon filtering completed for ", dataset_name,
    ": ", length(groups), " classes; ",
    length(class_pairs), " class pairs; ",
    nrow(significant_gene_table), " retained genes."
  )
  
  return(significant_gene_table)
}

tcga_sig <- run_wilcoxon(tcga_expr, tcga_class, "TCGA")

head(tcga_sig$Gene, 20)

# GO analysis

library(clusterProfiler)
library(org.Hs.eg.db)

run_GO_all <- function(gene_list, dataset_name) {
  
  genes <- as.character(gene_list$Gene)
  
  # BP
  go_BP <- enrichGO(
    gene = genes,
    OrgDb = org.Hs.eg.db,
    keyType = "ENTREZID",
    ont = "BP",
    pvalueCutoff = 0.05
  )
  
  # MF
  go_MF <- enrichGO(
    gene = genes,
    OrgDb = org.Hs.eg.db,
    keyType = "ENTREZID",
    ont = "MF",
    pvalueCutoff = 0.05
  )
  
  # CC
  go_CC <- enrichGO(
    gene = genes,
    OrgDb = org.Hs.eg.db,
    keyType = "ENTREZID",
    ont = "CC",
    pvalueCutoff = 0.05
  )
  
  # Save
  write.csv(as.data.frame(go_BP), paste0("GO_BP_", dataset_name, ".csv"))
  write.csv(as.data.frame(go_MF), paste0("GO_MF_", dataset_name, ".csv"))
  write.csv(as.data.frame(go_CC), paste0("GO_CC_", dataset_name, ".csv"))
  
  return(list(BP = go_BP, MF = go_MF, CC = go_CC))
}

tcga_GO <- run_GO_all(tcga_sig, "TCGA")

# -----------------------------
# Extract genes from GO results
# -----------------------------
extract_GO_genes <- function(go_results) {
  
  all_genes <- c()
  
  for (ont in names(go_results)) {
    
    df <- as.data.frame(go_results[[ont]])
    
    if (nrow(df) > 0 && "geneID" %in% colnames(df)) {
      genes <- unlist(strsplit(df$geneID, "/"))
      all_genes <- c(all_genes, genes)
    }
  }
  
  all_genes <- unique(all_genes)
  all_genes <- gsub("^X", "", all_genes)
  
  return(all_genes)
}

tcga_GO_genes <- extract_GO_genes(tcga_GO)

length(tcga_GO_genes)


# -----------------------------
# Create filtered datasets
# -----------------------------
create_filtered_dataset <- function(expr_matrix, class_vector, selected_genes, dataset_name) {
  
  colnames(expr_matrix) <- gsub("^X", "", colnames(expr_matrix))
  selected_genes <- gsub("^X", "", selected_genes)
  
  common_genes <- intersect(colnames(expr_matrix), selected_genes)
  
  cat("Dataset:", dataset_name, "\n")
  cat("Number of selected GO genes:", length(selected_genes), "\n")
  cat("Number of genes found in expression matrix:", length(common_genes), "\n")
  
  filtered_expr <- expr_matrix[, common_genes, drop = FALSE]
  
  filtered_data <- as.data.frame(filtered_expr)
  filtered_data$Class <- class_vector
  
  write.csv(
    filtered_data,
    paste0(dataset_name, "_GO_filtered_data.csv"),
    row.names = FALSE
  )
  
  return(filtered_data)
}

tcga_GO_filtered <- create_filtered_dataset(
  tcga_expr,
  tcga_class,
  tcga_GO_genes,
  "TCGA"
)

library(enrichplot)
library(ggplot2)
library(patchwork)

# -----------------------------
# Function for GO dotplot
# -----------------------------
plot_GO_dotplot <- function(go_results, dataset_name, show_n = 10) {
  
  p_BP <- dotplot(go_results$BP, showCategory = show_n) +
    ggtitle(paste0(dataset_name, " - BP")) +
    theme_minimal(base_size = 15)
  
  p_MF <- dotplot(go_results$MF, showCategory = show_n) +
    ggtitle(paste0(dataset_name, " - MF")) +
    theme_minimal(base_size = 15)
  
  p_CC <- dotplot(go_results$CC, showCategory = show_n) +
    ggtitle(paste0(dataset_name, " - CC")) +
    theme_minimal(base_size = 15)
  
  combined <- p_BP / p_MF / p_CC
  
  ggsave(
    paste0("results/GO/", dataset_name, "_GO_dotplot_BP_MF_CC.png"),
    combined,
    width = 14,
    height = 16,
    dpi = 300
  )
  
  return(combined)
}

tcga_GO_dotplot <- plot_GO_dotplot(tcga_GO, "TCGA", show_n = 20)
