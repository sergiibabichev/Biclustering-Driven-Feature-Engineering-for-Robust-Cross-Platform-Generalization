library(isa2)
library(PMA)
library(ggplot2)

# Data loading
tcga_filtered <- read.csv('TCGA_GO_filtered_data.csv')

# -----------------------------
# Remove 'X' from gene names
# -----------------------------
clean_gene_names <- function(data) {
  colnames(data) <- gsub("^X", "", colnames(data))
  return(data)
}

tcga_filtered <- clean_gene_names(tcga_filtered)


# Function for ensemble biclustering

run_ISA_PMD_frequency <- function(data_go,
                                  class_col = "Class",
                                  n_boot = 50,
                                  isa_weight_threshold = 0.50,
                                  pmd_quantile = 0.90,
                                  seed = 123) {
  set.seed(seed)
  
  if (!class_col %in% colnames(data_go)) {
    stop(paste("Column", class_col, "was not found in the data."))
  }
  
  expr_go <- data_go[, colnames(data_go) != class_col, drop = FALSE]
  gene_ids <- colnames(expr_go)
  
  expr_go <- as.matrix(expr_go)
  mode(expr_go) <- "numeric"
  
  # rows = samples, columns = genes
  X_scaled <- scale(expr_go)
  X_scaled[is.na(X_scaled)] <- 0
  
  gene_counter <- setNames(rep(0, length(gene_ids)), gene_ids)
  isa_bicluster_counts <- integer(n_boot)
  
  for (b in seq_len(n_boot)) {
    
    cat("Bootstrap iteration:", b, "/", n_boot, "\n")
    
    boot_samples <- sample(seq_len(nrow(X_scaled)), replace = TRUE)
    Xb <- X_scaled[boot_samples, , drop = FALSE]
    
    # rows = genes, columns = samples
    Xb_t <- t(Xb)
    rownames(Xb_t) <- gene_ids
    
    # ======================================================
    # ISA biclustering
    # ======================================================
    isa_res <- isa(
      Xb_t,
      thr.row = 0.30,
      thr.col = 3.00,
      no.seeds = 100
    )
    
    isa_genes <- character()
    
    if (!is.null(isa_res$rows) && is.matrix(isa_res$rows)) {
      isa_bicluster_counts[b] <- ncol(isa_res$rows)
      
      for (k in seq_len(ncol(isa_res$rows))) {
        isa_genes <- c(
          isa_genes,
          rownames(Xb_t)[which(abs(isa_res$rows[, k]) >= isa_weight_threshold)]
        )
      }
    }
    
    # ======================================================
    # PMD biclustering
    # ======================================================
    pmd_res <- PMD(
      Xb_t,
      type = "standard",
      sumabsu = 40,
      sumabsv = 5
    )
    
    if (is.matrix(pmd_res$u)) {
      pmd_u <- pmd_res$u[, 1]
    } else {
      pmd_u <- pmd_res$u
    }
    
    thr_pmd <- quantile(abs(pmd_u), pmd_quantile, na.rm = TRUE)
    
    pmd_genes <- rownames(Xb_t)[which(abs(pmd_u) >= thr_pmd)]
    
    # ======================================================
    # One hit per gene per bootstrap iteration
    # ======================================================
    genes_iteration <- unique(c(isa_genes, pmd_genes))
    
    gene_counter[genes_iteration] <- gene_counter[genes_iteration] + 1
  }
  
  gene_frequency <- gene_counter / n_boot
  
  gene_frequency_table <- data.frame(
    gene_id = names(gene_frequency),
    frequency = as.numeric(gene_frequency),
    hit_count = as.integer(gene_counter),
    row.names = NULL
  )
  
  gene_frequency_table <- gene_frequency_table[
    order(gene_frequency_table$frequency, decreasing = TRUE),
  ]
  
  list(
    gene_frequency_table = gene_frequency_table,
    isa_bicluster_counts = isa_bicluster_counts,
    parameters = list(
      ISA = list(
        thr.row = 0.30,
        thr.col = 3.00,
        no.seeds = 100,
        isa_weight_threshold = isa_weight_threshold
      ),
      PMD = list(
        sumabsu = 40,
        sumabsv = 5,
        pmd_quantile = pmd_quantile
      ),
      n_boot = n_boot
    )
  )
}

# Histogram plot for selectinh thresholding

plot_gene_frequency_histogram <- function(freq_res,
                                          threshold = NULL,
                                          bins = 40,
                                          title = "Distribution of gene participation frequency in biclusters") {
  
  df <- freq_res$gene_frequency_table
  
  p <- ggplot(df, aes(x = frequency)) +
    geom_histogram(bins = bins, fill = "steelblue", color = "black") +
    theme_minimal(base_size = 14) +
    labs(
      title = title,
      x = "Frequency of gene participation",
      y = "Number of genes"
    )
  
  if (!is.null(threshold)) {
    p <- p +
      geom_vline(
        xintercept = threshold,
        color = "red",
        linetype = "dashed",
        linewidth = 1
      )
  }
  
  return(p)
}

# Function to form reduced data
create_reduced_data_by_frequency <- function(data_go,
                                             freq_res,
                                             class_col = "Class",
                                             threshold = 0.10) {
  
  df_freq <- freq_res$gene_frequency_table
  
  selected_genes <- df_freq$gene_id[df_freq$frequency >= threshold]
  
  reduced_data <- data_go[, c(selected_genes, class_col), drop = FALSE]
  
  list(
    selected_genes = selected_genes,
    reduced_data = reduced_data,
    threshold = threshold,
    n_selected_genes = length(selected_genes)
  )
}




res_tcga_freq <- run_ISA_PMD_frequency(
  data_go = tcga_filtered,
  class_col = "Class",
  n_boot = 50,
  isa_weight_threshold = 0.50,
  pmd_quantile = 0.80,
  seed = 123
)

sum(res_tcga_freq$gene_frequency_table$frequency >= 0.02)
sum(res_tcga_freq$gene_frequency_table$frequency >= 0.05)
sum(res_tcga_freq$gene_frequency_table$frequency >= 0.1)
sum(res_tcga_freq$gene_frequency_table$frequency >= 0.2)

# Histogram visualization

plot_gene_frequency_hits <- function(freq_res,
                                     n_boot = 50,
                                     min_hits = 1,
                                     bins = 50,
                                     title = "TCGA data: distribution of gene participation in biclusters") {
  
  library(ggplot2)
  
  df <- freq_res$gene_frequency_table
  
  # Convert frequency to number of bootstrap hits
  df$hit_count <- df$frequency * n_boot
  
  p <- ggplot(df, aes(x = hit_count)) +
    geom_histogram(
      bins = bins,
      fill = "steelblue",
      color = "black"
    ) +
    
    geom_vline(
      xintercept = min_hits,
      color = "red",
      linetype = "dashed",
      linewidth = 1
    ) +
    
    theme_minimal(base_size = 14) +
    
    labs(
      title = title,
      x = "Number of bootstrap iterations with gene participation",
      y = "Number of genes"
    )
  
  return(p)
}

p1 <- plot_gene_frequency_hits(
  freq_res = res_tcga_freq,
  n_boot = 50,
  min_hits = 1
)

plot(p1)

create_reduced_data_by_hits <- function(data_go,
                                        freq_res,
                                        class_col = "Class",
                                        min_hits = 1) {
  
  df_freq <- freq_res$gene_frequency_table
  
  selected_genes <- df_freq$gene_id[
    df_freq$hit_count >= min_hits
  ]
  
  reduced_data <- data_go[, c(selected_genes, class_col), drop = FALSE]
  
  list(
    selected_genes = selected_genes,
    reduced_data = reduced_data,
    min_hits = min_hits,
    n_selected_genes = length(selected_genes)
  )
}


res_tcga_reduced <- create_reduced_data_by_hits(
  data_go = tcga_filtered,
  freq_res = res_tcga_freq,
  class_col = "Class",
  min_hits = 1
)

length(res_tcga_reduced$selected_genes)

dim(res_tcga_reduced$reduced_data)

tcga_reduced_data <- res_tcga_reduced$reduced_data 


write.csv(tcga_reduced_data, 'tcga_BC_reduced.csv', row.names = FALSE)


png(file = 'histogram_tcga.png', width = 2500, height = 1200,res = 300)
plot(p1) 
dev.off()

