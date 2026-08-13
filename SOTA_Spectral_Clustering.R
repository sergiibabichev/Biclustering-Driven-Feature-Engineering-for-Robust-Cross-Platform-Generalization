
# Data loading
tcga_data <- read.csv('tcga_cancer_common_genes.csv')
cancer_validation <- read.csv('All_validation_common_genes.csv')

# -----------------------------
# Remove 'X' from gene names
# -----------------------------
clean_gene_names <- function(data) {
  colnames(data) <- gsub("^X", "", colnames(data))
  return(data)
}

tcga_data <- clean_gene_names(tcga_data)
cancer_validation <- clean_gene_names(cancer_validation)



dim(tcga_data)


# -----------------------------
# 1. Packages
# -----------------------------


library(clValid)
library(kernlab)
library(Matrix)
library(dplyr)
library(readr)
library(ggplot2)
library(grid)
library(gridExtra)



# -----------------------------
# 2. Helper: prepare matrix
# -----------------------------
prepare_expression_matrix <- function(data, class_col = "Class") {
  
  if (!class_col %in% colnames(data)) {
    stop("Class column was not found.")
  }
  
  expr <- data[, setdiff(colnames(data), class_col), drop = FALSE]
  gene_ids <- colnames(expr)
  
  expr <- as.matrix(expr)
  storage.mode(expr) <- "double"
  
  # Genes as rows, samples as columns
  expr_t <- t(expr)
  rownames(expr_t) <- gene_ids
  
  # Standardization of gene profiles across samples
  expr_t <- t(scale(t(expr_t)))
  expr_t[is.na(expr_t)] <- 0
  
  return(expr_t)
}

# -----------------------------
# 3. Helper: extract SOTA labels
# -----------------------------
extract_sota_labels <- function(sota_model, n_objects) {
  
  possible_slots <- c("cluster", "clust", "clustering", "labels")
  
  for (slot in possible_slots) {
    if (!is.null(sota_model[[slot]])) {
      lab <- as.integer(sota_model[[slot]])
      if (length(lab) == n_objects) return(lab)
    }
  }
  
  stop("Cannot extract cluster labels from SOTA object. Please inspect str(sota_model).")
}

# -----------------------------
# 4. One SOTA run
# -----------------------------
run_sota_once <- function(expr_gene_sample,
                          max_cycles = 50,
                          max_epochs = 50,
                          distance = "correlation") {
  
  sota_model <- clValid::sota(
    data = expr_gene_sample,
    maxCycles = max_cycles,
    maxEpochs = max_epochs,
    distance = distance
  )
  
  labels <- extract_sota_labels(
    sota_model = sota_model,
    n_objects = nrow(expr_gene_sample)
  )
  
  return(as.integer(as.factor(labels)))
}

# -----------------------------
# 5. Ensemble SOTA + consensus matrix
# -----------------------------
ensemble_sota_consensus <- function(data,
                                    dataset_name,
                                    class_col = "Class",
                                    n_bootstrap = 50,
                                    gene_fraction = 0.6,
                                    k_final = 3,
                                    distance = "correlation",
                                    seed = 123,
                                    out_dir = "SOTA_results",
                                    n_workers = 20,
                                    max_cycles = 50,
                                    max_epochs = 50) {
  
  set.seed(seed)
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  
  expr_gene_sample <- prepare_expression_matrix(
    data = data,
    class_col = class_col
  )
  
  n_genes <- nrow(expr_gene_sample)
  n_samples <- ncol(expr_gene_sample)
  gene_ids <- rownames(expr_gene_sample)
  
  message("Dataset: ", dataset_name)
  message("Genes: ", n_genes)
  message("Samples: ", n_samples)
  message("Bootstrap iterations: ", n_bootstrap)
  message("Gene fraction: ", gene_fraction)
  message("Final clusters: ", k_final)
  
  if (.Platform$OS.type == "windows") {
    future::plan(future::multisession, workers = n_workers)
  } else {
    future::plan(future::multicore, workers = n_workers)
  }
  
  bootstrap_results <- future.apply::future_lapply(
    X = seq_len(n_bootstrap),
    FUN = function(b) {
      
      set.seed(seed + b)
      
      message("Bootstrap iteration: ", b, " / ", n_bootstrap)
      
      sampled_genes <- sample(
        x = seq_len(n_genes),
        size = floor(gene_fraction * n_genes),
        replace = FALSE
      )
      
      expr_boot <- expr_gene_sample[sampled_genes, , drop = FALSE]
      
      sota_model <- clValid::sota(
        data = expr_boot,
        maxCycles = max_cycles,
        maxEpochs = max_epochs,
        distance = distance
      )
      
      labels <- extract_sota_labels(
        sota_model = sota_model,
        n_objects = nrow(expr_boot)
      )
      
      labels <- as.integer(as.factor(labels))
      
      return(list(
        sampled_genes = sampled_genes,
        labels = labels,
        n_clusters = length(unique(labels))
      ))
    },
    future.seed = TRUE
  )
  
  future::plan(future::sequential)
  
  consensus <- matrix(
    0,
    nrow = n_genes,
    ncol = n_genes
  )
  
  rownames(consensus) <- gene_ids
  colnames(consensus) <- gene_ids
  
  cluster_counts <- integer(n_bootstrap)
  
  for (b in seq_len(n_bootstrap)) {
    
    message("Consensus update: ", b, " / ", n_bootstrap)
    
    sampled_genes <- bootstrap_results[[b]]$sampled_genes
    labels <- bootstrap_results[[b]]$labels
    cluster_counts[b] <- bootstrap_results[[b]]$n_clusters
    
    for (cl in sort(unique(labels))) {
      
      idx <- sampled_genes[labels == cl]
      
      if (length(idx) > 1) {
        consensus[idx, idx] <- consensus[idx, idx] + 1
      }
    }
  }
  
  consensus <- consensus / n_bootstrap
  diag(consensus) <- 1
  
  message("Running spectral clustering...")
  
  spectral_model <- kernlab::specc(
    kernlab::as.kernelMatrix(consensus),
    centers = k_final
  )
  
  final_labels <- as.integer(spectral_model)
  
  result_table <- data.frame(
    Gene_ID = gene_ids,
    Cluster = final_labels
  ) %>%
    dplyr::arrange(Cluster, Gene_ID)
  
  cluster_summary <- result_table %>%
    dplyr::group_by(Cluster) %>%
    dplyr::summarise(
      Number_of_genes = dplyr::n(),
      .groups = "drop"
    )
  
  cluster_count_summary <- data.frame(
    Bootstrap_iteration = seq_len(n_bootstrap),
    Number_of_SOTA_clusters = cluster_counts
  )
  
  readr::write_csv(
    result_table,
    file.path(out_dir, paste0(dataset_name, "_SOTA_spectral_gene_clusters.csv"))
  )
  
  readr::write_csv(
    cluster_summary,
    file.path(out_dir, paste0(dataset_name, "_SOTA_spectral_cluster_summary.csv"))
  )
  
  readr::write_csv(
    cluster_count_summary,
    file.path(out_dir, paste0(dataset_name, "_SOTA_bootstrap_cluster_counts.csv"))
  )
  
  saveRDS(
    consensus,
    file.path(out_dir, paste0(dataset_name, "_SOTA_consensus_matrix.rds"))
  )
  
  saveRDS(
    spectral_model,
    file.path(out_dir, paste0(dataset_name, "_spectral_model.rds"))
  )
  
  message("Finished: ", dataset_name)
  
  return(list(
    dataset_name = dataset_name,
    gene_clusters = result_table,
    cluster_summary = cluster_summary,
    bootstrap_cluster_counts = cluster_count_summary,
    consensus_matrix = consensus,
    spectral_model = spectral_model
  ))
}

# ============================================================
# 6. Run for TCGA training data 
# ============================================================

library(future)

plan(multisession, workers = 10)

options(future.globals.maxSize = 8*1024^3)


tcga_sota_result <- ensemble_sota_consensus(
  data = tcga_data,
  dataset_name = "TCGA_training",
  class_col = "Class",
  n_bootstrap = 40,
  gene_fraction = 0.6,
  k_final = 2,
  distance = "correlation",
  seed = 123,
  out_dir = "SOTA_results_new",
  n_workers = 10
)


# View summaries
tcga_sota_result$cluster_summary

# Results visualization

# 1. Bar plots of the clusters size

plot_cluster_sizes_ggplot <- function(cluster_summary,
                                      dataset_name,
                                      out_dir = "SOTA_results_new") {
  
  # -----------------------------
  # Create output directory
  # -----------------------------
  dir.create(out_dir,
             showWarnings = FALSE,
             recursive = TRUE)
  
  
  # -----------------------------
  # Prepare data
  # -----------------------------
  cluster_summary$Cluster <- factor(
    cluster_summary$Cluster,
    levels = cluster_summary$Cluster,
    labels = paste("Cluster", cluster_summary$Cluster)
  )
  
  ymax <- max(cluster_summary$Number_of_genes) * 1.15
  
  # -----------------------------
  # Create plot
  # -----------------------------
  p <- ggplot(
    cluster_summary,
    aes(
      x = Cluster,
      y = Number_of_genes,
      fill = Cluster
    )
  ) +
    
    geom_bar(
      stat = "identity",
      width = 0.8,
      color = "black"
    ) +
    
    geom_text(
      aes(label = Number_of_genes),
      vjust = -0.5,
      size = 6,
      fontface = "bold"
    ) +
    
    scale_fill_manual(
      values = c(
        "Cluster 1" = "darkgreen",
        "Cluster 2" = "steelblue3"
      )
    ) +
    
    labs(
      title = paste("Gene cluster sizes:", dataset_name),
      x = "Cluster",
      y = "Number of genes"
    ) +
    
    ylim(0, ymax) +
    
    theme_bw(base_size = 18) +
    
    theme(
      plot.title = element_text(
        hjust = 0.5,
        face = "bold",
        size = 24,
        margin = margin(b = 20)
      ),
      
      axis.title = element_text(
        face = "bold",
        size = 20
      ),
      
      axis.text = element_text(
        size = 16,
        face = "bold"
      ),
      
      panel.grid.major = element_line(
        color = "gray85",
        linetype = "dashed"
      ),
      
      panel.grid.minor = element_blank(),
      
      legend.position = "none",
      
      plot.margin = margin(
        t = 20,
        r = 20,
        b = 20,
        l = 20
      )
    )
  
  # -----------------------------
  # Save figure
  # -----------------------------
  ggsave(
    filename = file.path(
      out_dir,
      paste0(dataset_name, "_cluster_sizes_ggplot.png")
    ),
    
    plot = p,
    
    width = 10,
    height = 7,
    dpi = 300
  )
  
  return(p)
}

p1 <- plot_cluster_sizes_ggplot(
  tcga_sota_result$cluster_summary,
  "TCGA Cancer Dataset"
)


# 3. GO analysis

library(clusterProfiler)
library(org.Hs.eg.db)

run_combined_go_enrichment_for_clusters <- function(sota_result,
                                                    dataset_name,
                                                    universe_genes = NULL,
                                                    top_n = 20,
                                                    out_dir = "SOTA_results_new/GO_clusters") {
  
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  
  
  
  gene_clusters <- sota_result$gene_clusters
  
  if (is.null(universe_genes)) {
    universe_genes <- unique(gene_clusters$Gene_ID)
  }
  
  ontologies <- c("BP", "MF", "CC")
  plot_list <- list()
  
  for (cl in sort(unique(gene_clusters$Cluster))) {
    
    message("Combined GO enrichment: ", dataset_name, ", Cluster ", cl)
    
    cluster_genes <- gene_clusters %>%
      filter(Cluster == cl) %>%
      pull(Gene_ID) %>%
      unique()
    
    go_all <- list()
    
    for (ont in ontologies) {
      
      ego <- enrichGO(
        gene          = cluster_genes,
        universe      = universe_genes,
        OrgDb         = org.Hs.eg.db,
        keyType       = "ENTREZID",
        ont           = ont,
        pAdjustMethod = "BH",
        pvalueCutoff  = 0.05,
        qvalueCutoff  = 0.2,
        readable      = TRUE
      )
      
      ego_df <- as.data.frame(ego)
      
      if (nrow(ego_df) > 0) {
        ego_df$Ontology <- ont
        go_all[[ont]] <- ego_df
      }
    }
    
    if (length(go_all) == 0) {
      message("No significant GO terms for Cluster ", cl)
      next
    }
    
    go_combined <- bind_rows(go_all) %>%
      mutate(
        GeneRatio_num = sapply(
          strsplit(as.character(GeneRatio), "/"),
          function(x) as.numeric(x[1]) / as.numeric(x[2])
        )
      ) %>%
      arrange(p.adjust) %>%
      slice_head(n = top_n) %>%
      arrange(desc(GeneRatio_num)) %>%
      mutate(
        Description = paste0("[", Ontology, "] ", Description),
        Description = factor(Description, levels = rev(Description))
      )
    
    p <- ggplot(
      go_combined,
      aes(
        x = GeneRatio_num,
        y = Description
      )
    ) +
      geom_point(
        aes(
          size = Count,
          color = p.adjust
        ),
        alpha = 0.85
      ) +
      scale_color_gradient(
        low = "red",
        high = "blue",
        trans = "reverse"
      ) +
      labs(
        title = paste0("Cluster ", cl),
        x = "Gene ratio",
        y = "GO term",
        size = "Gene count",
        color = "Adjusted p-value"
      ) +
      theme_bw(base_size = 14) +
      theme(
        plot.title = element_text(
          hjust = 0.5,
          face = "bold",
          size = 18
        ),
        axis.title = element_text(
          face = "bold",
          size = 16
        ),
        axis.text.y = element_text(
          size = 12
        ),
        axis.text.x = element_text(
          angle = 45,
          hjust = 1,
          size = 12
        ),
        legend.title = element_text(
          face = "bold"
        ),
        panel.grid.major = element_line(
          color = "gray90"
        ),
        panel.grid.minor = element_blank()
      )
    
    plot_list[[paste0("Cluster_", cl)]] <- p
  }
  
  combined_plot <- grid.arrange(
    grobs = plot_list,
    ncol = 1,
    top = textGrob(
      paste0(dataset_name, ": combined GO enrichment of SOTA-derived clusters"),
      gp = gpar(
        fontsize = 20,
        fontface = "bold"
      )
    )
  )
  
  ggsave(
    filename = file.path(
      out_dir,
      paste0(dataset_name, "_combined_GO_clusters_vertical.png")
    ),
    plot = combined_plot,
    width = 16,
    height = 14,
    dpi = 300
  )
  
  return(plot_list)
}

tcga_go_plots <- run_combined_go_enrichment_for_clusters(
  sota_result = tcga_sota_result,
  dataset_name = "TCGA_Cancer",
  top_n = 20
)



# ============================================================
# 7. Formation of cluster-based training and validation datasets
# ============================================================

create_cluster_datasets <- function(train_data,
                                    validation_data,
                                    sota_result,
                                    dataset_name,
                                    class_col = "Class",
                                    out_dir = "SOTA_results_new/cluster_datasets") {
  
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  
  gene_clusters <- sota_result$gene_clusters
  
  clusters <- sort(unique(gene_clusters$Cluster))
  
  result_list <- list()
  
  for (cl in clusters) {
    
    message("Creating cluster dataset: ", dataset_name, ", Cluster ", cl)
    
    cluster_genes <- gene_clusters$Gene_ID[gene_clusters$Cluster == cl]
    
    # Keep only genes that are present in both training and validation data
    common_genes <- intersect(cluster_genes, colnames(train_data))
    common_genes <- intersect(common_genes, colnames(validation_data))
    
    if (length(common_genes) == 0) {
      warning("No common genes for ", dataset_name, " Cluster ", cl)
      next
    }
    
    train_cluster <- train_data[, c(common_genes, class_col), drop = FALSE]
    validation_cluster <- validation_data[, c(common_genes, class_col), drop = FALSE]
    
    train_file <- file.path(
      out_dir,
      paste0(dataset_name, "_Cluster_", cl, "_train.csv")
    )
    
    validation_file <- file.path(
      out_dir,
      paste0(dataset_name, "_Cluster_", cl, "_validation.csv")
    )
    
    write.csv(train_cluster, train_file, row.names = FALSE)
    write.csv(validation_cluster, validation_file, row.names = FALSE)
    
    result_list[[paste0("Cluster_", cl)]] <- list(
      train = train_cluster,
      validation = validation_cluster,
      genes = common_genes,
      n_genes = length(common_genes)
    )
  }
  
  summary_table <- data.frame(
    Dataset = dataset_name,
    Cluster = clusters,
    Number_of_genes = sapply(
      clusters,
      function(cl) {
        key <- paste0("Cluster_", cl)
        if (!is.null(result_list[[key]])) result_list[[key]]$n_genes else 0
      }
    )
  )
  
  write.csv(
    summary_table,
    file.path(out_dir, paste0(dataset_name, "_cluster_dataset_summary.csv")),
    row.names = FALSE
  )
  
  return(list(
    cluster_datasets = result_list,
    summary = summary_table
  ))
}


tcga_cluster_data <- create_cluster_datasets(
  train_data = tcga_data,
  validation_data = cancer_validation,
  sota_result = tcga_sota_result,
  dataset_name = "TCGA_Cancer",
  class_col = "Class"
)

