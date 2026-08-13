library(ggplot2)
library(dplyr)


data <- read.csv("tcga_cancer.csv", check.names = FALSE)

data_no_normal <- data[data$Class != "normal", ]

table(data_no_normal$Class)

write.csv(
  data_no_normal,
  "tcga_cancer_without_normal.csv",
  row.names = FALSE
)

library(ggplot2)

plot_class_distribution <- function(
    data,
    class_column = "Class",
    output_file = "TCGA_classes_visualization.png",
    width = 12,
    height = 7,
    dpi = 300
) {
  
  class_vec <- as.character(data[[class_column]])
  class_vec <- class_vec[class_vec != "normal"]
  
  class_counts <- as.data.frame(table(class_vec))
  colnames(class_counts) <- c("Class", "Count")
  
  class_counts <- class_counts[order(-class_counts$Count), ]
  
  p <- ggplot(
    class_counts,
    aes(x = reorder(Class, -Count), y = Count, fill = Class)
  ) +
    geom_col(show.legend = FALSE) +
    geom_text(aes(label = Count), vjust = -0.3, size = 6) +
    labs(
      title = "Distribution of samples across TCGA classes",
      x = "Class",
      y = "Number of samples"
    ) +
    theme_minimal(base_size = 18) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 24),
      axis.text.x = element_text(angle = 45, hjust = 1)
    )
  
  print(p)
  
  ggsave(output_file, plot = p, width = width, height = height, dpi = dpi)
  
  return(class_counts)
}

plot_class_distribution(
  data = data_no_normal,
  class_column = "Class",
  output_file = "TCGA_classes_visualization.png"
)