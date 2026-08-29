#!/usr/bin/env Rscript

required <- c("ggplot2", "patchwork")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) > 0) {
  stop("Missing existing R packages: ", paste(missing, collapse = ", "), ". No package installation is performed.")
}

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
})

args <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", args[grep("^--file=", args)])
script_path <- normalizePath(file_arg, winslash = "/", mustWork = TRUE)
project_root <- dirname(dirname(script_path))
evidence_root <- file.path(project_root, "artifacts", "paper_evidence_v1")
table_root <- file.path(evidence_root, "tables")
latency_root <- file.path(evidence_root, "latency")
figure_root <- file.path(evidence_root, "figures")
dir.create(figure_root, recursive = TRUE, showWarnings = FALSE)

paths <- c(
  runs = file.path(table_root, "run_index.csv"),
  effects = file.path(table_root, "paired_effects_run_level.csv"),
  conditions = file.path(table_root, "paired_effects_condition_level.csv"),
  latency = file.path(latency_root, "latency_condition_aggregates.csv")
)
missing_paths <- paths[!file.exists(paths)]
if (length(missing_paths) > 0) {
  stop("Figure 6 inputs are missing: ", paste(missing_paths, collapse = ", "))
}

runs <- read.csv(paths[["runs"]], stringsAsFactors = FALSE, check.names = FALSE)
effects <- read.csv(paths[["effects"]], stringsAsFactors = FALSE, check.names = FALSE)
conditions <- read.csv(
  paths[["conditions"]],
  stringsAsFactors = FALSE,
  check.names = FALSE,
  colClasses = c(recording = "character")
)
latency <- read.csv(paths[["latency"]], stringsAsFactors = FALSE, check.names = FALSE)

black <- "#161616"
grey <- "#6B6B6B"
light_grey <- "#E8E8E8"
blue <- "#0072B2"
sky <- "#56B4E9"
orange <- "#E69F00"
green <- "#009E73"
vermillion <- "#D55E00"
palette <- c(blue, orange, sky, green, vermillion)
font_family <- "Arial"

theme_nature <- function(base_size = 7.2) {
  theme_classic(base_size = base_size, base_family = font_family) +
    theme(
      axis.line = element_line(linewidth = 0.35, colour = black),
      axis.ticks = element_line(linewidth = 0.35, colour = black),
      axis.text = element_text(size = 6.6, colour = black),
      axis.title = element_text(size = 7.1, colour = black),
      legend.title = element_blank(),
      legend.text = element_text(size = 6.2, colour = black),
      plot.tag = element_text(size = 8.2, face = "bold", colour = black),
      plot.margin = margin(3, 4, 3, 3, unit = "mm")
    )
}

label_map <- c(
  paper_fullres_b8d3_mae = "G1 Full-res / MAE",
  paper_fullres_b8d3_mse = "G2 Full-res / MSE",
  paper_subpixel_b8d3_mae = "G3 Subpixel / MAE",
  paper_subpixel_b8d3_mse = "G4 Subpixel / MSE",
  paper_fullres_b8d4_mse = "G5 B8-D4 / MSE"
)
effect_labels <- c(
  structure_mae = "G3-G1",
  structure_mse = "G4-G2",
  objective_fullres = "G2-G1",
  objective_subpixel = "G4-G3",
  capacity_reference = "G5-G4"
)

# a: factorial design encoded as a compact matrix rather than a text flow chart.
factorial <- data.frame(
  structure = factor(c("Full resolution", "Subpixel", "Full resolution", "Subpixel"), levels = c("Full resolution", "Subpixel")),
  objective = factor(c("MAE", "MAE", "MSE", "MSE"), levels = c("MSE", "MAE")),
  group = c("G1", "G3", "G2", "G4")
)
panel_a <- ggplot(factorial, aes(structure, objective, fill = structure)) +
  geom_tile(width = 0.86, height = 0.78, colour = black, linewidth = 0.45) +
  geom_text(aes(label = group), family = font_family, fontface = "bold", size = 3.0, colour = black) +
  scale_fill_manual(values = c("Full resolution" = "#DCEAF4", "Subpixel" = "#FCE5C2"), guide = "none") +
  labs(x = "Computation", y = "Training objective") +
  coord_cartesian(clip = "off") +
  theme_nature() +
  theme(axis.line = element_blank(), axis.ticks = element_blank(), panel.border = element_rect(fill = NA, colour = black, linewidth = 0.35))

# b: distributions display spread and boundaries; the box gives median and IQR.
selected_effects <- names(effect_labels)
effect_plot <- effects[effects$effect %in% selected_effects, ]
effect_plot$effect_label <- factor(effect_labels[effect_plot$effect], levels = unname(effect_labels))
panel_b <- ggplot(effect_plot, aes(effect_label, delta_psnr_db, fill = effect_label)) +
  geom_hline(yintercept = 0, linewidth = 0.4, colour = grey, linetype = 2) +
  geom_violin(width = 0.82, scale = "width", trim = TRUE, colour = black, linewidth = 0.35, alpha = 0.72) +
  geom_boxplot(width = 0.18, outlier.shape = NA, fill = "white", colour = black, linewidth = 0.35) +
  stat_summary(fun = mean, geom = "crossbar", width = 0.34, linewidth = 0.45, colour = black) +
  scale_fill_manual(values = setNames(palette, unname(effect_labels)), guide = "none") +
  labs(x = NULL, y = expression(Delta*"PSNR (dB)")) +
  theme_nature() +
  theme(axis.text.x = element_text(angle = 35, hjust = 1, vjust = 1))

# c: Pareto evidence joins quality and measured device cost. Lines expose the frontier scale.
quality <- aggregate(mean_psnr_db ~ variant, runs[runs$variant %in% names(label_map), ], mean)
latency_b1 <- latency[latency$batch_size == 1 & latency$variant %in% names(label_map), ]
latency_mean <- aggregate(median_of_round_p50_ms ~ variant, latency_b1, median)
pareto <- merge(quality, latency_mean, by = "variant", all = FALSE)
pareto$label <- unname(label_map[pareto$variant])
pareto$group <- factor(pareto$label, levels = unname(label_map))
pareto <- pareto[order(pareto$median_of_round_p50_ms), ]
panel_c <- ggplot(pareto, aes(median_of_round_p50_ms, mean_psnr_db, colour = group)) +
  geom_path(colour = light_grey, linewidth = 0.65) +
  geom_point(shape = 21, aes(fill = group), colour = black, stroke = 0.4, size = 2.5) +
  scale_colour_manual(values = setNames(palette, unname(label_map)), guide = "none") +
  scale_fill_manual(values = setNames(palette, unname(label_map)), guide = guide_legend(nrow = 3, byrow = TRUE)) +
  labs(x = "Batch-1 p50 latency (ms)", y = "Mean PSNR (dB)") +
  theme_nature() +
  theme(legend.position = "bottom", legend.key.width = unit(3, "mm"))

# d: nine-condition structure effect, with values printed for direct audit.
heat <- conditions[conditions$effect == "structure_mse", ]
heat$window <- factor(heat$window, levels = c("w1", "w2", "w4"))
heat$recording <- factor(heat$recording, levels = c("01.00", "01.01", "03.00"))
bound <- max(abs(heat$mean_delta_psnr_db))
panel_d <- ggplot(heat, aes(window, recording, fill = mean_delta_psnr_db)) +
  geom_tile(colour = "white", linewidth = 1.0) +
  geom_text(aes(label = sprintf("%+.3f", mean_delta_psnr_db)), family = font_family, size = 2.25, colour = black) +
  scale_fill_gradient2(low = blue, mid = "white", high = vermillion, midpoint = 0, limits = c(-bound, bound), name = expression(Delta*"PSNR")) +
  labs(x = "Temporal window", y = "Recording") +
  coord_equal() +
  theme_nature() +
  theme(legend.position = "bottom", legend.key.width = unit(8, "mm"))

# e: numeric table anchors the plots to the actual capacity and device values.
complexity <- aggregate(cbind(parameters, conv_gflops, mean_psnr_db) ~ variant, runs[runs$variant %in% names(label_map), ], mean)
complexity <- merge(complexity, latency_mean, by = "variant", all.x = TRUE)
complexity$Model <- unname(label_map[complexity$variant])
complexity$Parameters <- sprintf("%.0fK", complexity$parameters / 1000)
complexity$GFLOPs <- sprintf("%.2f", complexity$conv_gflops)
complexity$PSNR <- sprintf("%.3f", complexity$mean_psnr_db)
complexity$Latency <- sprintf("%.3f", complexity$median_of_round_p50_ms)
complexity <- complexity[match(unname(label_map), complexity$Model), c("Model", "Parameters", "GFLOPs", "PSNR", "Latency")]
long_table <- reshape(
  complexity,
  varying = names(complexity)[-1],
  v.names = "value",
  timevar = "metric",
  times = names(complexity)[-1],
  direction = "long"
)
long_table$row <- factor(long_table$Model, levels = rev(complexity$Model))
long_table$metric <- factor(long_table$metric, levels = names(complexity)[-1])
panel_e <- ggplot(long_table, aes(metric, row)) +
  geom_tile(aes(fill = metric), colour = "white", linewidth = 0.8) +
  geom_text(aes(label = value), family = font_family, size = 2.25, colour = black) +
  scale_fill_manual(values = c(Parameters = "#F1F1F1", GFLOPs = "#E3EDF5", PSNR = "#FCE8C6", Latency = "#E1F1EA"), guide = "none") +
  labs(x = NULL, y = NULL) +
  theme_minimal(base_size = 7, base_family = font_family) +
  theme(
    panel.grid = element_blank(),
    axis.ticks = element_blank(),
    axis.text.x = element_text(size = 6.5, colour = black, face = "bold"),
    axis.text.y = element_text(size = 6.2, colour = black),
    plot.margin = margin(3, 4, 3, 3, unit = "mm")
  )

figure_6 <- (panel_a | panel_b) / (panel_c | panel_d) / panel_e +
  plot_layout(heights = c(1.0, 1.05, 0.86), guides = "keep") +
  plot_annotation(tag_levels = "a", theme = theme(plot.tag = element_text(family = font_family, face = "bold", size = 8.2)))

save_figure <- function(plot, stem, width_mm = 180, height_mm = 205) {
  cairo_pdf(file.path(figure_root, paste0(stem, ".pdf")), width = width_mm / 25.4, height = height_mm / 25.4, family = font_family)
  print(plot)
  dev.off()
  svg(file.path(figure_root, paste0(stem, ".svg")), width = width_mm / 25.4, height = height_mm / 25.4, family = font_family)
  print(plot)
  dev.off()
  png(file.path(figure_root, paste0(stem, ".png")), width = width_mm, height = height_mm, units = "mm", res = 600, type = "cairo", bg = "white")
  print(plot)
  dev.off()
}

save_figure(figure_6, "Figure6_objective_aligned_subpixel_evidence")
message("Wrote Figure 6 to ", normalizePath(figure_root, winslash = "/", mustWork = TRUE))
