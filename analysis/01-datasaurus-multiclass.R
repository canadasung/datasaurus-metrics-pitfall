# Replication of Julia Silge's "Multiclass predictive modeling for the
# Datasaurus Dozen" blog post.
#
# Source: https://juliasilge.com/blog/datasaurus-multiclass/
#
# This script explores the Datasaurus Dozen dataset, fits a random forest
# classifier that predicts which of the 13 datasets a point belongs to
# based on its x and y coordinates, and evaluates the model with
# bootstrap resampling. Plots are saved to the output/ directory.

library(tidyverse)
library(datasauRus)
library(tidymodels)
library(doParallel)

# Create the output directory if it does not already exist.
if (!dir.exists("output")) {
  dir.create("output")
}

## ---------------------------------------------------------------------------
## Explore the data
## ---------------------------------------------------------------------------

datasaurus_dozen

# Each of the 13 datasets looks very different when plotted.
dataset_facets <- datasaurus_dozen |>
  ggplot(aes(x, y, color = dataset)) +
  geom_point(show.legend = FALSE) +
  facet_wrap(~dataset, ncol = 5)

dataset_facets

ggsave("output/01-dataset-facets.png", dataset_facets, width = 10, height = 6)

# Despite looking very different, the datasets share nearly identical
# summary statistics.
summary_table <- datasaurus_dozen |>
  group_by(dataset) |>
  summarise(
    across(c(x, y), list(mean = mean, sd = sd)),
    x_y_cor = cor(x, y)
  )

summary_table

summary_table_grob <- summary_table |>
  mutate(across(where(is.numeric), \(x) round(x, 1))) |>
  gridExtra::tableGrob(rows = NULL)

ggsave("output/00-summary-table.png", summary_table_grob, width = 8, height = 6)

# Each dataset contains the same number of points.
datasaurus_dozen |>
  count(dataset)

## ---------------------------------------------------------------------------
## Build a model
## ---------------------------------------------------------------------------

set.seed(123)
dino_folds <- datasaurus_dozen |>
  mutate(dataset = factor(dataset)) |>
  bootstraps()

dino_folds

rf_spec <- rand_forest(trees = 1000) |>
  set_mode("classification") |>
  set_engine("ranger")

dino_wf <- workflow() |>
  add_model(rf_spec) |>
  add_formula(dataset ~ x + y)

dino_wf

## ---------------------------------------------------------------------------
## Fit the model to the resamples
## ---------------------------------------------------------------------------

doParallel::registerDoParallel()

dino_rs <- fit_resamples(
  dino_wf,
  resamples = dino_folds,
  control = control_resamples(save_pred = TRUE)
)

dino_rs

## ---------------------------------------------------------------------------
## Evaluate the model
## ---------------------------------------------------------------------------

metrics_table <- collect_metrics(dino_rs)

metrics_table

# Summarize per-resample PPV (precision) the same way collect_metrics()
# summarizes accuracy, roc_auc, and brier_class: mean and standard error
# across the 25 bootstrap resamples.
ppv_by_resample <- dino_rs |>
  collect_predictions() |>
  group_by(id) |>
  ppv(dataset, .pred_class)

ppv_by_resample

ppv_summary <- ppv_by_resample |>
  ungroup() |>
  summarise(
    .metric = "precision",
    .estimator = "macro",
    mean = mean(.estimate),
    n = n(),
    std_err = sd(.estimate) / sqrt(n()),
    .config = NA_character_
  )

metrics_table_combined <- bind_rows(metrics_table, ppv_summary)

metrics_table_grob <- metrics_table_combined |>
  mutate(across(where(is.numeric), \(x) round(x, 3))) |>
  gridExtra::tableGrob(rows = NULL)

ggsave("output/05-metrics-table.png", metrics_table_grob, width = 6, height = 2)

# ROC curves for each of the 13 classes across resamples.
roc_curves <- dino_rs |>
  collect_predictions() |>
  group_by(id) |>
  roc_curve(dataset, .pred_away:.pred_x_shape) |>
  ggplot(aes(1 - specificity, sensitivity, color = id)) +
  geom_abline(lty = 2, color = "gray80", linewidth = 1.5) +
  geom_path(show.legend = FALSE, alpha = 0.6, linewidth = 1.2) +
  facet_wrap(~.level, ncol = 5) +
  coord_equal()

roc_curves

ggsave("output/02-roc-curves.png", roc_curves, width = 10, height = 6)

# Confusion matrix across all resamples.
dino_rs |>
  collect_predictions() |>
  conf_mat(dataset, .pred_class)

conf_mat_heatmap <- dino_rs |>
  collect_predictions() |>
  conf_mat(dataset, .pred_class) |>
  autoplot(type = "heatmap") +
  scale_fill_gradient(low = "white", high = "steelblue")

conf_mat_heatmap

ggsave("output/03-confusion-matrix.png", conf_mat_heatmap, width = 8, height = 7)

# Confusion matrix filtered to misclassifications only.
conf_mat_misclassified <- dino_rs |>
  collect_predictions() |>
  filter(.pred_class != dataset) |>
  conf_mat(dataset, .pred_class) |>
  autoplot(type = "heatmap") +
  scale_fill_gradient(low = "white", high = "steelblue")

conf_mat_misclassified

ggsave("output/04-confusion-matrix-misclassified.png", conf_mat_misclassified, width = 8, height = 7)
