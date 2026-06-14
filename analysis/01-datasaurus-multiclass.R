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
datasaurus_dozen |>
  group_by(dataset) |>
  summarise(
    across(c(x, y), list(mean = mean, sd = sd)),
    x_y_cor = cor(x, y)
  )

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
  add_formula(dataset ~ x + y) |>
  add_model(rf_spec)

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

collect_metrics(dino_rs)

dino_rs |>
  collect_predictions() |>
  group_by(id) |>
  ppv(dataset, .pred_class)

# ROC curves for each of the 13 classes across resamples.
roc_curves <- dino_rs |>
  collect_predictions() |>
  group_by(id) |>
  roc_curve(dataset, .pred_away:.pred_x_shape) |>
  ggplot(aes(1 - specificity, sensitivity, color = id)) +
  geom_abline(lty = 2, color = "gray80", size = 1.5) +
  geom_path(show.legend = FALSE, alpha = 0.6, size = 1.2) +
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

dino_rs

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
