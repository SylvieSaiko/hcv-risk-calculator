#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = FALSE)
file_arg <- "--file="
script_path <- normalizePath(sub(file_arg, "", args[grep(file_arg, args)]))
script_dir <- dirname(script_path)
site_root <- dirname(script_dir)
attempt_root <- dirname(site_root)
workspace_root <- normalizePath(file.path(attempt_root, "..", ".."))
local_r_lib <- file.path(workspace_root, "NHANES_MINE", "r_libs")
if (dir.exists(local_r_lib)) {
  .libPaths(c(local_r_lib, .libPaths()))
}

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(jsonlite)
  library(xgboost)
})

options(stringsAsFactors = FALSE)

set.seed(20260305L)

analysis_path <- file.path(
  attempt_root,
  "outputs",
  "hcv_demographics_prediction_1999_2023_prespecified_demo5",
  "data",
  "hcv_demographics_prediction_analysis_dataset.rds"
)

perf_path <- file.path(
  attempt_root,
  "outputs",
  "hcv_demographics_prediction_1999_2023_prespecified_demo5",
  "main_tables",
  "table3_model_performance_fixed_split.csv"
)

artifacts_dir <- file.path(site_root, "artifacts")
dir.create(artifacts_dir, recursive = TRUE, showWarnings = FALSE)

required_cols <- c("hcv_broad", "birth_year", "male", "race_ethnicity", "born_usa", "incomepir")
predictor_formula <- ~ birth_year + male + race_ethnicity + born_usa + incomepir - 1

sigmoid <- function(x) 1 / (1 + exp(-x))
clamp_prob <- function(x, eps = 1e-6) pmin(pmax(as.numeric(x), eps), 1 - eps)
to_float32 <- function(x) {
  readBin(writeBin(as.numeric(x), raw(), size = 4), what = "numeric", n = length(x), size = 4)
}

if (!file.exists(analysis_path)) {
  stop("Analysis dataset not found: ", analysis_path)
}

if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("Package 'jsonlite' is required to export the web artifact.")
}

dat <- readRDS(analysis_path) %>%
  mutate(
    hcv_broad = as.integer(hcv_broad),
    birth_year = suppressWarnings(as.numeric(birth_year)),
    male = as.integer(male),
    born_usa = as.integer(born_usa),
    incomepir = suppressWarnings(as.numeric(incomepir)),
    race_ethnicity = factor(
      as.character(race_ethnicity),
      levels = c("White", "Black", "Hispanic", "Asian/Other")
    )
  )

missing_cols <- setdiff(required_cols, names(dat))
if (length(missing_cols) > 0) {
  stop("Dataset missing required columns: ", paste(missing_cols, collapse = ", "))
}

dat <- dat %>%
  filter(if_all(all_of(required_cols), ~ !is.na(.x)))

if (nrow(dat) == 0 || length(unique(dat$hcv_broad)) < 2) {
  stop("Modeling dataset is empty or lacks both outcome classes after filtering.")
}

design_matrix <- model.matrix(predictor_formula, data = dat)
label_vector <- dat$hcv_broad
scale_pos_weight <- sum(label_vector == 0L) / sum(label_vector == 1L)
dmat_cv <- xgboost::xgb.DMatrix(data = design_matrix, label = label_vector)

params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  eta = 0.05,
  max_depth = 3,
  min_child_weight = 1,
  subsample = 0.8,
  colsample_bytree = 0.8,
  scale_pos_weight = scale_pos_weight
)

message("Running 5-fold CV to recover the deployment XGB booster.")
cv_fit <- xgboost::xgb.cv(
  params = params,
  data = dmat_cv,
  nrounds = 300,
  nfold = 5,
  stratified = TRUE,
  showsd = TRUE,
  early_stopping_rounds = 30,
  maximize = TRUE,
  verbose = 0
)

best_nrounds <- cv_fit$best_iteration
if (is.null(best_nrounds) || is.na(best_nrounds)) {
  best_nrounds <- cv_fit$early_stop$best_iteration
}
if (is.null(best_nrounds) || is.na(best_nrounds)) {
  eval_log <- cv_fit$evaluation_log
  auc_col <- if ("test_auc_mean" %in% names(eval_log)) "test_auc_mean" else "test-auc-mean"
  if (!auc_col %in% names(eval_log)) {
    stop("Could not locate test AUC column in xgb.cv() evaluation log.")
  }
  best_nrounds <- which.max(eval_log[[auc_col]])
}
if (is.null(best_nrounds) || is.na(best_nrounds) || best_nrounds < 1) {
  stop("Failed to determine best_nrounds from xgb.cv().")
}

make_stratified_folds <- function(y, nfold = 5L, seed = 20260305L) {
  set.seed(seed)
  fold_id <- integer(length(y))
  pos_idx <- sample(which(y == 1L))
  neg_idx <- sample(which(y == 0L))
  fold_id[pos_idx] <- rep(seq_len(nfold), length.out = length(pos_idx))
  fold_id[neg_idx] <- rep(seq_len(nfold), length.out = length(neg_idx))
  fold_id
}

message("Generating out-of-fold XGB predictions for probability calibration.")
fold_id <- make_stratified_folds(label_vector, nfold = 5L, seed = 20260305L)
oof_raw_predictions <- rep(NA_real_, length(label_vector))
for (fold in sort(unique(fold_id))) {
  train_idx <- which(fold_id != fold)
  valid_idx <- which(fold_id == fold)
  fold_fit <- xgboost::xgb.train(
    params = params,
    data = xgboost::xgb.DMatrix(data = design_matrix[train_idx, , drop = FALSE], label = label_vector[train_idx]),
    nrounds = best_nrounds,
    verbose = 0
  )
  oof_raw_predictions[valid_idx] <- as.numeric(
    predict(
      fold_fit,
      xgboost::xgb.DMatrix(data = design_matrix[valid_idx, , drop = FALSE])
    )
  )
}
if (anyNA(oof_raw_predictions)) {
  stop("Out-of-fold calibration predictions contain missing values.")
}

calibration_df <- tibble(
  outcome = label_vector,
  raw_logit = stats::qlogis(clamp_prob(oof_raw_predictions))
)
calibration_fit <- glm(outcome ~ raw_logit, data = calibration_df, family = binomial())
calibration_coef <- stats::coef(calibration_fit)
calibration_intercept <- unname(calibration_coef[["(Intercept)"]])
calibration_slope <- unname(calibration_coef[["raw_logit"]])
if (!is.finite(calibration_intercept) || !is.finite(calibration_slope)) {
  stop("Calibration fit produced non-finite coefficients.")
}

message("Fitting final XGB model on the full complete-case cohort with ", best_nrounds, " rounds.")
final_fit <- xgboost::xgb.train(
  params = params,
  data = xgboost::xgb.DMatrix(data = design_matrix, label = label_vector),
  nrounds = best_nrounds,
  verbose = 0
)

dump_text <- xgboost::xgb.dump(final_fit, with_stats = FALSE, dump_format = "json")[[1]]
dump_path <- file.path(artifacts_dir, "xgb_model_dump.json")
writeLines(dump_text, dump_path)

trees <- jsonlite::fromJSON(dump_text, simplifyVector = FALSE)

find_child_node <- function(node, child_id) {
  if (is.null(node$children) || length(node$children) == 0) {
    stop("Tree node has no children for requested child_id=", child_id)
  }
  child_ids <- vapply(node$children, function(child) as.integer(child$nodeid), integer(1))
  idx <- match(as.integer(child_id), child_ids)
  if (is.na(idx)) {
    stop("Unable to find child node ", child_id, " beneath split node ", node$nodeid)
  }
  node$children[[idx]]
}

score_tree <- function(node, feature_row) {
  if (!is.null(node$leaf)) {
    return(as.numeric(node$leaf))
  }

  split_name <- as.character(node$split)
  split_value <- to_float32(node$split_condition)
  feature_value <- suppressWarnings(to_float32(feature_row[[split_name]]))

  next_id <- if (is.na(feature_value)) {
    as.integer(node$missing)
  } else if (feature_value < split_value) {
    as.integer(node$yes)
  } else {
    as.integer(node$no)
  }

  score_tree(find_child_node(node, next_id), feature_row)
}

score_all_trees <- function(tree_list, feature_row) {
  sum(vapply(tree_list, score_tree, numeric(1), feature_row = feature_row))
}

pred_margin <- as.numeric(
  predict(
    final_fit,
    xgboost::xgb.DMatrix(data = design_matrix[1, , drop = FALSE]),
    outputmargin = TRUE
  )
)
tree_margin <- score_all_trees(trees, as.list(design_matrix[1, ]))
base_margin <- pred_margin - tree_margin

raw_predictions <- as.numeric(
  predict(
    final_fit,
    xgboost::xgb.DMatrix(data = design_matrix)
  )
)
predictions <- sigmoid(
  calibration_intercept + calibration_slope * stats::qlogis(clamp_prob(raw_predictions))
)

verification_idx <- unique(c(1L, sample.int(nrow(design_matrix), min(250L, nrow(design_matrix)))))
replayed_predictions <- vapply(
  verification_idx,
  function(i) {
    margin <- base_margin + score_all_trees(trees, as.list(design_matrix[i, ]))
    sigmoid(margin)
  },
  numeric(1)
)
max_abs_diff <- max(abs(replayed_predictions - raw_predictions[verification_idx]))
if (max_abs_diff > 1e-6) {
  stop("Exported tree replay does not match XGBoost predictions. max_abs_diff=", signif(max_abs_diff, 6))
}

decile_thresholds <- as.numeric(stats::quantile(predictions, probs = seq(0.1, 0.9, by = 0.1), type = 8, names = FALSE))

pred_df <- tibble(
  pred = predictions,
  outcome = label_vector
)

pred_df$decile <- dplyr::ntile(pred_df$pred, 10L)
decile_stats <- pred_df %>%
  group_by(decile) %>%
  summarise(
    n = n(),
    predicted_risk_mean = mean(pred),
    observed_prevalence = mean(outcome),
    risk_min = min(pred),
    risk_max = max(pred),
    .groups = "drop"
  )
decile_stats_list <- lapply(seq_len(nrow(decile_stats)), function(i) {
  as.list(decile_stats[i, , drop = FALSE])
})

validation_xgb <- NULL
if (file.exists(perf_path)) {
  validation_xgb <- readr::read_csv(perf_path, show_col_types = FALSE) %>%
    filter(split == "fixed_cycle", set == "Validation", model == "XGB") %>%
    slice(1) %>%
    transmute(
      auroc = auroc,
      pr_auc = pr_auc,
      sensitivity = sensitivity,
      specificity = specificity,
      brier = brier,
      threshold = threshold
    )
  if (nrow(validation_xgb) == 0) {
    validation_xgb <- NULL
  } else {
    validation_xgb <- as.list(validation_xgb[1, , drop = FALSE])
  }
}

artifact <- list(
  model_name = "attempt_5_demographic_xgb_full_cohort",
  outcome = "Broad HCV positivity",
  exported_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"),
  cohort = list(
    size = nrow(dat),
    positive_cases = as.integer(sum(label_vector == 1L)),
    prevalence = as.numeric(mean(label_vector)),
    source_dataset = normalizePath(analysis_path, winslash = "/")
  ),
  training = list(
    recovered_from = "attempt_5 prespecified_demo5 XGB pipeline",
    feature_mode = "prespecified_demo5",
    predictor_columns = colnames(design_matrix),
    xgb_params = params,
    best_nrounds_from_cv = as.integer(best_nrounds),
    max_abs_replay_error = as.numeric(max_abs_diff),
    calibration = list(
      method = "Platt scaling",
      source = "5-fold out-of-fold XGB predictions",
      intercept = as.numeric(calibration_intercept),
      slope = as.numeric(calibration_slope)
    )
  ),
  scoring = list(
    base_margin = as.numeric(base_margin),
    objective = "binary:logistic",
    calibration = list(
      method = "platt",
      intercept = as.numeric(calibration_intercept),
      slope = as.numeric(calibration_slope)
    ),
    trees = trees
  ),
  inputs = list(
    birth_year = list(type = "number", min = 1900, max = 2100),
    sex = list(type = "categorical", levels = c("Female", "Male")),
    birthplace = list(type = "categorical", levels = c("Not_USA", "USA")),
    ethnicity = list(type = "categorical", levels = c("White", "Black", "Hispanic", "Asian/Other")),
    incomepir = list(type = "number", min = 0, max = 5)
  ),
  deciles = list(
    thresholds = decile_thresholds,
    stats = decile_stats_list
  ),
  reference_validation = validation_xgb
)

artifact_json <- jsonlite::toJSON(
  artifact,
  auto_unbox = TRUE,
  digits = 16,
  pretty = TRUE,
  null = "null"
)

artifact_json_path <- file.path(artifacts_dir, "xgb_web_artifact.json")
writeLines(artifact_json, artifact_json_path)

model_data_js_path <- file.path(site_root, "model-data.js")
writeLines(
  c(
    "window.HCV_XGB_MODEL = ",
    artifact_json,
    ";"
  ),
  model_data_js_path
)

message("Wrote XGB artifact: ", artifact_json_path)
message("Wrote browser model bundle: ", model_data_js_path)
