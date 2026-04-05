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
  library(pROC)
  library(xgboost)
})

options(stringsAsFactors = FALSE)

set.seed(20260305L)

output_root <- file.path(
  attempt_root,
  "outputs",
  "hcv_demographics_prediction_1999_2023_prespecified_demo5"
)

analysis_path <- file.path(
  output_root,
  "data",
  "hcv_demographics_prediction_analysis_dataset.rds"
)

perf_path <- file.path(
  output_root,
  "main_tables",
  "table3_model_performance_fixed_split.csv"
)

calibration_path <- file.path(
  output_root,
  "supplement",
  "model_probability_calibration_fixed_cycle.csv"
)

prediction_path <- file.path(
  output_root,
  "data",
  "predictions_all_splits.csv"
)

split_manifest_path <- file.path(
  output_root,
  "supplement",
  "split_manifest.csv"
)

artifacts_dir <- file.path(site_root, "artifacts")
dir.create(artifacts_dir, recursive = TRUE, showWarnings = FALSE)

required_cols <- c(
  "hcv_broad",
  "birth_year",
  "male",
  "race_ethnicity",
  "born_usa",
  "incomepir",
  "cycle_index"
)
predictor_formula <- ~ birth_year + male + race_ethnicity + born_usa + incomepir - 1

sigmoid <- function(x) 1 / (1 + exp(-x))
clamp_prob <- function(x, eps = 1e-6) pmin(pmax(as.numeric(x), eps), 1 - eps)
to_float32 <- function(x) {
  readBin(writeBin(as.numeric(x), raw(), size = 4), what = "numeric", n = length(x), size = 4)
}

required_paths <- c(
  analysis_path,
  perf_path,
  calibration_path,
  prediction_path,
  split_manifest_path
)
missing_paths <- required_paths[!file.exists(required_paths)]
if (length(missing_paths) > 0) {
  stop("Required source files not found: ", paste(missing_paths, collapse = ", "))
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
    ),
    cycle_index = suppressWarnings(as.integer(cycle_index))
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

paper_validation_cycles <- c(2L, 4L, 12L)
paper_training_cycles <- setdiff(sort(unique(dat$cycle_index)), paper_validation_cycles)

train_df <- dat %>%
  filter(cycle_index %in% paper_training_cycles)
valid_df <- dat %>%
  filter(cycle_index %in% paper_validation_cycles)

if (nrow(train_df) == 0 || nrow(valid_df) == 0) {
  stop("Fixed-cycle split produced an empty training or validation set.")
}

split_manifest <- readr::read_csv(split_manifest_path, show_col_types = FALSE)
expected_train_n <- split_manifest %>%
  filter(split == "fixed_cycle", set == "Training") %>%
  pull(n) %>%
  .[[1]]
expected_valid_n <- split_manifest %>%
  filter(split == "fixed_cycle", set == "Validation") %>%
  pull(n) %>%
  .[[1]]

if (nrow(train_df) != expected_train_n || nrow(valid_df) != expected_valid_n) {
  stop(
    "Fixed-cycle split counts do not match the paper source. ",
    "Expected train/valid=", expected_train_n, "/", expected_valid_n,
    "; got ", nrow(train_df), "/", nrow(valid_df), "."
  )
}

train_x <- model.matrix(predictor_formula, data = train_df)
valid_x <- model.matrix(predictor_formula, data = valid_df)
all_x <- rbind(train_x, valid_x)

scale_pos_weight <- sum(train_df$hcv_broad == 0L) / sum(train_df$hcv_broad == 1L)
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

message("Fitting fixed-cycle XGB model on the primary analysis training cohort.")
fixed_fit <- xgboost::xgb.train(
  params = params,
  data = xgboost::xgb.DMatrix(data = train_x, label = train_df$hcv_broad),
  nrounds = 300,
  evals = list(
    train = xgboost::xgb.DMatrix(data = train_x, label = train_df$hcv_broad),
    valid = xgboost::xgb.DMatrix(data = valid_x, label = valid_df$hcv_broad)
  ),
  early_stopping_rounds = 30,
  verbose = 0
)

best_nrounds <- fixed_fit$best_iteration
if (is.null(best_nrounds) || is.na(best_nrounds) || best_nrounds < 1) {
  best_nrounds <- fixed_fit$early_stop$best_iteration
}
if (is.null(best_nrounds) || is.na(best_nrounds) || best_nrounds < 1) {
  best_attr <- xgboost::xgb.attributes(fixed_fit)[["best_iteration"]]
  if (!is.null(best_attr) && length(best_attr) == 1) {
    best_nrounds <- as.integer(best_attr) + 1L
  }
}
if (is.null(best_nrounds) || is.na(best_nrounds) || best_nrounds < 1) {
  stop("Unable to recover best_nrounds from the fitted XGBoost model.")
}
if (best_nrounds < xgboost::xgb.get.num.boosted.rounds(fixed_fit)) {
  fixed_fit <- xgboost::xgb.slice.Booster(
    fixed_fit,
    start = 1L,
    end = as.integer(best_nrounds)
  )
}

train_raw_predictions <- as.numeric(
  predict(
    fixed_fit,
    xgboost::xgb.DMatrix(data = train_x)
  )
)
valid_raw_predictions <- as.numeric(
  predict(
    fixed_fit,
    xgboost::xgb.DMatrix(data = valid_x)
  )
)
all_raw_predictions <- c(train_raw_predictions, valid_raw_predictions)

calibration_row <- readr::read_csv(calibration_path, show_col_types = FALSE) %>%
  filter(split == "fixed_cycle", model == "XGB") %>%
  slice(1)
if (nrow(calibration_row) == 0) {
  stop("Missing fixed-cycle XGB calibration row in: ", calibration_path)
}

calibration_intercept <- as.numeric(calibration_row$calibration_intercept[[1]])
calibration_slope <- as.numeric(calibration_row$calibration_slope[[1]])
if (!is.finite(calibration_intercept) || !is.finite(calibration_slope)) {
  stop("Calibration file contains non-finite coefficients.")
}

perf_row <- readr::read_csv(perf_path, show_col_types = FALSE) %>%
  filter(split == "fixed_cycle", set == "Validation", model == "XGB") %>%
  slice(1)
if (nrow(perf_row) == 0) {
  stop("Missing fixed-cycle validation XGB row in: ", perf_path)
}

validation_xgb <- list(
  split = "fixed_cycle",
  set = "Validation",
  reference_label = "Fixed-cycle validation cohort used in the paper",
  reference_set_size = nrow(valid_df),
  auroc = as.numeric(perf_row$auroc[[1]]),
  pr_auc = as.numeric(perf_row$pr_auc[[1]]),
  sensitivity = as.numeric(perf_row$sensitivity[[1]]),
  specificity = as.numeric(perf_row$specificity[[1]]),
  brier = as.numeric(perf_row$brier[[1]]),
  threshold = as.numeric(perf_row$threshold[[1]])
)

reproduced_validation_auroc <- as.numeric(
  pROC::auc(
    pROC::roc(
      response = valid_df$hcv_broad,
      predictor = valid_raw_predictions,
      quiet = TRUE
    )
  )
)
validation_auroc_abs_diff <- abs(reproduced_validation_auroc - validation_xgb$auroc)
if (!is.finite(validation_auroc_abs_diff) || validation_auroc_abs_diff > 0.01) {
  stop(
    "Rebuilt fixed-cycle XGB AUROC diverged from the paper reference. ",
    "Expected ", signif(validation_xgb$auroc, 6),
    ", got ", signif(reproduced_validation_auroc, 6), "."
  )
}

dump_text <- xgboost::xgb.dump(fixed_fit, with_stats = FALSE, dump_format = "json")[[1]]
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
    fixed_fit,
    xgboost::xgb.DMatrix(data = train_x[1, , drop = FALSE]),
    outputmargin = TRUE
  )
)
tree_margin <- score_all_trees(trees, as.list(train_x[1, ]))
base_margin <- pred_margin - tree_margin

verification_idx <- unique(c(1L, sample.int(nrow(all_x), min(250L, nrow(all_x)))))
replayed_predictions <- vapply(
  verification_idx,
  function(i) {
    margin <- base_margin + score_all_trees(trees, as.list(all_x[i, ]))
    sigmoid(margin)
  },
  numeric(1)
)
max_abs_diff <- max(abs(replayed_predictions - all_raw_predictions[verification_idx]))
if (max_abs_diff > 1e-6) {
  stop("Exported tree replay does not match XGBoost predictions. max_abs_diff=", signif(max_abs_diff, 6))
}

paper_validation_predictions <- readr::read_csv(prediction_path, show_col_types = FALSE) %>%
  filter(split == "fixed_cycle", set == "Validation", model == "XGB") %>%
  transmute(
    pred = sigmoid(calibration_intercept + calibration_slope * stats::qlogis(clamp_prob(pred))),
    outcome = as.integer(outcome)
  )
if (nrow(paper_validation_predictions) != validation_xgb$reference_set_size) {
  stop(
    "Paper validation prediction count mismatch. Expected ",
    validation_xgb$reference_set_size,
    ", got ",
    nrow(paper_validation_predictions),
    "."
  )
}

paper_validation_predictions <- paper_validation_predictions %>%
  mutate(decile = dplyr::ntile(pred, 10L))

decile_stats <- paper_validation_predictions %>%
  group_by(decile) %>%
  summarise(
    n = n(),
    predicted_risk_mean = mean(pred),
    observed_prevalence = mean(outcome),
    risk_min = min(pred),
    risk_max = max(pred),
    .groups = "drop"
  ) %>%
  arrange(decile)

if (nrow(decile_stats) != 10) {
  stop("Expected 10 deciles in the paper validation reference, found ", nrow(decile_stats), ".")
}

decile_thresholds <- decile_stats %>%
  filter(decile < 10) %>%
  pull(risk_max) %>%
  as.numeric()
decile_stats_list <- lapply(seq_len(nrow(decile_stats)), function(i) {
  as.list(decile_stats[i, , drop = FALSE])
})

artifact <- list(
  model_name = "attempt_5_demographic_xgb_fixed_cycle_primary_analysis",
  display_name = "FIXED-CYCLE DEMOGRAPHIC XGB",
  outcome = "Evidence of current or past HCV infection",
  exported_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"),
  cohort = list(
    size = nrow(dat),
    positive_cases = as.integer(sum(dat$hcv_broad == 1L)),
    prevalence = as.numeric(mean(dat$hcv_broad)),
    source_dataset = normalizePath(analysis_path, winslash = "/")
  ),
  training = list(
    recovered_from = "attempt_5 prespecified_demo5 fixed-cycle primary analysis",
    feature_mode = "prespecified_demo5",
    split = "fixed_cycle",
    training_cycles = as.integer(sort(paper_training_cycles)),
    validation_cycles = as.integer(sort(paper_validation_cycles)),
    training_n = nrow(train_df),
    validation_n = nrow(valid_df),
    predictor_columns = colnames(train_x),
    xgb_params = params,
    best_nrounds = as.integer(best_nrounds),
    max_abs_replay_error = as.numeric(max_abs_diff),
    calibration = list(
      method = "Platt scaling",
      source = normalizePath(calibration_path, winslash = "/"),
      fitted_on = "fixed-cycle training cohort",
      intercept = calibration_intercept,
      slope = calibration_slope
    )
  ),
  scoring = list(
    base_margin = as.numeric(base_margin),
    objective = "binary:logistic",
    calibration = list(
      method = "platt",
      intercept = calibration_intercept,
      slope = calibration_slope
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
    calibrated = TRUE,
    reference_set = "fixed_cycle_validation",
    reference_label = "fixed-cycle validation cohort used in the paper",
    reference_size = nrow(paper_validation_predictions),
    thresholds = decile_thresholds,
    stats = decile_stats_list
  ),
  reference_validation = validation_xgb,
  paper_alignment = list(
    validation_auroc_rebuilt = reproduced_validation_auroc,
    validation_auroc_reference = validation_xgb$auroc,
    validation_auroc_abs_diff = validation_auroc_abs_diff
  )
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
