# run_experiment using DelphiRF

library(DelphiRF)
library(rlang)
library(dplyr)
library(tidyr)
library(ggplot2)

# Load config from external file
source("madph_config.R")

# ------------------------
# HELPER FUNCTIONS
# ------------------------

# -------------------------------------------------------------------
# Function: prepare_data
# Purpose : Preprocess input data for a given test window using
#           configuration-defined column names and parameters.
#
# Description:
#   This function filters the input dataset based on reporting delays
#   and relevant date constraints. It then applies DelphiRF's
#   `data_preprocessing()` using column names and parameters provided
#   in the `config` list. The function also measures the runtime of
#   the preprocessing step and returns it alongside the filtered data.
#
# Arguments:
#   data            - The raw input dataset (e.g., ma_dph).
#   test_start_date - The start of the current test window.
#   test_end_date   - The end of the current test window (exclusive).
#   config          - A named list containing all parameter and column
#                     configurations. Must include:
#                     * report_date_col (e.g., "issue_date")
#                     * lag_col
#                     * value_col
#                     * refd_col
#                     * ref_lag
#                     * smoothed_raw (boolean)
#                     * testing_window
#
# Returns:
#   A preprocessed and filtered data frame ready for DelphiRF modeling.
# -------------------------------------------------------------------
prepare_data <- function(data, test_start_date, test_end_date, config) {
  data %>%
    dplyr::filter(.data[[config$report_date_col]] < (test_start_date + config$ref_lag * 2)) %>%
    dplyr::filter(.data[[config$lag_col]] <= (config$ref_lag + 100)) %>%
    data_preprocessing(
      value_col = config$value_col,
      refd_col = config$refd_col,
      lag_col = config$lag_col,
      ref_lag = config$ref_lag,
      suffixes = config$value_suffixes,
      smoothed = config$smoothed_raw,
      lagged_term_list = config$lagged_term_list,
      value_type=config$value_type,
      temporal_resol=config$temporal_resol
      ) %>%
    dplyr::filter(report_date < test_end_date) # only consider the test dates
}

# Run DelphiRF for a single test date and return results with timing
run_single_test_date <- function(data, test_start_date, test_end_date, config) {
  ref_lag <- config$ref_lag
  training_window <- config$training_window
  testing_window <- config$testing_window

  pre_start <- Sys.time()
  preprocessed <- prepare_data(data, test_start_date, test_end_date, config)
  pre_end <- Sys.time()

  run_start <- Sys.time()
  results <- DelphiRF(
    preprocessed,
    test_start_date,
    lag_pad = config$lag_pad,
    training_days = training_window,
    model_save_dir = config$model_save_dir,
    indicator = config$indicator,
    signal = config$signal,
    value_type = config$value_type,
    lambda = config$lambda,
    smoothed_target = config$smoothed_target,
    test_lag_groups = config$test_lag_groups,
    temporal_resol = config$temporal_resol,
    train_models = config$train_models,
    make_predictions = config$make_predictions,
  )
  run_end <- Sys.time()

  results$test_date <- test_start_date
  results$time_for_preprocessing <- as.numeric(pre_end - pre_start)
  results$time_for_running <- as.numeric(run_end - run_start)

  return(results)
}

# ------------------------
# MAIN FUNCTION
# ------------------------

main <- function(config) {
  # Load the data first
  df <- get("ma_dph")
  df$geo_value <- "ma"
  
  
  # Get the list of locations
  locs <- unique(df$geo_value)

  # Generate valid test dates
  valid_dates <- seq(config$start_date, config$end_date, by = "days")
  test_dates <- valid_dates[seq(1, length(valid_dates), by = config$testing_window)]
  test_dates <- c(test_dates, config$end_date + 1)

  for (loc in locs) {
    print(loc)
    subdf <- df %>% filter(geo_value == loc)
    if (dim(subdf)[1] < 100) next
    # Run experiments for each test window
    all_results <- lapply(1:(length(test_dates) - 1), function(i) {
      test_start_date <- test_dates[i]
      test_end_date <- test_dates[i+1]
      cat("Running test for date:", format(test_start_date), "\n")
      run_single_test_date(subdf, test_start_date, test_end_date, config)
    })

    # Combine and save results
    combined <- bind_rows(all_results)
    file_name <- paste0(
      config$indicator,
      "_", loc,
      "_", config$temporal_resol,
      "_trainingwindow", config$training_window,
      "_testingwindow", config$testing_window,
      "_reflag", config$ref_lag,
      "_lagpad", config$lag_pad,
      "_DelphiRF.csv"
    )
    write.csv(combined, file = file.path(config$model_save_dir, file_name), row.names = FALSE)
  }

}

# Run everything
main(config)
