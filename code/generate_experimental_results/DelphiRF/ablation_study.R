# Load necessary packages
library(dplyr)
library(stringr)
library(readr)
library(rlang)
library(ggplot2)

library(DelphiRF)

source("DelphiRF_exp.R")

configs <- list(
  list(
    file = "ilinet_config.R",
    data_path = "../../../data/raw/ilicases_weekly_preprocessed.csv",
    geo_value = "nat",
    loader = "load_data"
  ),
  list(
    file = "dengue_config.R",
    data_path = "../../../data/raw/dengue_weekly_preprocessed.csv",
    geo_value = "pr",
    loader = "load_data"
  ),
  list(
    file = "madph_config.R",
    loader = "get",
    object = "ma_dph",
    geo_value = "ma"
  ),
  list(
    file = "quidel_config.R",
    data_path = "../../../data/raw/quidel_allages_state_combined_df_until20230216.csv",
    loader = "load_data"
  ),
  list(
    file = "chng_count_config.R",
    data_path = "../../../data/raw/CHNG_outpatient_state_combined_df_until20230218.csv",
    loader = "load_data"
  ),
  list(
    file = "chng_fraction_config.R",
    data_path = "../../../data/raw/CHNG_outpatient_state_combined_df_until20230218.csv",
    loader = "load_data"
  )
)

for (cfg in configs) {
  source(cfg$file)
  config$model_save_dir <- "../../../data/ablation"
  
  
  if (!is.null(cfg$data_path)) {
    config$data_path <- cfg$data_path
  }
  
  if (cfg$loader == "load_data") {
    df <- load_data(config)
  } else if (cfg$loader == "get") {
    df <- get(cfg$object)
  }
  
  if (!is.null(cfg$geo_value)) {
    df$geo_value <- cfg$geo_value
  }
  
  # ⬇️ here you can operate on df each time
  
  # Get the list of locations
  locs <- unique(df$geo_value)
  
  # Generate valid test dates
  valid_dates <- seq(config$start_date, config$end_date, by = "days")
  test_dates <- valid_dates[seq(1, length(valid_dates), by = config$testing_window)]
  #test_dates <- c(test_dates, config$end_date + 7)
  
  
  if (config$temporal_resol == "daily") {
    lagged_term_list <- c(1, 7)
  } else if (config$temporal_resol == "weekly") {
    lagged_term_list <- c(7, 14)
  }
  dayofweek <- c("Mon", "Weekends")
  covariate_groups <- list( # Define covariate groups
    week_issue = WEEK_ISSUES[1],
    y7dav = Y7DAV,
    value_lags = paste0("log_value_7dav_lag", lagged_term_list),
    delta_lags = paste0("log_delta_value_7dav_lag", lagged_term_list),
    sqrtscale = SQRT_SCALES 
  )
  if (config$temporal_resol == "daily") {
    covariate_groups$daily_ref <- paste0(dayofweek, "_ref")
    covariate_groups$daily_issue <- paste0(dayofweek, "_issue")
  }
  
  # Flatten all covariates into one list for full model
  params_list <- unlist(covariate_groups, use.names = FALSE)
  
  
  for (loc in locs) {
    print(loc)
    subdf <- df %>% filter(geo_value == loc)
    if (dim(subdf)[1] < 100) next
    # Run experiments for each test window
    # Loop: Drop one group at a time
    for (group_name in names(covariate_groups)[5]) {
      group_to_drop <- covariate_groups[[group_name]]
      covariates_used <- setdiff(params_list, group_to_drop)
      
      cat("Dropping group:", group_name, "\n")
      print("Using covariates:")
      print(covariates_used)
      
      all_results <- lapply(1:(length(test_dates) - 1), function(i) {
        test_start_date <- test_dates[i]
        test_end_date <- test_dates[i+1]
        cat("Running test for date:", format(test_start_date), "\n")
        run_single_test_date(subdf, test_start_date, test_end_date, config, covariates_used)
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
        "_ablation_drop_", group_name,
        "_DelphiRF.csv"
      )
      write.csv(combined, file = file.path(config$model_save_dir, file_name), row.names = FALSE)
    }
  }  
}
