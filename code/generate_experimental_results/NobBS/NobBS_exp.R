# --- Script for generating NobBS experiment results ---

# --- Load Libraries ---
library(NobBS)
library(tidyverse)
library(evalcast)

# Load config from external file
source("madph_config.R")


# --- Load and preprocess raw data ---
load_and_prepare_data <- function(config) {
  df <- read.csv(config$data_path, colClasses = setNames(rep("Date", 2), c(config$refd_col, config$report_date_col)))
  df[[config$lag_col]] <- as.integer(difftime(df[[config$report_date_col]], 
                                              df[[config$refd_col]], units = "days"))
  return(df)
}


# --- Extract the target values for evaluation ---
get_target_df <- function(df, config) {
  target_df <- df[df$lag == config$ref_lag, c(config$refd_col, config$value_col, config$report_date_col, "geo_value")]
  names(target_df)[names(target_df) == config$value_col] <- "value_target"
  names(target_df)[names(target_df) == config$report_date_col] <- "target_date"
  return(target_df)
}

# --- Construct line list for NobBS ---
generate_line_list <- function(df, config) {
  subdf <- df %>%
    select(
      !!sym(config$refd_col),
      !!sym(config$report_date_col),
      value_col = !!sym(config$value_col)
    ) %>%
    mutate(value_col = round(value_col)) %>%
    group_by(!!sym(config$refd_col)) %>%
    mutate(diff_reports = value_col - lag(value_col, default = 0)) %>%
    ungroup()
  
  line_list <- subdf %>%
    filter(diff_reports > 0) %>%
    select(config$refd_col, config$report_date_col, diff_reports) %>%
    uncount(diff_reports) %>%
    as.data.frame()
  
  return(line_list)
}

# --- Run NobBS for a single test date ---
run_nobbs_nowcast <- function(test_date, line_list, target_df, config) {
  cat("Processing:", as.character(test_date), "\n")
  start_time <- proc.time()
  
  if (config$temporal_resol == "daily"){
    units = "1 day"
  } else if (config$temporal_resol == "weekly"){
    units = "1 week"
  }
  
  test_nowcast <- NobBS(
    data = line_list,
    now = as.Date(test_date),
    units = units,
    onset_date = config$refd_col,
    report_date = config$report_date_col,
    max_D = config$max_D,
    moving_window = config$moving_window,
    specs = list(nAdapt = 10000)
  )
  
  elapsed_time <- proc.time() - start_time
  cat("Elapsed time:", elapsed_time["elapsed"], "\n")
  
  test_data$elapsed_time <- elapsed_time["elapsed"]
  test_data <- test_nowcast$estimates
  test_data[[config$report_date_col]] <- as.Date(test_date)
  test_data[[config$refd_col]] <- test_data$onset_date
  
  test_data <- merge(test_data, target_df, by = c(config$refd_col, "geo_value"))
  
  tau_list <- c(0.025, 0.5, 0.975)
  pred_cols <- c("lower", "estimate", "upper")
  predicted_trans <- as.list(data.frame(t(log(test_data[, pred_cols] + 1) - log(test_data$value_target + 1))))
  n_row <- nrow(test_data)
  taus_list <- replicate(n_row, tau_list, simplify = FALSE)
  test_data$wis <- mapply(weighted_interval_score, taus_list, predicted_trans, 0)
  
  return(test_data)
}

# --- Main procedure ---
main <- function(config) {
  df <- get("ma_dph")
  df$geo_value <- "ma"
  
  #df <- load_and_prepare_data(config)
  
  target_df <- get_target_df(df, config)
  line_list <- generate_line_list(df, config)
  
  valid_dates <- seq(config$start_date, config$end_date, by = "day")
  test_dates <- valid_dates[seq(1, length(valid_dates), by = config$testing_window)]
  
  test_data_list <- purrr::map(test_dates, run_nobbs_nowcast,
                               line_list = line_list,
                               target_df = target_df,
                               config = config)
  
  test_combined <- bind_rows(test_data_list)
  
  output_file <- file.path(
    config$export_dir,
    paste0(
      config$indicator,
      "_", loc,
      "_", config$value_type,
      "_", config$temporal_resol,
      "_testingwindow", config$testing_window,
      "_training_window", config$training_window,
      "_reflag", config$ref_lag,
      "_nobBS.csv"
    )
  )
  write.csv(test_combined, output_file, row.names = FALSE)
  cat("Saved:", output_file, "\n")
}

# --- Run the pipeline ---
main(config)