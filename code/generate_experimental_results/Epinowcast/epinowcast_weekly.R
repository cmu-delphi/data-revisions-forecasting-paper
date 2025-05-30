# --- Script for generating epinowcast experiment results for weekly data ---

# --- Load Libraries ---
library(epinowcast)
library(DelphiRF)
library(data.table)
library(ggplot2)
library(dplyr)
library(tidyr)
library(evalcast)
library(furrr)

# Load config from external file
source("dengue_config.R")

# --- Load and preprocess raw data ---
load_and_prepare_data <- function(config) {
  df <- read.csv(config$data_path, colClasses = setNames(rep("Date", 2), c(config$refd_col, config$report_date_col)))
  df[[config$lag_col]] <- as.integer(difftime(df[[config$report_date_col]], 
                                              df[[config$refd_col]], units = "days"))
  df$reference_date <- df[[config$refd_col]]
  df$report_date <- df[[config$report_date_col]]
  df$confirm <- round(df[[value_col]])
  return(df)
}

# --- Extract the target values for evaluation ---
get_target_df <- function(df, config) {
  target_df <- df[df[[config$lag_col]] == config$ref_lag, c("reference_date", config$value_col)]
  names(target_df)[2] <- "value_target"
  return(target_df)
}

# --- Perform nowcasting for a single test date ---
run_nowcast <- function(test_date, backfill_df, target_df, config) {
  tryCatch({
    cat("Processing:", as.character(as.Date(test_date)), "\n")
    
    start_time <- proc.time()
    test_df <- backfill_df %>%
      filter(.data$lag <= config$ref_lag) %>%
      filter(.data$reference_date > (test_date - config$training_window - config$ref_lag)) %>%
      filter(.data$report_date >= (test_date - config$training_window)) %>%
      filter(.data$report_date <= test_date) %>%
      filter(.data$report_date >= .data$reference_date) %>%
      filter(.data$confirm > 0) %>%
      select(reference_date, report_date, confirm) %>%
      arrange(report_date, reference_date) %>%
      drop_na()
    
    pobs <- enw_preprocess_data(test_df, max_delay = config$ref_lag, timestep = "week")
    
    nowcast <- epinowcast(
      data = pobs,
      expectation = enw_expectation(~ 0 + (1 | week), data = pobs),
      reference = enw_reference(~1, distribution = "lognormal", data = pobs),
      report = enw_report(~ 1, data = pobs),
      fit = enw_fit_opts(
        save_warmup = FALSE, pp = TRUE,
        chains = 2, threads_per_chain = 2,
        iter_sampling = 1000, iter_warmup = 1000,
        show_messages = FALSE
      ),
      model = enw_model(threads = TRUE)
    )
    
    elapsed <- proc.time() - start_time
    cat("Elapsed time:", elapsed["elapsed"], "\n")
    
    result <- enw_posterior(
      nowcast$fit[[1]],
      variables = "pp_inf_obs",
      probs = TAUS
    )
    max_delay_model <- nrow(result) / max(nowcast$latest[[1]]$.group)
    ord_obs <- build_ord_obs(nowcast$latest[[1]], config$ref_lag, 7, "week")
    # add observations for modelled dates
    obs_model <- subset_obs(ord_obs, max_delay_model, 7, reference_subset = ">")
    result <- as.data.frame(cbind(obs_model, result))
    result$reference_date <- as.Date(result$reference_date)
    result$report_date <- as.Date(result$report_date)
    
    combined <- merge(result, target_df, by = "reference_date")
    pred_cols <- paste0("q", config$tau_list * 100)
    predicted_trans <- as.list(data.frame(t(log(combined[, pred_cols] + 1) - log(combined$value_target + 1))))
    taus_list <- replicate(nrow(combined), config$tau_list, simplify = FALSE)
    
    combined$wis <- mapply(weighted_interval_score, taus_list, predicted_trans, 0)
    combined$test_date <- test_date
    combined$elapsed_time <- elapsed["elapsed"]
    
    return(combined)
  }, error = function(e) {
    message("Error on ", test_date, ": ", e$message)
    return(NULL)
  })
}

# --- Main procedure ---
main <- function(config) {
  
  raw_df <- load_and_prepare_data(config)
  
  # Get the list of locations
  locs <- unique(df$geo_value)
  
  for (loc in locs) {
    print(loc)
    df <- raw_df %>% filter(geo_value == loc)
    
    target_df <- get_target_df(df, config)
    backfill_df <- df %>% select(reference_date, report_date, confirm, lag)
    
    test_dates <- seq(config$start_date, config$end_date, by = config$testing_window)
    results <- purrr::map(test_dates, run_nowcast, 
                          backfill_df = backfill_df, target_df = target_df, 
                          config = config)
    combined_results <- bind_rows(results)
    
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
        "_epinowcast.csv"
      )
    )
    write.csv(combined_results, output_file, row.names = FALSE)
    cat("Results saved to:", output_file, "\n")
  }
}

# --- Run main with default parameters ---
main(config)




