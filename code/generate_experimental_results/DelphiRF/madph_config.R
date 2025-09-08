# Load shared constants used across models (GAMMA, LP_SOLVER, TAUS, TEST_LAG_GROUPS, etc.)
source("constants.R")

# ----------------------------------------------------------------
# Configuration for MA-DPH dataset forecasting using DelphiRF
# ----------------------------------------------------------------
config <- list(

  # ---- Dataset metadata ----
  geo_level        = "state",           # Aggregation level
  indicator        = "madph",           # Source indicator name
  signal           = "confirmed_cases", # Signal type
  signal_suffix    = "",                # Optional suffix for alternate signal versions
  value_type       = "count",           # Type of observed values
  temporal_resol   = "daily",           # Temporal resolution of data

  # ---- Temporal configuration ----
  start_date       = as.Date("2021-07-01"),  # Start date for experiments
  end_date         = as.Date("2022-06-24"),  # End date for experiments
  training_window  = 180,                    # Number of training days
  testing_window   = 7,                      # Interval between test windows
  ref_lag          = 14,                     # Maximum lag used for training targets
  lag_pad          = 0,                      # Padding before lag group models

  # ---- Preprocessing column mappings ----
  value_col        = "confirmed_case_7d_avg", # Target value column in the raw data
  refd_col         = "test_date",             # Reference (event) date column
  lag_col          = "lag",                   # Lag column (days since reference date)
  report_date_col  = "issue_date",            # Date when the value was reported

  # ---- Modeling parameters ----
  lambda           = LAMBDA,          # Regularization for main effect
  gamma            = GAMMA,         # Regularization for interaction term (from constants)
  lp_solver        = LP_SOLVER,     # LP solver selection (e.g., "gurobi", "glpk")
  taus             = TAUS,          # Quantile levels to estimate (from constants)
  test_lag_groups  = NULL,          # Lag group definitions for testing

  # ---- Modeling control flags ----
  smoothed_raw     = TRUE,         # Whether the raw data is smoothed
  smoothed_target  = TRUE,          # Whether to use smoothed as the target
  train_models     = TRUE,          # Whether to train models (FALSE = skip training)
  make_predictions = TRUE,          # Whether to make predictions after training

  # ---- Model components (optional override) ----
  value_suffixes  = c(""),          # optional character vector containing suffixes for value columns
  lagged_term_list = NULL,          # Optional custom lagged predictors
  params_list      = NULL,          # Optional parameter overrides

  # ---- Output directory ----
  model_save_dir   = "./receiving"  # Directory where models and results will be saved
)
