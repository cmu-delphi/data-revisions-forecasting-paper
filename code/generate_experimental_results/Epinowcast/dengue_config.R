# Load shared constants used across models (GAMMA, LP_SOLVER, TAUS, TEST_LAG_GROUPS, etc.)

# ----------------------------------------------------------------
# Configuration for dengue cases dataset count forecasting using epinowcast
# ----------------------------------------------------------------
config <- list(
  # ---- Dataset metadata ----
  data_path        = "",                # file dir
  indicator        = "dengue",          # Source indicator name
  value_type       = "count",           # Type of observed values
  temporal_resol   = "weekly",          # Temporal resolution of data

  # ---- Temporal configuration ----
  start_date       = as.Date("1991-12-23"),  # Start date for experiments
  end_date         = as.Date("2010-11-29"),  # End date for experiments
  training_window  = 104 * 7,                # Number of training days
  testing_window   = 7,                      # Interval between test windows
  ref_lag          = 10 * 7,                 # Maximum lag used for training targets
  tau_list         = c(0.01, 0.025, 0.1,     # keep it the same as experiments for DelphiRF
                       0.25, 0.5, 0.75, 
                       0.9, 0.975, 0.99),

  # ---- Preprocessing column mappings ----
  value_col        = "value_7dav",            # Target value column in the raw data
  refd_col         = "time_value",            # Reference (event) date column
  lag_col          = "lag",                   # Lag column (days since reference date)
  report_date_col  = "issue_date",            # Date when the value was reported

  # ---- Output directory ----
  export_dir   = "./receiving"  # Directory where models and results will be saved
)
