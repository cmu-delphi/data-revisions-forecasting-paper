#' Constants
#'

TAUS <- c(0.01, 0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975, 0.99)

REF_LAG <- 60

LAG_PAD <- 1

TRAINING_DAYS <- 365

LAMBDA <- 0.1

GAMMA <- 0.1

LP_SOLVER <- "gurobi" # LP solver to use in quantile_lasso(); "gurobi" or "glpk"

YITL <- "log_value_raw"

SLOPE <- "log_7dav_slope"

Y7DAV <- "log_value_7dav"

RESPONSE <- "log_value_target"

LOG_LAG <- "inv_log_lag"

# Dates
WEEKDAYS_ABBR <- c("Mon", "Tue", "Wed", "Thurs", "Fri", "Sat", "Sun") # wd

WEEK_ISSUES <- c("W1_issue", "W2_issue", "W3_issue") # wm

SQRT_SCALES <- c("sqrty0", "sqrty1", "sqrty2")

TODAY <- Sys.Date()

TEST_LAG_GROUPS_DAILY <- c(as.character(0:14), c("15-21", "22-35", "36-49", "50-59"))

TEST_LAG_GROUPS_WEEKLY <- c(as.character(seq(0, 70, 7)))


# For Delphi Signals
INDICATORS_AND_SIGNALS <- tibble::tribble(
  ~indicator, ~signal, ~name_suffix, ~sub_dir,
  "changehc", "covid", "", "chng",
  "changehc", "flu", "", "chng",
  "claims_hosp", "", "", "claims_hosp",
  # "dv",,,
  "quidel", "covidtest", c("total", "age_0_4", "age_5_17", "age_18_49", "age_50_64", "age_65plus", "age_0_17"), "quidel_covidtest"
)
