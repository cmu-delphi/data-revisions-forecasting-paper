#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revision forecast run-time comparison
DelphiRF, Epinowcast, NobBS
for daily data only

@author: jingjingtang
"""
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages

import seaborn as sns
import pandas as pd
import numpy as np

import warnings


warnings.filterwarnings("ignore")

from constants import (signals, data_dir, fig_dir, taus,
                        filtered_states, map_list)
from _utils_ import *

# DelphiRF results
dfs = {}
dfs["COVID-19 cases"] = read_experimental_results(madph_config, "DelphiRF")
dfs["CHNG Outpatient Count"] = read_experimental_results(chng_count_config, "DelphiRF")
dfs["dengue"] = read_experimental_results(dengue_config, "DelphiRF")
dfs["ilicases"] = read_experimental_results(ilicases_config, "DelphiRF")

# nobBS results
nobBS_dfs = {}
nobBS_dfs["COVID-19 cases"] = read_experimental_results(madph_config, "NobBS")
nobBS_dfs["CHNG Outpatient Count"] = read_experimental_results(chng_count_config, "NobBS")
nobBS_dfs["dengue"] = read_experimental_results(dengue_config, "NobBS")
nobBS_dfs["ilicases"] = read_experimental_results(ilicases_config, "NobBS")

# epinowcast results
epinowcast_dfs = {}
epinowcast_dfs["COVID-19 cases"] = read_experimental_results(madph_config, "Epinowcast")
epinowcast_dfs["CHNG Outpatient Count"] = read_experimental_results(chng_count_config, "Epinowcast")
epinowcast_dfs["dengue"] = read_experimental_results(dengue_config, "Epinowcast")
epinowcast_dfs["ilicases"] = read_experimental_results(ilicases_config, "Epinowcast")




####################################
#### Run-time comparison
#### The run-time is calculated for both
#### data preprocessing and model training/testing
#### Performance comparison with other methods
#### performance comparison for difference training window
####################################

rt_summary = pd.DataFrame(columns=["method", "data", "training_window", "mean", "sem"])
idx = 0
for signal in ["COVID-19 cases", "CHNG Outpatient Count"]:
    for tw in [180, 365]:
        epinowcast_df = epinowcast_dfs[signal].loc[
            (epinowcast_dfs[signal]["test_date"] == epinowcast_dfs[signal]["report_date"])
            & (epinowcast_dfs[signal]["tw"] == tw),
            ["report_date", "geo_value", "elapsed_time"]].drop_duplicates()
        mean_elapsed_time = epinowcast_df["elapsed_time"].mean()
        std_elapsed_time = epinowcast_df["elapsed_time"].std(ddof=1)
        sem_elapsed_time = std_elapsed_time / np.sqrt(len(epinowcast_df))
        rt_summary.loc[idx] = ["epinowcast", signal, tw, mean_elapsed_time, sem_elapsed_time]
        idx += 1
    
        nobBS_df = nobBS_dfs[signal].loc[
            nobBS_dfs[signal]["tw"] == tw,
            ["issue_date", "geo_value", "elapsed_time"]].drop_duplicates()
        mean_elapsed_time = nobBS_df["elapsed_time"].mean()
        std_elapsed_time = nobBS_df["elapsed_time"].std(ddof=1)
        sem_elapsed_time = std_elapsed_time / np.sqrt(len(nobBS_df))
        rt_summary.loc[idx] = ["nobBS", signal, tw, mean_elapsed_time, sem_elapsed_time]
        idx += 1
    
        delphi_df = dfs[signal].loc[
            (dfs[signal]["tw"] == tw),
            ['report_date', 'time_for_running', 'time_for_preprocessing']
        ].drop_duplicates()
        delphi_df["elapsed_time"] = delphi_df["time_for_running"] + delphi_df["time_for_preprocessing"]
        mean_elapsed_time = delphi_df["elapsed_time"].mean()
        std_elapsed_time = delphi_df["elapsed_time"].std(ddof=1)
        sem_elapsed_time = std_elapsed_time / np.sqrt(len(delphi_df))
        rt_summary.loc[idx] = ["Delphi-Training", signal, tw, mean_elapsed_time, sem_elapsed_time]
        idx += 1
rt_summary.head(20)

