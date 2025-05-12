#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revision forecast performance comparison
DelphiRF, Epinowcast, NobBS

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
#### WIS comparison over lag
#### Performance comparison with other methods
####################################
titles = {
    "COVID-19 cases": "COVID-19 cases\n(Daily, State, MA only)", 
    "CHNG Outpatient Count": "Insurance claims\n(Daily, State, All states)", 
    "dengue": "Dengue fever cases\n(Weekly, State, PR only)",
    "ilicases": "ILI cases\n(Weekly, National)"
    }

plt.style.use('default')
fig = plt.figure(figsize=(16, 12))
for idx, signal in enumerate(["COVID-19 cases", "CHNG Outpatient Count", "dengue", "ilicases"]):
    
    nobBS_df = nobBS_dfs[signal].loc[
        (nobBS_dfs[signal]["tw"] == nobBS_dfs[signal]["tw"].min())].groupby(["lag"]).agg(
        mean=('wis', 'mean'),
        sem=('wis', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
        quantile10=('wis', lambda x: np.quantile(x.dropna(), 0.1)),         # Median (50th percentile)
        quantile90=('wis', lambda x: np.quantile(x.dropna(), 0.9))
        ).reset_index()
    
    epinowcast_df = epinowcast_dfs[signal].loc[
        (epinowcast_dfs[signal]["tw"] == epinowcast_dfs[signal]["tw"].min())
        # & (epinowcast_dfs[signal]["test_date"] == epinowcast_dfs[signal]["report_date"])
        ].groupby(["lag"]).agg(
        mean=('wis', 'mean'),
        sem=('wis', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
        quantile10=('wis', lambda x: np.quantile(x, 0.1)),         # Median (50th percentile)
        quantile90=('wis', lambda x: np.quantile(x, 0.9))
        ).reset_index()
    
    delphi_df = dfs[signal].loc[ # only kept the result on the training dates 
        (dfs[signal]["tw"] == dfs[signal]["tw"].min())
        & (dfs[signal]["report_date"].isin(nobBS_dfs[signal]["issue_date"].unique()))
        ].groupby(["lag"]).agg(
        mean=('wis', 'mean'),
        sem=('wis', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
        quantile10=('wis', lambda x: np.nanquantile(x, 0.1)),         # Median (50th percentile)
        quantile90=('wis', lambda x: np.nanquantile(x, 0.9))
        ).reset_index()
       
    ax1 = plt.subplot(2, 2, idx+1)
    ax1.plot(delphi_df["lag"], delphi_df["mean"], label="Delphi-RF", linewidth=3.0, color="tab:orange")
    ax1.fill_between(delphi_df["lag"], 
                      delphi_df["quantile10"], 
                      delphi_df["quantile90"], alpha=0.1, color="tab:orange")
    ax1.plot(nobBS_df["lag"], nobBS_df["mean"], label="NobBS", linewidth=3.0, color="tab:blue")
    ax1.fill_between(nobBS_df["lag"], 
                     nobBS_df["quantile10"], 
                     nobBS_df["quantile90"], alpha=0.1, color="tab:blue")
    ax1.plot(epinowcast_df["lag"], epinowcast_df["mean"], label="Epinowcast", linewidth=3.0, color="tab:purple")
    ax1.fill_between(epinowcast_df["lag"], 
                     epinowcast_df["quantile10"], 
                     epinowcast_df["quantile90"], alpha=0.1, color="tab:purple")   
    ax1.grid(True)
        
    # ax1.set_ylabel("WIS", fontsize=30)
    if idx in [0, 2]:
        ax1.set_ylabel("WIS", fontsize=30)
    if idx < 2:
        ax1.set_xlim((0, 14))
        ax1.set_ylim((0, 0.62))
        yticks = [0, 5, 10, 20, 50]
        ax1.set_xticks(np.arange(0, 15, 1))
        ax1.set_yticks(np.linspace(0, 0.6, 7))
        ax1.set_xlabel("Lag (Days)", fontsize=30)
    else:
        ax1.set_xlim((0, 9))
        ax1.set_ylim((0, 0.19))
        ax1.set_xticks(np.arange(0, 57, 7), np.arange(0, 9))
        yticks = [0, 5, 10, 20]
        ax1.set_xlabel("Lag (Weeks)", fontsize=30)
   
    ax1.tick_params(axis='both', labelsize=20)
    
    ax2 = ax1.twinx()
    ax2.plot([],[])
    ax2.set_yticks(list(map(re_to_wis, yticks)))
    ax2.set_yticklabels([str(x)+"%" for x in yticks])
    ax2.set_ylim((ax1.get_ylim()[0], ax1.get_ylim()[1]))
    ax2.tick_params(axis='both', labelsize=20)  
    ax2.grid(True)    
    ax1.set_title(titles[signal], fontsize=35)
    if idx in [1, 3]:
        ax2.set_ylabel('Absolute Relative Error', fontsize=30, rotation=270, labelpad=30)
labels_handles = {
  label: handle for ax in fig.axes for handle, label in zip(*ax.get_legend_handles_labels())
}

fig.legend(
  labels_handles.values(),
  labels_handles.keys(),
  loc = "upper center",
  bbox_to_anchor = (0.5, 0),
  bbox_transform = plt.gcf().transFigure,
  ncol = 3,
  fontsize=30
)
plt.tight_layout()
plt.savefig(fig_dir + "experiment_count_result_evl_general_for_comparison.pdf", bbox_inches = 'tight')


####################################
#### Run-time comparison
#### The run-time is calculated for both
#### data preprocessing and model training
#### Performance comparison with other methods
####################################

rt_summary = pd.DataFrame(columns=["method", "data", "mean", "sem"])
idx = 0
for signal in ["COVID-19 cases", "CHNG Outpatient Count", "dengue", "ilicases"]:
    epinowcast_df = epinowcast_dfs[signal].loc[
        (epinowcast_dfs[signal]["test_date"] == epinowcast_dfs[signal]["report_date"])
        & (epinowcast_dfs[signal]["tw"] == epinowcast_dfs[signal]["tw"].min()),
        ["report_date", "geo_value", "elapsed_time"]].drop_duplicates()
    mean_elapsed_time = epinowcast_df["elapsed_time"].mean()
    std_elapsed_time = epinowcast_df["elapsed_time"].std(ddof=1)
    sem_elapsed_time = std_elapsed_time / np.sqrt(len(epinowcast_df))
    rt_summary.loc[idx] = ["epinowcast", signal, mean_elapsed_time, sem_elapsed_time]
    idx += 1

    nobBS_df = nobBS_dfs[signal].loc[
        nobBS_dfs[signal]["tw"] == nobBS_dfs[signal]["tw"].min(),
        ["issue_date", "geo_value", "elapsed_time"]].drop_duplicates()
    mean_elapsed_time = nobBS_df["elapsed_time"].mean()
    std_elapsed_time = nobBS_df["elapsed_time"].std(ddof=1)
    sem_elapsed_time = std_elapsed_time / np.sqrt(len(nobBS_df))
    rt_summary.loc[idx] = ["nobBS", signal, mean_elapsed_time, sem_elapsed_time]
    idx += 1

    delphi_df = dfs[signal].loc[
        (dfs[signal]["tw"] == dfs[signal]["tw"].min()),
        ['report_date', 'time_for_running', 'time_for_preprocessing']
    ].drop_duplicates()
    delphi_df["elapsed_time"] = delphi_df["time_for_running"] + delphi_df["time_for_preprocessing"]
    mean_elapsed_time = delphi_df["elapsed_time"].mean()
    std_elapsed_time = delphi_df["elapsed_time"].std(ddof=1)
    sem_elapsed_time = std_elapsed_time / np.sqrt(len(delphi_df))

    rt_summary.loc[idx] = ["Delphi-Training", signal, mean_elapsed_time, sem_elapsed_time]
    idx += 1
rt_summary

