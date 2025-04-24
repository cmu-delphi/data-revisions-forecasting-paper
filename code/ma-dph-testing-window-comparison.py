#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 12:06:04 2025

@author: jingjingtang
"""

from datetime import datetime, timedelta
import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages

import seaborn as sns
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

from .constants import (signals, data_dir, fig_dir, taus,
                        filtered_states)
from ._utils_ import (read_chng_outpatient_result, read_ma_dph_result, read_quidel_result,
                      read_chng_outpatient_count_result, re_to_wis, read_proj)

### Read results
dfs = {}
dfs["COVID-19 cases"] = {}
def read_ma_dph_result(folder):
    pdList = []
    plt.figure()
    for tw in [180, 365]:
        # df = read_proj("~/Downloads/%s/ma_daily_tw%d.csv"%(folder, tw))
        df = read_proj(data_dir + "results/%s/ma_daily_tw%d.csv"%(folder, tw))
        df = df.loc[(df["time_value"] <= datetime(2022, 3, 1))
                    &(df["time_value"] >= datetime(2021, 9, 1))
                    & (~df["geo_value"].isin(set(filtered_states)))].sort_values(["issue_date", "time_value"])
        df["mae"] = abs(df["log_value_7dav"]-df["log_value_target"])
        df["tw"] = tw
        
        subdf = df.loc[df["lag"] == 1]
        plt.plot(subdf["time_value"], subdf["wis"], label="tw=%d"%tw)
        pdList.append(df)    
    plt.legend()        
    return pd.concat(pdList)

for tw in [7, 14, 30]:
    dfs["COVID-19 cases"][tw] = read_ma_dph_result("ma-dph-testwindow%d"%tw)
dfs["COVID-19 cases"][1] = read_ma_dph_result("ma-dph")


####################################
### Result evaluation in general
### Three datasets share label? 
####################################
yticks = [0, 20, 50, 100, 500, 1000, 1500]
plt.style.use('default')
fig = plt.figure(figsize=(10, 6))
for idx, signal in enumerate(["COVID-19 cases"]):
    baseline_result = dfs[signal][1].groupby(["lag", "tw"]).agg(
        mean=('mae', 'mean'),
        sem=('mae', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
        quantile10=('mae', lambda x: np.quantile(x, 0.1)),         # Median (50th percentile)
        quantile90=('mae', lambda x: np.quantile(x, 0.9))
        ).reset_index()
    
    ax1 = plt.subplot(1, 1, idx+1)
    baseline_subdf = baseline_result.loc[baseline_result["tw"] == 180].sort_values("lag")
    ax1.plot(baseline_subdf["lag"], baseline_subdf["mean"], label="Baseline", linewidth=3.0, color="tab:gray")    
    ax1.fill_between(baseline_subdf["lag"], 
                      baseline_subdf["quantile10"], 
                      baseline_subdf["quantile90"], alpha=0.1, color="tab:gray")
    colors = {
        1: "tab:orange",
        7: "tab:red",
        14: "tab:green",
        30: "tab:blue"}
    for testing_w in [1, 7, 14, 30]:
        delphi_result = dfs[signal][testing_w].groupby(["lag", "tw"]).agg(
            mean=('wis', 'mean'),
            sem=('wis', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
            quantile10=('wis', lambda x: np.quantile(x, 0.1)),         # Median (50th percentile)
            quantile90=('wis', lambda x: np.quantile(x, 0.9))
            ).reset_index()
        print(delphi_result)
       
    
        for training_w in [180, 365]:
            if training_w == 180:
                linestyle="--"
            else:
                linestyle="-"
            delphi_subdf = delphi_result.loc[delphi_result["tw"] == training_w].sort_values("lag")
            ax1.plot(delphi_subdf["lag"], 
                     delphi_subdf["mean"], label="Forecasted (%d-day testing, %d-day training)"%(testing_w, training_w),linewidth=3.0,
                     linestyle=linestyle, color=colors[testing_w])
            ax1.fill_between(delphi_subdf["lag"], 
                             delphi_subdf["quantile10"], 
                             delphi_subdf["quantile90"], alpha=0.1)
                

    ax1.grid(True)
    ax1.set_xlabel("Lag (Days)", fontsize=30)
    if idx == 0:
        ax1.set_ylabel("WIS", fontsize=30)
    ax1.set_xlim((0, 14))
    ax1.set_ylim((0, 3))
    ax1.set_xticks(np.arange(0, 15, 1))
    # ax1.set_yticks(np.append(ax1.get_yticks(), [0.1]))
    ax1.tick_params(axis='both', labelsize=20)
    
    ax2 = ax1.twinx()
    ax2.plot([],[])
    # ax2.set_yticks(ax1.get_yticks())
    # ax2.set_yticklabels(list(map(wis_to_re, ax1.get_yticks())))
    ax2.set_yticks(list(map(re_to_wis, yticks)))
    ax2.set_yticklabels([str(x)+"%" for x in yticks])
    ax2.set_ylim((ax1.get_ylim()[0], ax1.get_ylim()[1]))
    ax2.tick_params(axis='both', labelsize=18)  
    ax2.grid(True)
    # ax1.legend(bbox_to_anchor=(0.8, 0.85), fontsize=25)
    if signal == "CHNG Outpatient Count":
        ax1.set_title("Insurance claims", fontsize=35)
    else:
        ax1.set_title(signal, fontsize=35)
    if idx == 1:
        ax2.set_ylabel('Absolute Relative Error', fontsize=30, rotation=270, labelpad=30)
labels_handles = {
  label: handle for ax in fig.axes for handle, label in zip(*ax.get_legend_handles_labels())
}

fig.legend(
  labels_handles.values(),
  labels_handles.keys(),
  loc = "upper center",
  bbox_to_anchor = (1.45, 0.95),
  bbox_transform = plt.gcf().transFigure,
  ncol = 1,
  fontsize=24
)
plt.tight_layout()
plt.savefig(fig_dir + "experiment_count_result_%s_hypertuning.pdf"%"_".join(signal.split(" ")), bbox_inches = 'tight')





