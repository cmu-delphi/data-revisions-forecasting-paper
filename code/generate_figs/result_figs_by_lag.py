#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregated evl results over lags

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

from constants import (signals, data_dir, fig_dir, taus,
                        filtered_states)
from _utils_ import (read_chng_outpatient_result, read_ma_dph_result, read_quidel_result,
                      read_chng_outpatient_count_result, re_to_wis)

### Read results
dfs = {}
dfs["COVID-19 cases in MA"] = read_experimental_results(madph_config, "DelphiRF")
dfs["CHNG Outpatient Count"] = read_experimental_results(chng_count_config, "DelphiRF")
dfs["Insurance claims"] = read_experimental_results(chng_fraction_config, "DelphiRF")
dfs["Antigen tests"] = read_experimental_results(quidel_config, "DelphiRF")


####################################
### Result evaluation in general
### Three datasets share label? 
####################################
yticks = [0, 20, 100, 500, 1000, 1500]
plt.style.use('default')
fig = plt.figure(figsize=(15, 6))
for idx, signal in enumerate(["COVID-19 cases in MA", "CHNG Outpatient Count"]):
    if signal == "COVID-19 cases in MA":
        test_w = 7
    else:
        test_w = 30

    delphi_result = dfs[signal].groupby(["lag", "tw"]).agg(
        mean=('wis', 'mean'),
        sem=('wis', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
        quantile10=('wis', lambda x: np.quantile(x, 0.1)),         # Median (50th percentile)
        quantile90=('wis', lambda x: np.quantile(x, 0.9))
        ).reset_index()
    
    baseline_result = dfs[signal].groupby(["lag", "tw"]).agg(
        mean=('mae', 'mean'),
        sem=('mae', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
        quantile10=('mae', lambda x: np.quantile(x, 0.1)),         # Median (50th percentile)
        quantile90=('mae', lambda x: np.quantile(x, 0.9))
        ).reset_index()
       
    ax1 = plt.subplot(1, 2, idx+1)
    baseline_subdf = baseline_result.loc[baseline_result["tw"] == 180].sort_values("lag")
    ax1.plot(baseline_subdf["lag"], baseline_subdf["mean"], label="Baseline", linewidth=3.0,
             color = "tab:gray")    
    ax1.fill_between(baseline_subdf["lag"], 
                     baseline_subdf["quantile10"], 
                     baseline_subdf["quantile90"], alpha=0.1, color="tab:gray")
    for tw in [180, 365]:
        if tw == 180:
            color = "tab:orange"
        else:
            color = "tab:green"
        delphi_subdf = delphi_result.loc[delphi_result["tw"] == tw].sort_values("lag")
        ax1.plot(delphi_subdf["lag"], 
                 delphi_subdf["mean"], label="Forecasted (Training window: %d days)"%(tw),
                 linewidth=3.0, color=color)
        ax1.fill_between(delphi_subdf["lag"], 
                         delphi_subdf["quantile10"], 
                         delphi_subdf["quantile90"], alpha=0.1, color=color)

    ax1.grid(True)
    ax1.set_xlabel("Lag (Days)", fontsize=30)
    if idx == 0:
        ax1.set_ylabel("WIS", fontsize=30)
    ax1.set_xlim((0, 14))
    ax1.set_ylim((0, 3))
    ax1.set_xticks(np.arange(0, 15, 1))
    ax1.tick_params(axis='both', labelsize=20)
    
    ax2 = ax1.twinx()
    ax2.plot([],[])
    ax2.set_yticks(list(map(re_to_wis, yticks)))
    ax2.set_yticklabels([str(x)+"%" for x in yticks])
    ax2.set_ylim((ax1.get_ylim()[0], ax1.get_ylim()[1]))
    ax2.tick_params(axis='both', labelsize=18)  
    ax2.grid(True)
    # ax1.legend(bbox_to_anchor= (1.1, -0.3), fontsize=20)
    if signal == "CHNG Outpatient Count":
        ax1.set_title("Insurance claims\n(Model retrained every %d days)"%test_w, fontsize=30, pad=13)
    else:
        ax1.set_title(signal + "\n(Model retrained every %d days)"%test_w, fontsize=30, pad=13)
    if idx == 1:
        ax2.set_ylabel('Absolute Relative Error', fontsize=30, rotation=270, labelpad=30)
labels_handles = {
  label: handle for ax in fig.axes for handle, label in zip(*ax.get_legend_handles_labels())
}

labels = ['Forecasted (Training window: 180 days)', 'Forecasted (Training window: 365 days)', 'Baseline']
handles = [labels_handles[x] for x in labels]
fig.legend(
    handles,
    labels,   
    # labels_handles.values(),
    # labels_handles.keys(),
  loc = "upper center",
  bbox_to_anchor = (0.5, 0),
  bbox_transform = plt.gcf().transFigure,
  ncol = 2,
  fontsize=24
)
plt.tight_layout()
plt.suptitle("Count Forecasting", fontsize=40, y=1.07)
plt.savefig(fig_dir + "experiment_count_result_evl_general.png", bbox_inches = 'tight')



###################
###### Zoom in
###################
colors = {
    180: "tab:orange",
    365: "tab:green"}
for idx, signal in enumerate(["COVID-19 cases", "CHNG Outpatient Count"]):
    
    delphi_result = dfs[signal].groupby(["lag", "tw"]).agg(
        mean=('wis', 'mean'),
        sem=('wis', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
        quantile10=('wis', lambda x: np.quantile(x, 0.1)),         # Median (50th percentile)
        quantile90=('wis', lambda x: np.quantile(x, 0.9))
        ).reset_index()
       
    plt.figure()
    for tw in [180, 365]:
        delphi_subdf = delphi_result.loc[delphi_result["tw"] == tw].sort_values("lag")
        plt.plot(delphi_subdf["lag"], 
                 delphi_subdf["mean"], label="Forecasted(tw=%d)"%tw,linewidth=3.0, color = colors[tw])
        plt.fill_between(delphi_subdf["lag"], 
                         delphi_subdf["quantile10"], 
                         delphi_subdf["quantile90"], alpha=0.1,
                         color = colors[tw])
    plt.grid(True)
    plt.xlim((0, 7))
    # plt.set_ylim((0, 3))
    plt.xticks(np.arange(0, 8, 1), fontsize=40)
    plt.yticks(fontsize=40)

####################################
#### For fraction
####################################
test_w = 30
yticks = {
    "Insurance claims": [0, 10, 20, 50, 100],
    "Antigen tests": [0, 5, 10, 20, 30]}
ylims = {
    "Insurance claims": (-0.02, 0.6),
    "Antigen tests": (-0.01, 0.2)}
plt.style.use('default')
fig = plt.figure(figsize=(15, 6))
for idx, signal in enumerate(["Insurance claims", "Antigen tests"]):
    delphi_result = dfs[signal].groupby(["lag", "tw"]).agg(
        mean=('wis', 'mean'),
        sem=('wis', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
        quantile10=('wis', lambda x: np.quantile(x, 0.1)),         # Median (50th percentile)
        quantile90=('wis', lambda x: np.quantile(x, 0.9))
        ).reset_index()
    
    baseline_result = dfs[signal].groupby(["lag", "tw"]).agg(
        mean=('mae', 'mean'),
        sem=('mae', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
        quantile10=('mae', lambda x: np.quantile(x, 0.1)),         # Median (50th percentile)
        quantile90=('mae', lambda x: np.quantile(x, 0.9))
        ).reset_index()
    baseline_subdf = baseline_result.loc[baseline_result["tw"] == 180].sort_values("lag")
    
    ax1 = plt.subplot(1, 2, idx+1)
    baseline_subdf = baseline_result.loc[baseline_result["tw"] == 180].sort_values("lag")
    ax1.plot(baseline_subdf["lag"], baseline_subdf["mean"], label="Baseline", linewidth=3.0, 
             color="tab:gray")    
    ax1.fill_between(baseline_subdf["lag"], 
                     baseline_subdf["quantile10"], 
                     baseline_subdf["quantile90"], alpha=0.1, color="tab:gray")
    for tw in [180, 365]:
        if tw==180:
            color= "tab:orange"
        else:
            color="tab:green"
        delphi_subdf = delphi_result.loc[delphi_result["tw"] == tw].sort_values("lag")
        ax1.plot(delphi_subdf["lag"], 
                 delphi_subdf["mean"], label="Forecasted (Training window: %d days)"%(tw),
                 linewidth=3.0, color=color)
        ax1.fill_between(delphi_subdf["lag"], 
                         delphi_subdf["quantile10"], 
                         delphi_subdf["quantile90"], alpha=0.1, color = color)
    
    ax1.grid(True)
    ax1.set_xlabel("Lag (Days)", fontsize=30)
    if idx == 0:
        ax1.set_ylabel("WIS", fontsize=30)
    ax1.set_xlim((0, 14))

    ax1.set_xticks(np.arange(0, 15, 1))
    ax1.tick_params(axis='both', labelsize=20)
    ax1.set_ylim(ylims[signal])
    
    ax2 = ax1.twinx()
    ax2.plot([],[])
    # ax2.set_yticks(ax1.get_yticks())
    # ax2.set_yticklabels(list(map(wis_to_re, ax1.get_yticks())))
    ax2.set_yticks(list(map(re_to_wis, yticks[signal])))
    ax2.set_yticklabels([str(x)+"%" for x in yticks[signal]])
    ax2.set_ylim((ax1.get_ylim()[0], ax1.get_ylim()[1]))
    ax2.tick_params(axis='both', labelsize=20)  
    # ax1.legend(bbox_to_anchor=(0.8, 0.85), fontsize=25)
    ax1.set_title(signal + "\n(Model retrained every %d days)"%test_w, fontsize=30, pad=13)
    plt.grid(True)
    if idx == 1:
        ax2.set_ylabel('Absolute Relative Error', fontsize=30, rotation=270, labelpad=30)
labels_handles = {
  label: handle for ax in fig.axes for handle, label in zip(*ax.get_legend_handles_labels())
}
labels = ['Forecasted (Training window: 180 days)', 'Forecasted (Training window: 365 days)', 'Baseline']
handles = [labels_handles[x] for x in labels]
fig.legend(
    handles,
    labels,   
   # labels_handles.values(),
   # labels_handles.keys(),
  loc = "upper center",
  bbox_to_anchor = (0.5, 0),
  bbox_transform = plt.gcf().transFigure,
  ncol = 2,
  fontsize=24
)
plt.grid(True)
plt.tight_layout()
plt.suptitle("Fraction Forecasting", fontsize=40, y=1.07)
plt.savefig(fig_dir + "experiment_frc_result_evl_general.pdf", bbox_inches = 'tight')

