#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Count projection example

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
from ._utils_ import (read_chng_outpatient_result, read_ma_dph_result, 
                      read_quidel_result, read_chng_outpatient_count_result)
dfs = {}
dfs["Insurance claims"] = read_chng_outpatient_result()
dfs["COVID-19 cases"] = read_ma_dph_result()
dfs["Antigen tests"] = read_quidel_result()
dfs["CHNG Outpatient Count"] = read_chng_outpatient_count_result()



####################################
### COUNT EXAMPLES
####################################

# How to choose the example? 
# Largest population state and lowest population state
# Lowever epidemic level and higher


df = dfs["CHNG Outpatient Count"].loc[dfs["CHNG Outpatient Count"]["tw"] == 180]

states = ["ca", "wy", "ma"]
statenames = ["California", "Wyoming", "Massachusetts"]
lags = [3, 7, 14]
fig = plt.figure(figsize=(15, 12))
for i in range(len(states)):
    state = states[i]
    for j in range(len(lags)):
        lag = lags[j]
        ax = plt.subplot(3, 3, i*len(lags)+j+1)
        subdf = df.loc[(df["geo_value"] == state)
                        & (df["lag"] == lag)
                        & (df["time_value"] <= datetime(2022, 3, 1))
                        & (df["time_value"] >= datetime(2021, 6, 1))].sort_values("time_value")
        subdf["value_7dav"] = np.exp(subdf["log_value_7dav"]-1)
        subdf["value_target"] = np.exp(subdf["log_value_target"]-1)
        
        plt.plot(subdf["time_value"], subdf["value_7dav"], color = "lightblue", label="Provisional")
        plt.plot(subdf["time_value"], subdf["value_target"], color = "tab:blue", label="Target (Lag=60)")
        plt.plot(subdf["time_value"], subdf["predicted_tau0.5"], color="tab:red", 
                 label="Projected", alpha=0.8)
        plt.fill_between(subdf["time_value"], 
                         subdf["predicted_tau0.25"],
                         subdf["predicted_tau0.75"],
                         color= "tab:red", alpha=0.3)
        # plt.yticks([0, 1, 2, 3, 4], fontsize=15)
        if i == 0:
            plt.title("Lag=%d"%lag, fontsize=35)
        if j == 0:
            plt.ylabel("#COVID-19\nInsurance\nClaims", fontsize=25)
        if j == len(lags)-1:
            plt.text(1.02, 0.5, statenames[i], va='center', rotation=-90, transform=ax.transAxes, fontsize=25)

        plt.xticks(fontsize=18,rotation=45, ha = "right")
        plt.yticks(fontsize=15)
        plt.grid()
        # plt.legend(fontsize=20)
        if i == len(lags)-1:
            plt.xlabel("Reference Date", fontsize=25)
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
# plt.suptitle("Insurance claims", fontsize=40, y=1)
plt.tight_layout()
plt.savefig(fig_dir+"eg_chng_outpatient_count_combined.png", bbox_inches="tight")



# plt.style.use('default')
# fig = plt.figure(figsize=(21, 6))
# for idx, signal in enumerate(states):
#     subdf = df.loc[(df["geo_value"] == states[idx])
#                     & (df["time_value"] <= datetime(2022, 3, 1))
#                     & (df["time_value"] >= datetime(2021, 6, 1))]
#     mean_df = subdf.groupby(["lag"]).mean().reset_index()
#     q5_df = subdf.groupby(["lag"]).quantile(0.05).reset_index()
#     q95_df = subdf.groupby(["lag"]).quantile(0.95).reset_index()
       
#     ax1 = plt.subplot(1, 3, idx+1)
#     ax1.plot(mean_df["lag"], mean_df["wis"], label="Projected", linewidth=3.0)
#     ax1.plot(mean_df["lag"], mean_df["wis_median"], label="Projected Median", linewidth=3.0)
#     ax1.plot(mean_df["lag"], mean_df["mae"], label="Baseline",linewidth=3.0)
#     ax1.fill_between(mean_df["lag"], 
#                       q5_df["wis"], 
#                       q95_df["wis"], alpha=0.1)
#     ax1.fill_between(mean_df["lag"], 
#                       q5_df["wis_median"], 
#                       q95_df["wis_median"], alpha=0.1)
#     ax1.fill_between(mean_df["lag"], 
#                       q5_df["mae"], 
#                       q95_df["mae"], alpha=0.1)
#     ax1.set_xlabel("Lag", fontsize=30)
#     if idx == 0:
#         ax1.set_ylabel("WIS", fontsize=30)
#     ax1.set_xlim((0, 14))
#     ax1.set_ylim((0, 4))
#     ax1.axhline(np.log(1.1), linestyle="--", color="gray", label = "10% Absolute Relative Error", linewidth=3.0)
#     ax1.axhline(np.log(1.2), linestyle="-.", color="gray", label = "20% Absolute Relative Error", linewidth=3.0)
#     ax1.set_xticks(np.arange(0, 15, 1))
#     # ax1.set_yticks(np.append(ax1.get_yticks(), [0.1]))
#     ax1.tick_params(axis='both', labelsize=20)
#     ax1.grid(True)
    
#     ax2 = ax1.twinx()
#     ax2.plot([],[])
#     ax2.set_yticks(ax1.get_yticks())
#     ax2.set_yticklabels(list(map(wis_to_re, ax1.get_yticks())))
#     ax2.set_ylim((ax1.get_ylim()[0], ax1.get_ylim()[1]))
#     ax2.tick_params(axis='both', labelsize=20)  
#     # ax1.legend(bbox_to_anchor=(0.8, 0.85), fontsize=25)
#     ax1.set_title(statenames[idx], fontsize=35)
#     if idx == 2:
#         ax2.set_ylabel('Absolute Relative Error', fontsize=30, rotation=270, labelpad=30)
# labels_handles = {
#   label: handle for ax in fig.axes for handle, label in zip(*ax.get_legend_handles_labels())
# }

# fig.legend(
#   labels_handles.values(),
#   labels_handles.keys(),
#   loc = "upper center",
#   bbox_to_anchor = (0.5, 0),
#   bbox_transform = plt.gcf().transFigure,
#   ncol = 2,
#   fontsize=30
# )
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(fig_dir + "eg_chng_outpatient_count_result_evl_general.pdf", bbox_inches = 'tight')



##### Example for MA-DPH
# coef = pd.read_csv("/Users/jingjingtang/Documents/backfill-recheck/paper/data/results/ma-dph/ma_coefs.csv",
#                    parse_dates=["training_end_date", "training_start_date"])

# pd.set_option('display.max_columns', None)
# coef.loc[(coef["training_end_date"] <= datetime(2022, 1, 10))
#          & (coef["training_end_date"] >= datetime(2022, 1, 6))
#          & (coef["test_lag"] == 1)
#          & (coef["tau"] == 0.5)]
 
df = dfs["MA-DPH"]
state = "ma"


fig = plt.figure(figsize=(20, 6))
for idx, lag in enumerate([1, 3, 5]):
    ax = plt.subplot(1, 3, idx+1)
    subdf = df.loc[(df["geo_value"] == state)
                    & (df["lag"] == lag)
                    & (df["time_value"] <= datetime(2022, 3, 1))
                    & (df["time_value"] >= datetime(2021, 6, 1))]
    
    plt.plot(subdf["time_value"], subdf["value_7dav"], color = "lightblue", label="Real-time Report")
    plt.plot(subdf["time_value"], subdf["value_target_7dav"], color = "tab:blue", label="Target (Lag=14)")
    plt.plot(subdf["time_value"], subdf["predicted_tau0.5"], color="tab:red", label="Projected")
    plt.fill_between(subdf["time_value"], 
                     subdf["predicted_tau0.25"],
                     subdf["predicted_tau0.75"],
                     color= "tab:red", alpha=0.3)
    # plt.yticks([0, 1, 2, 3, 4], fontsize=15)
    plt.title("Lag=%d"%lag, fontsize=30)
    plt.ylabel("#COVID-19\nConfirmed Cases", fontsize=25)
    plt.xticks(fontsize=20,rotation=45, ha = "right",)
    plt.yticks(fontsize=20)
    plt.grid()
    plt.ylim(0, 36000)
    plt.xlabel("Reference Date", fontsize=25)
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
plt.suptitle("MA-DPH", fontsize=35, y=1)
plt.tight_layout()
plt.savefig(fig_dir + "eg_madph_ma_ts.png", bbox_inches="tight")

# state = "ma"
# fig = plt.figure(figsize=(7, 5))
# # _d = datetime(2021, 7, 14)
# _d = datetime(2021, 12, 6)
# subdf = df.loc[(df["geo_value"] == state)
#                & (df["time_value"] == _d)
#                & (df["lag"] <= 7)]
# plt.plot(subdf["lag"], subdf["value_7dav"]*100, color = "gray", label = "Observed", alpha=0.8)
# plt.scatter(subdf["lag"], subdf["value_7dav"]*100, color = "gray")
# plt.plot(subdf["lag"], subdf["predicted_tau0.5"]*100, color = "tab:red", label = "Projected Median", alpha=0.8)
# plt.fill_between(subdf["lag"],
#                  subdf["predicted_tau0.25"]*100,
#                  subdf["predicted_tau0.75"]*100, color = "tab:red", alpha=0.3)
# plt.axhline(subdf["value_target_7dav"].values[0]*100, color = "tab:blue", label = "Target (Lag=60)")
# plt.grid()
# plt.xlabel("Lag", fontsize=15)
# plt.ylabel("%COVID-19\nAntigen Test", fontsize=15)
# plt.yticks(fontsize=15)
# plt.legend()
# plt.title("Reference Date = %s"%_d.date(), fontsize=20)
# # plt.title("Reference Date = 2022-01-05", fontsize=20)
# # plt.axvline(datetime(2022, 1, 5), color="gray", alpha=0.3, linewidth = 7)
# # plt.axvline(datetime(2021, 8, 23), color="gray", alpha=0.3, linewidth = 7)
# plt.savefig(fig_dir + "eg_madph_ma_ts_zoomin1.png", bbox_inches="tight")



