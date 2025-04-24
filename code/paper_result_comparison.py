#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Count projection example

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

from .constants import (signals, data_dir, fig_dir, taus,
                        filtered_states, map_list)
from ._utils_ import (read_chng_outpatient_result, read_ma_dph_result, 
                      read_quidel_result, read_chng_outpatient_count_result, re_to_wis)
dfs = {}
dfs["COVID-19 cases"] = read_ma_dph_result()
dfs["CHNG Outpatient Count"] = read_chng_outpatient_count_result()
dfs["dengue"] = pd.read_csv("/Users/jingjingtang/Documents/backfill-recheck/paper/data/results/denguedat_weekly_ref70/pr_weekly_tw728.csv",
                            parse_dates=["time_value", "issue_date"])
dfs["dengue"]["tw"] = 180
dfs["ilinat"] = pd.read_csv("/Users/jingjingtang/Documents/backfill-recheck/paper/data/results/ilinat_weekly_ref182/nat_weekly_tw189.csv",
                            parse_dates=["time_value", "issue_date"])

dfs["ilinat"]["tw"] = 180

# delphi_df = dfs[signal].groupby(["lag", "geo_value"]).agg(
#     mean=('wis', 'mean'),
#     sem=('wis', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
#     quantile_0_5=('wis', lambda x: np.quantile(x, 0.5)),         # Median (50th percentile)
#     quantile_0_95=('wis', lambda x: np.quantile(x, 0.95))
#     ).reset_index()
# delphi_df.loc[delphi_df["geo_value"] == "ak"]


nobBS_dfs={}
nobBS_dfs["CHNG Outpatient Count"] = pd.read_csv("/Users/jingjingtang/Documents/backfill-recheck/nobBS_result/chng_outpatient_covid/combined_result_daily.csv",
                           parse_dates=["time_value", "issue_date"])
nobBS_dfs["CHNG Outpatient Count"]["lag"] = [(x-y).days for x, y in zip(nobBS_dfs["CHNG Outpatient Count"]["issue_date"], nobBS_dfs["CHNG Outpatient Count"]["time_value"])]
# nobBS_dfs["CHNG Outpatient Count"]["time_value"] = nobBS_dfs["CHNG Outpatient Count"]["onset_date"] 
# nobBS_dfs["CHNG Outpatient Count"]["wis"] = abs(np.log(nobBS_dfs["CHNG Outpatient Count"]["estimate"] + 1) - np.log(nobBS_dfs["CHNG Outpatient Count"]["value_target"] + 1))

nobBS_dfs["COVID-19 cases"] = pd.read_csv("/Users/jingjingtang/Documents/backfill-recheck/nobBS_result/ma-dph/ma_daily.csv",
                           parse_dates=["time_value", "issue_date"])
nobBS_dfs["COVID-19 cases"]["lag"] = [(x-y).days for x, y in zip(nobBS_dfs["COVID-19 cases"]["issue_date"], nobBS_dfs["COVID-19 cases"]["time_value"])]
nobBS_dfs["COVID-19 cases"] = nobBS_dfs["COVID-19 cases"].loc[nobBS_dfs["COVID-19 cases"]["lag"] >= 1]

nobBS_dfs["dengue"] = pd.read_csv("/Users/jingjingtang/Documents/backfill-recheck/nobBS_result/dengue/combined_result_weekly.csv", parse_dates=["issue_date", "time_value"])
nobBS_dfs["dengue"]["lag"] = [(x-y).days for x, y in zip(nobBS_dfs["dengue"]["issue_date"], nobBS_dfs["dengue"]["time_value"])]

nobBS_dfs["ilinat"]= pd.read_csv("/Users/jingjingtang/Documents/backfill-recheck/nobBS_result/ilicases/ilinat_weekly.csv", parse_dates=["issue_date", "time_value"])
nobBS_dfs["ilinat"]["lag"] = [(x-y).days for x, y in zip(nobBS_dfs["ilinat"]["issue_date"], nobBS_dfs["ilinat"]["time_value"])]


epinowcast_dfs = {}
pdList = []
for state in map_list:
    try:
        temp = pd.read_csv("/Users/jingjingtang/Documents/backfill-recheck/epinowcast_result/chng_outpatient_covid/%s.csv"%state,
                                   parse_dates=["reference_date", "report_date"])
        temp["geo_value"] = state
        pdList.append(temp)
    except:
        pass
epinowcast_dfs["CHNG Outpatient Count"] = pd.concat(pdList)
epinowcast_dfs["CHNG Outpatient Count"]["lag"] = [(x-y).days for x, y in zip(epinowcast_dfs["CHNG Outpatient Count"]["report_date"], epinowcast_dfs["CHNG Outpatient Count"]["reference_date"])]
epinowcast_dfs["CHNG Outpatient Count"]["time_value"] = epinowcast_dfs["CHNG Outpatient Count"]["reference_date"] 

epinowcast_dfs["COVID-19 cases"] = pd.read_csv("/Users/jingjingtang/Documents/backfill-recheck/epinowcast_result/ma-dph/ma.csv",
                           parse_dates=["reference_date", "report_date"])
epinowcast_dfs["COVID-19 cases"]["geo_value"] = "ma"
epinowcast_dfs["COVID-19 cases"]["lag"] = [(x-y).days for x, y in zip(epinowcast_dfs["COVID-19 cases"]["report_date"], epinowcast_dfs["COVID-19 cases"]["reference_date"])]
epinowcast_dfs["COVID-19 cases"]["time_value"] = epinowcast_dfs["COVID-19 cases"]["reference_date"] 

epinowcast_dfs["dengue"] = pd.read_csv("/Users/jingjingtang/Documents/backfill-recheck/epinowcast_result/dengue/pr_weekly.csv", parse_dates=["report_date", "reference_date"])
part2 = pd.read_csv("/Users/jingjingtang/Documents/backfill-recheck/epinowcast_result/dengue/pr_weekly_part2.csv", parse_dates=["report_date", "reference_date"])
epinowcast_dfs["dengue"] = pd.concat([epinowcast_dfs["dengue"], part2])
epinowcast_dfs["dengue"]["lag"] = [(x-y).days for x, y in zip(epinowcast_dfs["dengue"]["report_date"], epinowcast_dfs["dengue"]["reference_date"])]
epinowcast_dfs["dengue"]["time_value"] = epinowcast_dfs["dengue"]["reference_date"] 
epinowcast_dfs["dengue"]["issue_date"] = epinowcast_dfs["dengue"]["report_date"] 

epinowcast_dfs["ilinat"] = pd.read_csv("/Users/jingjingtang/Documents/backfill-recheck/epinowcast_result/ilicases/ilinat_weekly.csv", parse_dates=["report_date", "reference_date"])
epinowcast_dfs["ilinat"]["lag"] = [(x-y).days for x, y in zip(epinowcast_dfs["ilinat"]["report_date"], epinowcast_dfs["ilinat"]["reference_date"])]
epinowcast_dfs["ilinat"]["time_value"] = epinowcast_dfs["ilinat"]["reference_date"] 
####################################
### COUNT EXAMPLES
####################################

df = dfs["CHNG Outpatient Count"]
df = df.loc[(df["issue_date"] == df["issue_date"].isin(dfs["epinowcast"]["test_date"]))]
lag = 0
fig = plt.figure(figsize=(60, 42))
for i in range(len(map_list)):
    state = map_list[i]
    if state in ['', 'dc', 'al']:
        continue
    try:
        # dfs["epinowcast"] = pd.read_csv("/Users/jingjingtang/Documents/backfill-recheck/chng_outpatient_covid_epinowcast/%s.csv"%state,
        #                            parse_dates=["reference_date", "report_date"])
        # dfs["epinowcast"]["lag"] = [(x-y).days for x, y in zip(dfs["epinowcast"]["report_date"], dfs["epinowcast"]["reference_date"])]
        # dfs["epinowcast"]["time_value"] = dfs["epinowcast"]["reference_date"] 
        subdf = df.loc[(df["geo_value"] == state)
                        & (df["lag"] == lag)
                        & (df["time_value"] <= datetime(2022, 3, 1))
                        & (df["time_value"] >= datetime(2021, 6, 1))]
        nobBS = dfs["nobBS"].loc[(dfs["nobBS"]["time_value"] <= datetime(2022, 3, 1))
                                  &(dfs["nobBS"]["time_value"] >= datetime(2021, 6, 1) )
                                  & (dfs["nobBS"]["lag"] == lag)
                                  & (dfs["nobBS"]["geo_value"] == state)]
        epinowcast = dfs["epinowcast"].loc[(dfs["epinowcast"]["time_value"] <= datetime(2022, 3, 1))
                                  &(dfs["epinowcast"]["time_value"] >= datetime(2021, 6, 1) )
                                  & (dfs["epinowcast"]["lag"] == lag)
                                  & (dfs["epinowcast"]["geo_value"] == state)]
        plt.subplot(7, 11, i+1)
        plt.plot(nobBS["time_value"], nobBS["estimate"], color = "tab:orange", label="nobBS")
        plt.plot(epinowcast["time_value"], epinowcast["q50"], color = "tab:green", label="epinowcast")
        plt.plot(subdf["time_value"], subdf["value_7dav_covid"], color = "lightblue", label="Provisional")
        plt.plot(subdf["time_value"], subdf["value_target_covid"], color = "tab:blue", label="Target (Lag=60)")
        plt.plot(subdf["time_value"], subdf["predicted_tau0.5_covid"], color="tab:red", 
                 label="Projected", alpha=0.8)
        plt.fill_between(subdf["time_value"], 
                         subdf["predicted_tau0.25_covid"],
                         subdf["predicted_tau0.75_covid"],
                         color= "tab:red", alpha=0.3)
        plt.xticks(rotation=45, ha="right", fontsize=20)
        plt.title("%s"%state.upper(), fontsize=40)
        plt.ylabel("#COVID Claims", fontsize=40)
        plt.xlabel("Reference Date", fontsize=40)
        plt.grid()
    except:
        pass
plt.suptitle("CHNG Outpatient COVID count, lag=%d"%lag, fontsize=55, y = 1.01)
# plt.legend(fontsize=20,bbox="lower right")
plt.tight_layout()
plt.savefig(fig_dir+"eg_chng_outpatient_count_comparison_with_nobBS_lag%d.png"%lag, bbox_inches="tight")



####################################
### # WIS comparison
####################################
titles = {
    "COVID-19 cases": "COVID-19 cases in MA\n(Daily, State)", 
    "CHNG Outpatient Count": "Insurance claims\n(Daily, State)", 
    "dengue": "Dengue fever cases\n(Weekly, State, PR only)",
    "ilinat": "ILI cases\n(Weekly, National)"
    }

plt.style.use('default')
fig = plt.figure(figsize=(16, 12))
for idx, signal in enumerate(["COVID-19 cases", "CHNG Outpatient Count", "dengue", "ilinat"]):
    
    nobBS_df = nobBS_dfs[signal].groupby(["lag"]).agg(
        mean=('wis', 'mean'),
        sem=('wis', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
        quantile10=('wis', lambda x: np.quantile(x.dropna(), 0.1)),         # Median (50th percentile)
        quantile90=('wis', lambda x: np.quantile(x.dropna(), 0.9))
        ).reset_index()
    
    epinowcast_df = epinowcast_dfs[signal].groupby(["lag"]).agg(
        mean=('wis', 'mean'),
        sem=('wis', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
        quantile10=('wis', lambda x: np.quantile(x, 0.1)),         # Median (50th percentile)
        quantile90=('wis', lambda x: np.quantile(x, 0.9))
        ).reset_index()
    
    delphi_df = dfs[signal].loc[
        (dfs[signal]["tw"] == 180)
        & (dfs[signal]["issue_date"].isin(epinowcast_dfs[signal]["test_date"].unique()))
        ].groupby(["lag"]).agg(
        mean=('wis', 'mean'),
        sem=('wis', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
        quantile10=('wis', lambda x: np.quantile(x, 0.1)),         # Median (50th percentile)
        quantile90=('wis', lambda x: np.quantile(x, 0.9))
        ).reset_index()
       
    ax1 = plt.subplot(2, 2, idx+1)
    ax1.plot(delphi_df["lag"], delphi_df["mean"], label="Delphi-RF", linewidth=3.0, color="tab:orange")
    ax1.fill_between(delphi_df["lag"], 
                      delphi_df["quantile10"], 
                      delphi_df["quantile90"], alpha=0.1, color="tab:orange")
    ax1.plot(nobBS_df["lag"], nobBS_df["mean"], label="nobBS", linewidth=3.0, color="tab:blue")
    ax1.fill_between(nobBS_df["lag"], 
                     nobBS_df["quantile10"], 
                     nobBS_df["quantile90"], alpha=0.1, color="tab:blue")
    ax1.plot(epinowcast_df["lag"], epinowcast_df["mean"], label="Epinowcast", linewidth=3.0, color="tab:purple")
    ax1.fill_between(epinowcast_df["lag"], 
                     epinowcast_df["quantile10"], 
                     epinowcast_df["quantile90"], alpha=0.1, color="tab:purple")   
    ax1.grid(True)
    ax1.set_xlabel("Lag (Days)", fontsize=30)
    # ax1.set_ylabel("WIS", fontsize=30)
    if idx in [0, 2]:
        ax1.set_ylabel("WIS", fontsize=30)
    if idx < 2:
        ax1.set_xlim((0, 14))
        ax1.set_ylim((0, 0.62))
        yticks = [0, 5, 10, 20, 50]
        ax1.set_xticks(np.arange(0, 15, 1))
    else:
        ax1.set_xlim((0, 57))
        ax1.set_ylim((0, 0.19))
        ax1.set_xticks(np.arange(0, 57, 7))
        yticks = [0, 5, 10, 20]
   
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



####### Dengue comparison
signal = "ilinat"
nobBS_df = nobBS_dfs[signal].loc[
    (nobBS_dfs[signal]["lag"] == 0)].sort_values("time_value")
plt.figure(figsize=(15, 4))
plt.plot(nobBS_df["time_value"], nobBS_df["value_target"], label="target")
plt.plot(nobBS_df["time_value"], nobBS_df["estimate"], label="nobBS")
plt.legend()

delphi_df = dfs[signal].loc[
    (dfs[signal]["lag"] == 0)
    & (dfs[signal]["issue_date"].isin(epinowcast_dfs[signal]["test_date"].unique()))].sort_values("time_value")
plt.figure(figsize=(15, 4))
plt.plot(delphi_df["time_value"], np.exp(delphi_df["log_value_target_7dav"]), label="target")
plt.plot(delphi_df["time_value"], np.exp(delphi_df["predicted_tau0.5"]), label="nobBS")
plt.legend()

epinowcast_df = epinowcast_dfs[signal].loc[
    (epinowcast_dfs[signal]["lag"] == 0)].sort_values("time_value")
plt.figure(figsize=(15, 4))
plt.plot(epinowcast_df["time_value"], epinowcast_df["value_target"], label="target")
plt.plot(epinowcast_df["time_value"], epinowcast_df["q50"], label="nobBS")
plt.legend()




