#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Experimental Details
# Fraction examples for CHNG and Quidel

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
from ._utils_ import (read_chng_outpatient_result, read_ma_dph_result, read_quidel_result)

### Read results
dfs = {}
dfs["CHNG Outpatient"] = read_chng_outpatient_result()
dfs["Quidel"] = read_quidel_result()
dfs["MA-DPH"] = read_ma_dph_result()
dfs["Quidel"]["lag"] = dfs["Quidel"]["lag"] + 1
dfs["CHNG Oupatient Count"] = read_chng_outpatient_count_result()
dfs["CHNG Oupatient Count"]["lag"] =  dfs["CHNG Oupatient Count"]["lag"]-1

dfs_raw = {}
dfs_raw["CHNG Outpatient"] = read_chng_outpatient()
dfs_raw["MA-DPH"] = read_ma_dph()
dfs_raw["Quidel"] = read_quidel() 

####################################
### frc pred example for proposal
####################################
x_min=18837.55
x_max=18891.45
y_min=0.0054530262892008245
y_max=0.022
t = datetime(2021, 8, 22)
rd = datetime(2021, 8, 22)
df = dfs["CHNG Outpatient"].copy()
df_raw = dfs_raw["CHNG Outpatient"].copy()

subdf = df.loc[(df["geo_value"] == "ca") & (df["issue_date"] == t)
               & (df["time_value"] == rd)]
subdf_target = df.loc[(df["geo_value"] == "ca") & (df["time_value"] <= rd),
                      ["time_value", "value_target_7dav"]].drop_duplicates()
subdf_raw = df_raw.loc[(df_raw["geo_value"] == "ca") & (df_raw["issue_date"] == t)
                       & (df_raw["time_value"] <= rd)]
xticks = [datetime(2021, 9, 11) + timedelta(days=x) for x in list(range(-56, 1, 7))]


# red = 1. - 25/120
# blue = 25/120
# plt.figure(figsize=(6, 4))
# plt.plot(subdf_target["time_value"], subdf_target["value_target_7dav"], color="darkblue")
# plt.plot(subdf_raw["time_value"], subdf_raw["7dav_frac"], 
#           color = (red, 0, blue), alpha = 0.1 + (0.1 *4))
# for tau in taus[:4]:
#     d = subdf["time_value"].values[0]
#     _upper = subdf["predicted_tau%s"%str(1-tau)].values[0]
#     _lower = subdf["predicted_tau%s"%str(tau)].values[0]
#     plt.fill_between([d- pd.Timedelta(hours=12), 
#                       d+ pd.Timedelta(hours=12)],
#                      [_lower, _lower],
#                      [_upper, _upper], 
#                 color="tab:orange", alpha=0.7-abs(0.5-tau))
#     plt.text(d+pd.Timedelta(hours=24), _lower, r"$\tau$=%s"%str(tau), color="black")
#     plt.text(d+pd.Timedelta(hours=24), _upper, r"$\tau$=%s"%str(1-tau), color="black")

plt.scatter(d, _upper, s=20)
# plt.plot(subdf["time_value"], subdf["predicted_tau%s"%str(tau)], color="red", alpha=0.7-abs(0.5-tau),
#           label="Projected")
plt.xticks([rd], fontsize=15)
plt.yticks(fontsize=15)
plt.xlim(datetime(2021, 7, 14), datetime(2021, 9, 11))
# plt.xlabel("Reference Date", fontsize=20)
plt.yticks([])
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
# plt.ylabel("California\n% COVID Confirmed\nDoctor Visits", fontsize=20)
# plt.title("Backfill Correction in California", fontsize=25)
# plt.ylim(0, 120)


####################################
### frc pred example
####################################


plt.figure(figsize=(30, 6))
plt.subplot(1, 2, 1)
subdf = df.loc[(df["geo_value"] == "ma") & (df["issue_date"] == datetime(2021, 9, 11))]
xticks = [datetime(2021, 9, 11) + timedelta(days=x) for x in list(range(-56, 1, 7))]
plt.axvspan(
    xmin=datetime(2021, 9, 11),
    xmax=datetime(2021, 9, 20),
    alpha=0.2,
    color='gray'
)
plt.plot(subdf["time_value"], subdf["value_7dav"], color="gray", label="Observed")
plt.plot(subdf["time_value"], subdf["value_target_7dav"], color="black", label="Target(Lag=60)")
for tau in taus:
    plt.plot(subdf["time_value"], subdf["predicted_tau%s"%str(tau)], color="red", alpha=0.7-abs(0.5-tau))
plt.plot(subdf["time_value"], subdf["predicted_tau%s"%str(tau)], color="red", alpha=0.7-abs(0.5-tau),
          label="Projected")
plt.xticks(xticks, rotation=45, fontsize=15)
plt.yticks(fontsize=15)
plt.xlim(datetime(2021, 7, 14), datetime(2021, 9, 11))
plt.xlabel("Reference Date", fontsize=20)
plt.ylabel("COVID-19 Fraction", fontsize=20)
plt.title("Backfill Correction in California", fontsize=25)
# plt.ylim(0, 120)
plt.grid(True)


plt.subplot(1, 2, 2)
subdf = df.loc[(df["geo_value"] == "wy") & (df["issue_date"] == datetime(2021, 9, 12))]
xticks = [datetime(2021, 9, 11) + timedelta(days=x) for x in list(range(-56, 1, 7))]

plt.axvspan(
    xmin=datetime(2021, 9, 11),
    xmax=datetime(2021, 9, 20),
    alpha=0.2,
    color='gray'
)
plt.plot(subdf["time_value"], subdf["value_raw"], color="gray", label="Observed")
plt.plot(subdf["time_value"], subdf["value_target"], color="black", label="Target(lag=60)")
for tau in taus:
    plt.plot(subdf["time_value"], subdf["predicted_tau%s"%str(tau)], color="red", alpha=0.7-abs(0.5-tau))
plt.plot(subdf["time_value"], subdf["predicted_tau%s"%str(tau)], color="red", alpha=0.7-abs(0.5-tau),
          label="Projected")
plt.xticks(xticks, rotation=45, fontsize=15)
plt.yticks(fontsize=15)
plt.xlim(datetime(2021, 7, 14), datetime(2021, 9, 20))
plt.xlabel("Reference Date", fontsize=20)
plt.ylabel("COVID-19 Fraction", fontsize=20)
plt.title("Backfill Correction in Wyoming", fontsize=25)
# plt.ylim(0, 120)
plt.grid(True)
plt.legend(bbox_to_anchor=(0.3, -0.3), fontsize=20, ncol=3)

  
####################################
### CHNG outpatient result in general
### Three datasets share label? 
####################################

# How to choose the example? 
# Largest population state and lowest population state
# Lowever epidemic level and higher


df = dfs["CHNG Outpatient"]
df_raw = dfs_raw["CHNG Outpatient"]
state = "ca"


plt.figure(figsize=(15, 3))
subdf = df.loc[(df["geo_value"] == state)
                & (df["lag"] == 0)
                & (df["time_value"] <= datetime(2022, 3, 1))
                & (df["time_value"] >= datetime(2021, 6, 1))]
subdf["value_7dav"] = (subdf["value_7dav_covid"] + 1) / (subdf["value_7dav_total"] + 1)

plt.plot(subdf["time_value"], subdf["value_7dav"]*100, color = "lightblue", label="First Release (Lag=0)")
plt.plot(subdf["time_value"], subdf["value_target_7dav"]*100, color = "tab:blue", label="Target (Lag=60)")
plt.legend(fontsize=15)
plt.yticks([0, 1, 2, 3, 4], fontsize=15)
plt.title("CHNG Outpatient, California", fontsize=20)
plt.ylabel("%COVID-19\nInsurance Claims", fontsize=15)
plt.xticks(fontsize=15)
plt.grid()
plt.axvspan(datetime(2021, 8, 2), datetime(2021, 9,6), alpha=0.25, color='gray')
plt.axvspan(datetime(2021, 12, 15), datetime(2022, 1, 19), alpha=0.25, color='gray')
plt.xlabel("Reference Date", fontsize=15)
plt.savefig(fig_dir + "eg_quidel_ca_ts.png", bbox_inches="tight")


fig = plt.figure(figsize=(7, 2))
lags = [0, 3, 7, 14, 21, 28]
start_date = datetime(2021, 8, 2) # Sunday
end_date = datetime(2021, 8, 9)
# start_date = datetime(2021, 12, 15) # Sunday
# end_date = datetime(2021, 12, 22)
# h = "1f77b4"
# tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
# start_color = (214, 39, 40)  #tab:red : #d62728
# end_color = (31, 119, 180)    #tab:blue : #1f77b4

_ds = []
for i in range(len(lags)):
    lag = lags[i]
    _t = end_date + timedelta(days = lag)
    # red = start_color[0]/255 + (end_color[0] - start_color[0])/60/255 * lag
    # green = start_color[1]/255 + (end_color[1] - start_color[1])/60/255 * lag
    # blue = start_color[2]/255 + (end_color[2] - start_color[2])/60/255 * lag
    subdf = df_raw.loc[(df_raw["geo_value"] == state)
                    & (df_raw["time_value"] >= start_date)
                    & (df_raw["issue_date"] == _t)]
    plt.plot(subdf["time_value"], subdf["7dav_frac"]*100, color = colors[i], alpha = 0.6)
    _ds.append(_t.date())
subdf = df_raw.loc[(df_raw["geo_value"] == state)
                & (df_raw["time_value"] >= start_date)
                & (df_raw["lag"] == 150)
                & (df_raw["time_value"] <= _t)]
plt.plot(subdf["time_value"], subdf["7dav_frac"]*100, color = "tab:blue", label = "Target (Lag=60)")
plt.grid()
plt.yticks(fontsize=15)
plt.xticks(_ds, rotation=45, ha = "right", fontsize=15)
plt.legend()
# plt.axvline(datetime(2022, 1, 5), color="gray", alpha=0.3, linewidth = 7)
plt.axvline(datetime(2021, 8, 23), color="gray", alpha=0.3, linewidth = 7)
plt.savefig(fig_dir + "eg_wy_ts_zoomin1.png", bbox_inches="tight")


plt.figure(figsize=(15, 3))
subdf = df.loc[(df["geo_value"] == state)
                & (df["lag"] == 7)]
plt.plot(subdf["time_value"], subdf["wis"])


state = "ca"
fig = plt.figure(figsize=(7, 5))
_d = datetime(2021, 8, 23)
# _d = datetime(2022, 1, 5)
subdf = df.loc[(df["geo_value"] == state)
               & (df["time_value"] == _d)
               & (df["lag"] <= 7)]
plt.plot(subdf["lag"], subdf["value_7dav"]*100, color = "gray", label = "Observed", alpha=0.8)
plt.scatter(subdf["lag"], subdf["value_7dav"]*100, color = "gray")
plt.plot(subdf["lag"], subdf["predicted_tau0.5"]*100, color = "tab:red", label = "Projected Median", alpha=0.8)
plt.fill_between(subdf["lag"],
                 subdf["predicted_tau0.25"]*100,
                 subdf["predicted_tau0.75"]*100, color = "tab:red", alpha=0.3)
plt.axhline(subdf["value_target_7dav"].values[0]*100, color = "tab:blue", label = "Target (Lag=60)")
plt.grid()
plt.xlabel("Lag", fontsize=15)
plt.ylabel("%COVID-19\nInsurance Claims", fontsize=15)
plt.yticks(fontsize=15)
plt.legend()
plt.title("Reference Date = 2021-08-23", fontsize=20)
# plt.title("Reference Date = 2022-01-05", fontsize=20)
# plt.axvline(datetime(2022, 1, 5), color="gray", alpha=0.3, linewidth = 7)
plt.axvline(datetime(2021, 8, 23), color="gray", alpha=0.3, linewidth = 7)
plt.savefig(fig_dir + "eg_wy_ts_zoomin3.png", bbox_inches="tight")




## example for wy
state = "md"
statename = "California"


plt.figure(figsize=(15, 3))
subdf = df.loc[(df["geo_value"] == state)
                & (df["lag"] == 0)
                & (df["time_value"] <= datetime(2022, 3, 1))
                & (df["time_value"] >= datetime(2021, 6, 1))]
subdf["value_7dav"] = (subdf["value_7dav_covid"] + 1) / (subdf["value_7dav_total"] + 1)

plt.plot(subdf["time_value"], subdf["value_7dav"]*100, color = "lightblue", label="First Release (Lag=0)")
plt.plot(subdf["time_value"], subdf["value_target_7dav"]*100, color = "tab:blue", label="Target (Lag=60)")
plt.legend(fontsize=15)
plt.yticks([0, 1, 2, 3, 4], fontsize=15)
plt.title("CHNG Outpatient, %s"%statename, fontsize=20)
plt.ylabel("%COVID-19\nInsurance Claims", fontsize=15)
plt.xticks(fontsize=15)
plt.grid()
plt.axvspan(datetime(2021, 8, 30), datetime(2021, 9,27), alpha=0.25, color='gray')
plt.axvspan(datetime(2021, 12, 27), datetime(2022, 1, 24), alpha=0.25, color='gray')
plt.xlabel("Reference Date", fontsize=15)
plt.savefig(fig_dir + "eg_chng_wy_ts.png", bbox_inches="tight")


fig = plt.figure(figsize=(7, 2))
lags = [0, 3, 7, 14, 21, 28]
# start_date = datetime(2021, 8, 30) # Sunday
# end_date = datetime(2021, 9, 6)
start_date = datetime(2021, 12, 27) # Sunday
end_date = datetime(2022, 1, 3)
# h = "1f77b4"
# tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
# start_color = (214, 39, 40)  #tab:red : #d62728
# end_color = (31, 119, 180)    #tab:blue : #1f77b4
_ds = []
for i in range(len(lags)):
    lag = lags[i]
    _t = end_date + timedelta(days = lag)
    # red = start_color[0]/255 + (end_color[0] - start_color[0])/60/255 * lag
    # green = start_color[1]/255 + (end_color[1] - start_color[1])/60/255 * lag
    # blue = start_color[2]/255 + (end_color[2] - start_color[2])/60/255 * lag
    subdf = df_raw.loc[(df_raw["geo_value"] == state)
                    & (df_raw["time_value"] >= start_date)
                    & (df_raw["issue_date"] == _t)]
    plt.plot(subdf["time_value"], subdf["7dav_frac"]*100, color = colors[i], alpha = 0.6)
    _ds.append(_t.date())
subdf = df_raw.loc[(df_raw["geo_value"] == state)
                & (df_raw["time_value"] >= start_date)
                & (df_raw["lag"] == 150)
                & (df_raw["time_value"] <= _t)]
plt.plot(subdf["time_value"], subdf["7dav_frac"]*100, color = "tab:blue", label = "Target (Lag=60)")
plt.grid()
plt.yticks(fontsize=15)
plt.xticks(_ds, rotation=45, ha = "right", fontsize=15)
plt.legend()
# plt.axvline(datetime(2021, 9, 20), color="gray", alpha=0.3, linewidth = 7)
plt.axvline(datetime(2022, 1, 17), color="gray", alpha=0.3, linewidth = 7)
plt.savefig(fig_dir + "eg_wy_ts_zoomin2.png", bbox_inches="tight")

state = "wy"
plt.figure(figsize=(15, 3))
subdf = df.loc[(df["geo_value"] == state)
                & (df["lag"] == 7)]
plt.plot(subdf["time_value"], subdf["wis"])


state = "wy"
fig = plt.figure(figsize=(7, 5))
_d = datetime(2022, 1, 17)
# _d = datetime(2022, 1, 5)
subdf = df.loc[(df["geo_value"] == state)
               & (df["time_value"] == _d)
               & (df["lag"] <= 7)]
plt.plot(subdf["lag"], subdf["value_7dav"]*100, color = "gray", label = "Observed", alpha=0.8)
plt.scatter(subdf["lag"], subdf["value_7dav"]*100, color = "gray")
plt.plot(subdf["lag"], subdf["predicted_tau0.5"]*100, color = "tab:red", label = "Projected Median", alpha=0.8)
plt.fill_between(subdf["lag"],
                 subdf["predicted_tau0.25"]*100,
                 subdf["predicted_tau0.75"]*100, color = "tab:red", alpha=0.3)
plt.axhline(subdf["value_target_7dav"].values[0]*100, color = "tab:blue", label = "Target (Lag=60)")
plt.grid()
plt.xlabel("Lag", fontsize=15)
plt.ylabel("%COVID-19\nInsurance Claims", fontsize=15)
plt.yticks(fontsize=15)
plt.legend()
plt.title("Reference Date = %s"%_d.date(), fontsize=20)
# plt.title("Reference Date = 2022-01-05", fontsize=20)
# plt.axvline(datetime(2022, 1, 5), color="gray", alpha=0.3, linewidth = 7)
# plt.axvline(datetime(2021, 8, 23), color="gray", alpha=0.3, linewidth = 7)
plt.savefig(fig_dir + "eg_wy_ts_zoomin4.png", bbox_inches="tight")

#####################################
#####################################
##### Example for Quidel ############
#####################################
#####################################

# Example shooting
plt.figure(figsize=(15, 4))
mean_df = df.groupby(["time_value", "geo_value"]).mean().reset_index()
mean_df = mean_df.loc[mean_df["geo_value"] == "wy"]
plt.plot(mean_df["time_value"], mean_df["wis"], label="projected")
plt.plot(mean_df["time_value"], mean_df["mae"], label="raw")
plt.legend()
 
df = dfs["Quidel"]
# df_raw = dfs_raw["Quidel"]
state = "wy"

plt.figure(figsize=(15, 3))
subdf = df.loc[(df["geo_value"] == state)
                & (df["lag"] == 1)
                & (df["time_value"] <= datetime(2022, 3, 1))
                & (df["time_value"] >= datetime(2021, 6, 1))]
subdf["value_7dav"] = (subdf["value_7dav_covid"] + 1) / (subdf["value_7dav_total"] + 1)

plt.plot(subdf["time_value"], subdf["value_7dav"]*100, color = "lightblue", label="First Release (Lag=0)")
plt.plot(subdf["time_value"], subdf["value_target_7dav"]*100, color = "tab:blue", label="Target (Lag=60)")
plt.legend(fontsize=15)
plt.yticks([0, 1, 2, 3, 4], fontsize=15)
plt.title("CHNG Outpatient, California", fontsize=20)
plt.ylabel("%COVID-19\nInsurance Claims", fontsize=15)
plt.xticks(fontsize=15)
plt.grid()
plt.axvspan(datetime(2021, 8, 2), datetime(2021, 9,6), alpha=0.25, color='gray')
plt.axvspan(datetime(2021, 12, 15), datetime(2022, 1, 19), alpha=0.25, color='gray')
plt.xlabel("Reference Date", fontsize=15)


state = "ca"
fig = plt.figure(figsize=(7, 5))
# _d = datetime(2021, 7, 26)
# _d = datetime(2021, )
_d = datetime(2021, 12, 20)
subdf = df.loc[(df["geo_value"] == state)
               & (df["time_value"] == _d)
               & (df["lag"] <= 7)]
predicted_cols = ['predicted_tau0.01', 'predicted_tau0.025', 'predicted_tau0.1',
'predicted_tau0.25', 'predicted_tau0.5', 'predicted_tau0.75',
'predicted_tau0.9', 'predicted_tau0.975', 'predicted_tau0.99']

res = subdf[predicted_cols].values
res.sort(axis=1)
subdf.loc[:, predicted_cols] = res
plt.plot(subdf["lag"], subdf["value_7dav"]*100, color = "gray", label = "Observed", alpha=0.8)
plt.scatter(subdf["lag"], subdf["value_7dav"]*100, color = "gray")
plt.plot(subdf["lag"], subdf["predicted_tau0.5"]*100, color = "tab:red", label = "Projected Median", alpha=0.8)
plt.fill_between(subdf["lag"],
                 subdf["predicted_tau0.25"]*100,
                 subdf["predicted_tau0.75"]*100, color = "tab:red", alpha=0.3)
plt.axhline(subdf["value_target_7dav"].values[0]*100, color = "tab:blue", label = "Target (Lag=60)")
plt.grid()
plt.xlabel("Lag", fontsize=25)
plt.ylabel("%COVID-19\nAntigen Test", fontsize=25)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.legend()
plt.title("Reference Date=%s, California"%_d.date(), fontsize=20)
# plt.title("Reference Date = 2022-01-05", fontsize=20)
# plt.axvline(datetime(2022, 1, 5), color="gray", alpha=0.3, linewidth = 7)
# plt.axvline(datetime(2021, 8, 23), color="gray", alpha=0.3, linewidth = 7)
plt.savefig(fig_dir + "eg_quidel_ca_ts_zoomin2.png", bbox_inches="tight")



state = "wy"
fig = plt.figure(figsize=(7, 5))
# _d = datetime(2021, 7, 26)
_d = datetime(2021, 12, 27)
subdf = df.loc[(df["geo_value"] == state)
               & (df["time_value"] == _d)
               & (df["lag"] <= 7)]
res = subdf[predicted_cols].values
res.sort(axis=1)
subdf.loc[:, predicted_cols] = res
plt.plot(subdf["lag"], subdf["value_7dav"]*100, color = "gray", label = "Observed", alpha=0.8)
plt.scatter(subdf["lag"], subdf["value_7dav"]*100, color = "gray")
plt.plot(subdf["lag"], subdf["predicted_tau0.5"]*100, color = "tab:red", label = "Projected Median", alpha=0.8)
plt.fill_between(subdf["lag"],
                 subdf["predicted_tau0.25"]*100,
                 subdf["predicted_tau0.75"]*100, color = "tab:red", alpha=0.3)
plt.axhline(subdf["value_target_7dav"].values[0]*100, color = "tab:blue", label = "Target (Lag=60)")
plt.grid()
plt.xlabel("Lag", fontsize=15)
plt.ylabel("%COVID-19\nAntigen Test", fontsize=15)
plt.yticks(fontsize=15)
plt.legend()
plt.title("Reference Date=%s, Wyoming"%_d.date(), fontsize=20)
# plt.title("Reference Date = 2022-01-05", fontsize=20)
# plt.axvline(datetime(2022, 1, 5), color="gray", alpha=0.3, linewidth = 7)
# plt.axvline(datetime(2021, 8, 23), color="gray", alpha=0.3, linewidth = 7)
plt.savefig(fig_dir + "eg_quidel_wy_ts_zoomin2.png", bbox_inches="tight")



statenames = {"ca": "California",
              "wy": "Wyoming"}
yticks = {
    "ca": [0, 5, 10, 20],
    "wy": [0, 10, 20, 50, 80]}
plt.style.use('default')
fig = plt.figure(figsize=(15, 6))
for idx, state in enumerate(["ca", "wy"]):
    df = dfs["Quidel"].loc[dfs["Quidel"]["geo_value"] == state].copy()
    
    mean_df = df.groupby(["lag"]).mean().reset_index()
    q5_df = df.groupby(["lag"]).quantile(0.05).reset_index()
    q95_df = df.groupby(["lag"]).quantile(0.95).reset_index()
       
    ax1 = plt.subplot(1, 2, idx+1)
    ax1.plot(mean_df["lag"], mean_df["wis"], label="Projected", linewidth=3.0)
    # ax1.plot(mean_df["lag"], mean_df["wis_median"], label="Projected Median", linewidth=3.0)
    ax1.plot(mean_df["lag"], mean_df["mae"], label="Baseline",linewidth=3.0)
    ax1.fill_between(mean_df["lag"], 
                      q5_df["wis"], 
                      q95_df["wis"], alpha=0.1)
    # ax1.fill_between(mean_df["lag"], 
    #                   q5_df["wis_median"], 
    #                   q95_df["wis_median"], alpha=0.1)
    ax1.fill_between(mean_df["lag"], 
                      q5_df["mae"], 
                      q95_df["mae"], alpha=0.1)
    ax1.grid(True)
    ax1.set_xlabel("Lag", fontsize=25)
    if idx == 0:
        ax1.set_ylabel("WIS", fontsize=25)
    ax1.set_xlim((0, 14))

    ax1.axhline(np.log(1.1), linestyle="--", color="gray", label = "10% Absolute Relative Error", linewidth=3.0)
    ax1.axhline(np.log(1.2), linestyle="-.", color="gray", label = "20% Absolute Relative Error", linewidth=3.0)
    ax1.set_xticks(np.arange(0, 15, 1))
    # ax1.set_yticks(np.append(ax1.get_yticks(), [0.1]))
    ax1.tick_params(axis='both', labelsize=20)
    
    ax2 = ax1.twinx()
    ax2.plot([],[])
    # ax2.set_yticks(ax1.get_yticks())
    # ax2.set_yticklabels(list(map(wis_to_re, ax1.get_yticks())))
    ax2.set_yticks(list(map(re_to_wis, yticks[state])))
    ax2.set_yticklabels([str(x)+"%" for x in yticks[state]])
    ax2.set_ylim((ax1.get_ylim()[0], ax1.get_ylim()[1]))
    ax2.tick_params(axis='both', labelsize=20)  
    # ax1.legend(bbox_to_anchor=(0.8, 0.85), fontsize=25)
    ax1.set_title(statenames[state], fontsize=30)
    if idx == 2:
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
  ncol = 2,
  fontsize=30
)
plt.suptitle("Quidel", fontsize=35, y=1)
plt.grid(True)
plt.tight_layout()

plt.savefig(fig_dir + "eg_quidel_ca_wy_general.png", bbox_inches="tight")





