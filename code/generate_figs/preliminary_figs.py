#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Introduction

# Preliminaries

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

from delphi_utils import GeoMapper

from constants import (signals, data_dir, fig_dir)
from _utils_ import *

gmpr = GeoMapper()

# Read the raw data

dfs = {}
dfs["Insurance claims"] = read_raw_data_with_revisions(chng_fraction_config)
dfs["COVID-19 cases"] = read_raw_data_with_revisions(madph_config)
dfs["Antigen tests"] = read_raw_data_with_revisions(quidel_config)

####################################
### Fig 1: backfill problem
####################################
state = "ma"
  
fig = plt.figure(figsize = (20, 6))
for signal in ['Insurance claims', 'Antigen tests', 'COVID-19 cases']:
    print(signal)
    subdf = dfs[signal].loc[(dfs[signal]["lag"] <= 180)
                         & (dfs[signal]["geo_value"] == state)]
    subdf.index = list(range(subdf.shape[0]))

      
    ax1 = plt.subplot(1, 2, 1)
    df_covid = create_pivot_table_for_heatmap(subdf, "value_covid", melt=True).dropna()  
    df_covid["value_covid"] = df_covid["value_covid"]*100
    # df_covid = df_covid.loc[df_covid["lag"] <= 125]
    
    summary_covid = df_covid.groupby(["lag"]).agg(
        mean=('value_covid', 'mean'),
        sem=('value_covid', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
        quantile10=('value_covid', lambda x: np.quantile(x, 0.1)),         # Median (50th percentile)
        quantile90=('value_covid', lambda x: np.quantile(x, 0.9))
        ).reset_index()
        
    ax1.plot(summary_covid["lag"], summary_covid["mean"], label="%s"%signal, alpha=1)
    ax1.fill_between(summary_covid["lag"], 
                      summary_covid["quantile10"], 
                      summary_covid["quantile90"], alpha=0.25)
    ax1.set_xlabel("Lag (Days)", fontsize=30)
    ax1.set_ylabel("%Reported", fontsize=30)
    # plt.legend(loc="lower left")
    ax1.set_yticks(np.arange(0, 111, 20))
    ax1.set_xticks(np.append(np.arange(0, 121, 30), [7, 14]))
    ax1.set_title("COVID-19 Counts in Massachusetts", fontsize=32, loc="left")
    ax1.set_ylim(0, 120)
    ax1.set_xlim(-1, 181)
    ax1.tick_params(axis='both', labelsize=25)
    ax1.grid(True)
    
    
    ax2 = plt.subplot(1, 2, 2)
    df_frc = create_pivot_table_for_heatmap(subdf, "7dav_frac", melt=True).dropna()  
    # df_frc = df_frc.loc[df_frc["lag"] <= 125]
    summary_frc = df_frc.groupby(["lag"]).agg(
        mean=('7dav_frac', 'mean'),
        sem=('7dav_frac', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
        quantile10=('7dav_frac', lambda x: np.quantile(x, 0.1)),         # Median (50th percentile)
        quantile90=('7dav_frac', lambda x: np.quantile(x, 0.9))
        ).reset_index()
    
    ax2.plot(summary_frc["lag"], summary_frc["mean"], label="%s"%signal, alpha=1)
    ax2.fill_between(summary_frc["lag"], 
                     summary_frc["quantile10"], 
                     summary_frc["quantile90"], alpha=0.25)
    plt.xlabel("Lag (Days)", fontsize=30)
    plt.ylabel("Deviation from\nFinalized Value", fontsize=30)
    plt.title("COVID-19 Fractions in Massachusetts", fontsize=32, loc="left")
    yticks = np.arange(50, 130, 10) / 100
    plt.yticks(yticks, list(map(ratio_to_deviation, yticks)), fontsize=25)
    plt.ylim(0.48, 1.22)
    plt.xticks(np.append(np.arange(0, 121, 30), [7, 14]), fontsize=25)
    plt.xlim(-1, 181)
    
    ax2.grid(True)


ax1.axvspan(0, 7, alpha=0.25, color='gray')
ax2.axvspan(0, 7, alpha=0.25, color='gray')    
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

plt.savefig(fig_dir + "intro.png", bbox_inches = "tight")   

# Zoom in
    
plt.figure(figsize = (10, 6))
for signal in ['Insurance claims', 'Antigen tests', 'COVID-19 cases']:
    print(signal)
    subdf = dfs[signal].loc[(dfs[signal]["lag"] <= 180)
                         & (dfs[signal]["geo_value"] == "ma")]
    subdf.index = list(range(subdf.shape[0]))
    
    df_covid = create_pivot_table_for_heatmap(subdf, "7dav_frac", melt=True).dropna()  
    df_covid = df_covid.loc[df_covid["lag"] <= 8]
    
    mean_df = df_covid.groupby("lag").mean().reset_index()
    q10_df = df_covid.groupby("lag").quantile(0.1).reset_index()
    q90_df = df_covid.groupby("lag").quantile(0.9).reset_index()
    
    plt.plot(mean_df["lag"], mean_df["7dav_frac"], label="%s"%signal, alpha=1)
    # plt.plot(mean_df["lag"], mean_df["completeness_total"], color="blue", label="Total")
    plt.fill_between(mean_df["lag"], 
                      q10_df["7dav_frac"], 
                      q90_df["7dav_frac"], alpha=0.25)
    
    plt.yticks(np.arange(0.4, 1.3, 0.2), list(map(ratio_to_deviation, np.arange(0.4, 1.3, 0.2))), fontsize=30)
    plt.ylim(0.38, 1.23)
    plt.xticks(np.arange(0, 8, 1), fontsize=30)

    plt.grid(True)
plt.savefig(fig_dir + "intro_zoomin1.png", bbox_inches = "tight") 

plt.figure(figsize = (10, 6))
for signal in ['Insurance claims', 'Antigen tests', 'COVID-19 cases']:
    print(signal)
    subdf = dfs[signal].loc[(dfs[signal]["lag"] <= 180)
                         & (dfs[signal]["geo_value"] == "ma")]
    subdf.index = list(range(subdf.shape[0]))
    
    df_covid = create_pivot_table_for_heatmap(subdf, "value_covid", melt=True).dropna()  
    df_covid = df_covid.loc[df_covid["lag"] <= 8]
    df_covid["value_covid"] = df_covid["value_covid"]*100
    mean_df = df_covid.groupby("lag").mean().reset_index()
    q10_df = df_covid.groupby("lag").quantile(0.1).reset_index()
    q90_df = df_covid.groupby("lag").quantile(0.9).reset_index()
    
    plt.plot(mean_df["lag"], mean_df["value_covid"], label="%s"%signal, alpha=1)
    plt.fill_between(mean_df["lag"], 
                      q10_df["value_covid"], 
                      q90_df["value_covid"], alpha=0.25)

    plt.yticks(fontsize=30)
    plt.xticks(np.arange(0, 8, 1), fontsize=30)

    plt.grid(True)
plt.savefig(fig_dir + "intro_zoomin2.png", bbox_inches = "tight") 


####################################
### Fig 3: Backfill Pattern: Heat map and Lineplot
####################################


combined = dfs["Insurance claims"].copy()
# combined = dfs["Quidel"].copy()
statename = "Massachusetts"
# fig.colorbar(c1, ax=axs.ravel().tolist(), orientation='vertical', label='Color bar')

for state in ['ma']:
    subdf = combined.loc[(combined["geo_value"] == state)
                         & (combined["reference_date"] >= datetime(2021, 6, 1))
                         & (combined["lag"] <= 90)]
    subdf.index = list(range(subdf.shape[0]))
    
    start_date = datetime(2021, 6, 1)
    end_date = datetime(2023, 3, 11)
    n_days = (end_date - start_date).days + 1 
    time_index = np.array([(start_date + timedelta(i)).date() for i in range(n_days)])
    
    selected_xtix = [x.day == 1 for x in time_index] # Show Sundays on y_axis
    
    df_covid = create_pivot_table_for_heatmap(subdf, "value_covid")
    df_total = create_pivot_table_for_heatmap(subdf, "value_total")
    
    fig = plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    ax = sns.heatmap(df_covid.values*100,cmap="tab20c_r", xticklabels=True, cbar=False)
    ytix = ax.get_yticks()
    plt.ylabel("Lag (Days)", fontsize = 30)
    plt.xlabel("Reference Date", fontsize = 30)
    plt.title("COVID-19 Claims in %s"%statename, 
              fontsize = 35, loc="left")
    plt.yticks(fontsize=15)
    ax.set_xticks(np.arange(0.5, n_days + 0.5)[selected_xtix][::2])
    ax.set_xticklabels(time_index[selected_xtix][::2], rotation=45, ha="right")
    ax.set_yticks(np.arange(0.5, 91.5, 30))
    ax.tick_params(axis='both', labelsize=25)
    plt.axhline(30, linestyle="--")
    plt.axhline(60, linestyle="--")
    # plt.axhline(90, linestyle="--")
    plt.axhline(90, linestyle="--")
    ax.invert_yaxis()

    
    plt.subplot(1, 2, 2)
    ax = sns.heatmap(df_total.values*100,cmap="tab20c_r", xticklabels=True)
    cbar = ax.collections[0].colorbar
    # here set the labelsize by 20
    cbar.ax.tick_params(labelsize=25)
    cbar.ax.set_ylabel('%Reported', rotation=270, labelpad=16, fontsize=35)
    cbar.ax.yaxis.set_label_position('right')
    ytix = ax.get_yticks()
    plt.ylabel("Lag (Days)", fontsize = 30)
    plt.xlabel("Reference Date", fontsize = 30)
    plt.title("Total Claims in %s"%statename, 
              fontsize = 35, loc="left")
    plt.yticks(fontsize=15)
    ax.set_xticks(np.arange(0.5, n_days + 0.5)[selected_xtix][::2])
    ax.set_xticklabels(time_index[selected_xtix][::2], rotation=45, ha="right")
    ax.set_yticks(np.arange(0.5, 91.5, 30))
    ax.tick_params(axis='both', labelsize=25)
    plt.axhline(30, linestyle="--")
    plt.axhline(60, linestyle="--")
    # plt.axhline(90, linestyle="--")
    plt.axhline(90, linestyle="--")
    ax.invert_yaxis()
    # 
    # fig.colorbar(c1, ax=axs.ravel().tolist(), orientation='vertical', label='Color bar')

    plt.tight_layout()
plt.savefig(fig_dir + "completeness_%s.pdf"%statename, bbox_inches = "tight")



####################################
### Fig 4: lineplot
####################################

df = dfs["Insurance claims"].copy()    

df = gmpr.add_geocode(df, from_code="state_id", new_code="state_code",
                      from_col="geo_value", new_col="state_code")
df = gmpr.add_geocode(df, from_code="state_code", new_code="state_name",
                      from_col="state_code", new_col="state_name")
df = gmpr.add_geocode(df, from_code="state_code", new_code="hhs",
                      from_col="state_code", new_col="hhs")

combined = df.loc[df["hhs"].isin(["2", "1"])]

fig = plt.figure(figsize = (20, 6))
for state, state_name in zip(combined["geo_value"].unique(), combined["state_name"].unique()):
    subdf = combined.loc[(combined["geo_value"] == state)
                         & (combined["reference_date"] >= datetime(2021, 6, 1))
                         & (combined["lag"] <= 180)]
    subdf.index = list(range(subdf.shape[0]))
    
    start_date = subdf["reference_date"].min()
    end_date = subdf["reference_date"].max()
    n_days = (end_date - start_date).days + 1 
    time_index = np.array([(start_date + timedelta(i)).date() for i in range(n_days)])
    
    selected_ytix = [x.day == 1 for x in time_index] # Show Sundays on y_axis
    
    df_covid = create_pivot_table_for_heatmap(subdf, "value_covid", melt=True).dropna()
    df_total = create_pivot_table_for_heatmap(subdf, "value_total", melt=True).dropna()
    
    
    ax1 = plt.subplot(1, 2, 1)
    summary_covid = df_covid.groupby(["lag"]).agg(
        mean=('value_covid', 'mean'),
        sem=('value_covid', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
        quantile10=('value_covid', lambda x: np.quantile(x, 0.1)),         # Median (50th percentile)
        quantile90=('value_covid', lambda x: np.quantile(x, 0.9))
        ).reset_index()

    
    plt.plot(summary_covid ["lag"], summary_covid ["mean"]*100, label="%s"%state_name.title(), alpha=0.8)
    plt.fill_between(summary_covid ["lag"], 
                      summary_covid ["quantile10"]*100, 
                      summary_covid ["quantile90"]*100, alpha=0.1)
    plt.xlabel("Lag (Days)", fontsize=30)
    plt.ylabel("%Reported", fontsize=30)
    plt.title("COVID-19 Claims in HHS Region 1&2", fontsize=35, loc="left")
    plt.yticks(np.arange(0, 110, 10), fontsize=25)
    plt.xticks(np.arange(0, 181, 30), fontsize=25)
    plt.ylim(0, 105)
    plt.grid(True)
    
    ax2 = plt.subplot(1, 2, 2)
    summary_total = df_total.groupby(["lag"]).agg(
        mean=('value_total', 'mean'),
        sem=('value_total', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
        quantile10=('value_total', lambda x: np.quantile(x, 0.1)),         # Median (50th percentile)
        quantile90=('value_total', lambda x: np.quantile(x, 0.9))
        ).reset_index()
    
    plt.plot(summary_total["lag"], summary_total["mean"]*100, label="%s"%state_name.title(), alpha=0.8)
    # plt.plot(mean_df["lag"], mean_df["completeness_total"], color="blue", label="Total")
    plt.fill_between(summary_total["lag"], 
                      summary_total["quantile10"]*100, 
                      summary_total["quantile90"]*100, alpha=0.1)
    plt.xlabel("Lag (Days)", fontsize=30)
    plt.ylabel("%Reported", fontsize=30)
    plt.title("Total Claims in HHS Region 1&2", fontsize=35, loc="left")
    # plt.legend(loc="upper left")
    plt.yticks(np.arange(0, 110, 10), fontsize=25)
    plt.xticks(np.arange(0, 181, 30), fontsize=25)
    plt.ylim(0, 105)
    plt.grid(True)
    

    plt.tight_layout()

labels_handles = {
  label: handle for ax in fig.axes for handle, label in zip(*ax.get_legend_handles_labels())
}

fig.legend(
  labels_handles.values(),
  labels_handles.keys(),
  loc = "upper center",
  bbox_to_anchor = (0.5, 0),
  bbox_transform = plt.gcf().transFigure,
  ncol = 4,
  fontsize=25
)    

plt.savefig(fig_dir + "completeness_lineplot_hhs1&2.png", bbox_inches = "tight")    



############ Zoom in

fig = plt.figure(figsize=(10, 4))
for state, state_name in zip(combined["geo_value"].unique(), combined["state_name"].unique()):
    subdf = combined.loc[(combined["geo_value"] == state)
                         & (combined["reference_date"] >= datetime(2021, 6, 1))
                         & (combined["lag"] <= 180)]
    subdf.index = list(range(subdf.shape[0]))
    
    start_date = subdf["reference_date"].min()
    end_date = subdf["reference_date"].max()
    n_days = (end_date - start_date).days + 1 
    time_index = np.array([(start_date + timedelta(i)).date() for i in range(n_days)])
    
    selected_ytix = [x.day == 1 for x in time_index] # Show Sundays on y_axis
    
    df_covid = create_pivot_table_for_heatmap(subdf, "value_covid", melt=True).dropna()

    summary_covid = df_covid.groupby(["lag"]).agg(
        mean=('value_covid', 'mean'),
        sem=('value_covid', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
        quantile10=('value_covid', lambda x: np.quantile(x, 0.1)),         # Median (50th percentile)
        quantile90=('value_covid', lambda x: np.quantile(x, 0.9))
        ).reset_index()

    
    plt.plot(summary_covid ["lag"], summary_covid ["mean"]*100, label="%s"%state_name.title(), alpha=0.8)
    plt.fill_between(summary_covid ["lag"], 
                      summary_covid ["quantile10"]*100, 
                      summary_covid ["quantile90"]*100, alpha=0.1)
    plt.yticks(np.arange(70, 110, 10), fontsize=55)
    plt.xticks(np.arange(0, 91, 30), fontsize=55)
    plt.ylim(68, 105)
    plt.xlim(0, 95)
    plt.grid(True)
    
    
    
fig = plt.figure(figsize=(10, 4))
for state, state_name in zip(combined["geo_value"].unique(), combined["state_name"].unique()):
    subdf = combined.loc[(combined["geo_value"] == state)
                         & (combined["reference_date"] >= datetime(2021, 6, 1))
                         & (combined["lag"] <= 180)]
    subdf.index = list(range(subdf.shape[0]))
    
    start_date = subdf["reference_date"].min()
    end_date = subdf["reference_date"].max()
    n_days = (end_date - start_date).days + 1 
    time_index = np.array([(start_date + timedelta(i)).date() for i in range(n_days)])
    
    selected_ytix = [x.day == 1 for x in time_index] # Show Sundays on y_axis
    
    df_total = create_pivot_table_for_heatmap(subdf, "value_total", melt=True).dropna()

    summary_total = df_total.groupby(["lag"]).agg(
        mean=('value_total', 'mean'),
        sem=('value_total', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
        quantile10=('value_total', lambda x: np.quantile(x, 0.1)),         # Median (50th percentile)
        quantile90=('value_total', lambda x: np.quantile(x, 0.9))
        ).reset_index()

    plt.plot(summary_total["lag"], summary_total["mean"]*100, label="%s"%state_name.title(), alpha=0.8)
    plt.fill_between(summary_total["lag"], 
                      summary_total["quantile10"]*100, 
                      summary_total["quantile90"]*100, alpha=0.1)
    plt.yticks(np.arange(70, 110, 10), fontsize=55)
    plt.xticks(np.arange(0, 91, 30), fontsize=55)
    plt.ylim(68, 105)
    plt.xlim(0, 95)
    plt.grid(True)
