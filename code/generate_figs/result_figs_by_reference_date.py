#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregated evl results over reference dates

@author: jingjingtang
"""

from datetime import datetime
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
dfs["Insurance claims"] = read_chng_outpatient_result()
dfs["COVID-19 cases in MA"] = read_ma_dph_result()
dfs["Antigen tests"] = read_quidel_result()
dfs["CHNG Outpatient Count"] = read_chng_outpatient_count_result()

start_date = datetime(2021, 5, 15)
end_date = datetime(2022, 3, 1)

titles = {
    'COVID-19 cases in MA': 'COVID-19 cases in MA',
    'CHNG Outpatient Count': 'Insurance claims',
    "Insurance claims" : "Insurance claims",
    "Antigen tests": "Antigen tests"
}
target_cols = {
    'COVID-19 cases in MA': 'log_value_target',
    'CHNG Outpatient Count': 'log_value_target',
    "Insurance claims" : "log_value_target_7dav",
    "Antigen tests": "log_value_target_7dav"
}
subtitles = {
    'COVID-19 cases in MA': '#COVID-19 Cases\n(At Target Lag)',
    'CHNG Outpatient Count': '#COVID-19\nInsurance Claims\n(At Target Lag)',
    "Insurance claims" : "%COVID-19\nInsurance Claims\n(At Target Lag)",
    "Antigen tests": " %Positive COVID-19\nAntigen Tests\n(At Target Lag)"   
    }
plt.style.use('default')


yticks = [0, 20, 50]
ylims = [0, 0.5]
signal = "Insurance claims"
fig, ax = plt.subplots(2, 1, figsize=(30, 8), 
                       gridspec_kw={'height_ratios': [2.6, 0.8], 'hspace': 0.2})
delphi_result = dfs[signal].loc[dfs[signal]["lag"] == 7].groupby(["tw", "reference_date"]).agg(
    mean=('wis', 'mean'),
    sem=('wis', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
    quantile10=('wis', lambda x: np.quantile(x, 0.1)),         # Median (50th percentile)
    quantile90=('wis', lambda x: np.quantile(x, 0.9))
    ).reset_index()
delphi_subdf = delphi_result.loc[delphi_result["tw"] == 180].sort_values("reference_date")
ax[0].plot(delphi_subdf["reference_date"], delphi_subdf["mean"], color = "tab:orange", alpha=0.8,
               label="Forecast Accuracy at Lag 7", linewidth = 5)
ax[0].fill_between(
    delphi_subdf["reference_date"],
    delphi_subdf["quantile10"],
    delphi_subdf["quantile90"],
    alpha=0.2, color="tab:orange")     
ax[0].grid(True)
ax[0].set_ylabel("WIS" ,fontsize=50)
ax[0].tick_params(axis='both', labelsize=45)
ax[0].set_ylim(ylims)
ax[0].set_xlim([start_date, end_date])
ax[0].set_title(titles[signal], fontsize=55)
ax[0].legend(fontsize=50)

ax2 = ax[0].twinx()

ax2.plot([],[])
ax2.set_yticks(list(map(re_to_wis, yticks)))
ax2.set_yticklabels([str(x)+"%" for x in yticks])
ax2.set_ylim((ax[0].get_ylim()[0], ax[0].get_ylim()[1]))
ax2.tick_params(axis='both', labelsize=45)  
ax2.set_ylabel('Absolute\nRelative Error', fontsize=50, rotation=270, labelpad=100)
# Remove x-axis labels from the first plot to avoid clutter
ax[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

target_df = dfs[signal].loc[(dfs[signal]["tw"] == 180)
                            & (dfs[signal]["lag"] == 7)
                            ].groupby("reference_date").mean().reset_index()
full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')
df_full = pd.DataFrame({'reference_date': full_date_range})
df_full = pd.merge(df_full,  target_df, on='reference_date', how='left')
sns.heatmap(np.exp(df_full[[target_cols[signal]]].T), cmap='Reds', cbar=False, ax=ax[1], 
            xticklabels=False, yticklabels=False)
ax[1].text(1, 0.5, "%Positive COVID-19\nAntigen Tests\n(At Target Lag)", 
            fontsize=35, va='center', ha='left', rotation=0, transform=ax[1].transAxes)
# Apply the same x-ticks and labels to ax[1]
xticks = df_full[df_full["reference_date"].dt.day == 1].index.tolist()
ax[1].set_xlabel('Reference Date', fontsize=50)
ax[1].set_xticks(xticks)
ax[1].set_xticklabels(df_full['reference_date'].dt.date[xticks], fontsize=40, rotation=45, ha='right')
plt.savefig(fig_dir + "experiment_count_result_%s_time_series.pdf"%"_".join(signal.split(" ")), bbox_inches = 'tight')

### Combined
colors = {
    180: "tab:orange",
    365: "tab:green"}

fig, ax = plt.subplots(4, 1, figsize=(30, 16), 
                       gridspec_kw={'height_ratios': [2.6, 0.8, 2.6, 0.8], 'hspace':0.1})
# for idx, signal in enumerate(["COVID-19 cases in MA", "CHNG Outpatient Count"]):
for idx, signal in enumerate(["Insurance claims", "Antigen tests"]):    
    delphi_result = dfs[signal].loc[dfs[signal]["lag"] == 7].groupby(["tw", "reference_date"]).agg(
        mean=('wis', 'mean'),
        sem=('wis', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
        quantile10=('wis', lambda x: np.quantile(x, 0.1)),         # Median (50th percentile)
        quantile90=('wis', lambda x: np.quantile(x, 0.9))
        ).reset_index()
    
    if signal == "COVID-19 cases in MA":
        yticks = [0, 5, 10]
        ylims = [0, 0.1]
        test_w = 7
    else:
        yticks = [0, 20, 50]
        ylims = [0, 0.5]
        test_w = 30
       
    for tw in [180, 365]:
        delphi_subdf = delphi_result.loc[delphi_result["tw"] == tw].sort_values("reference_date")
        
    
        ax[idx*2].plot(delphi_subdf["reference_date"], delphi_subdf["mean"], color =  colors[tw], alpha=0.8,
                       label="Training window: %d days"%(tw), linewidth = 5)
        ax[idx*2].fill_between(
            delphi_subdf["reference_date"],
            delphi_subdf["quantile10"],
            delphi_subdf["quantile90"],
            alpha=0.2, color=colors[tw])     
    ax[idx*2].grid(True)
    ax[idx*2].set_ylabel("WIS" ,fontsize=60)
    ax[idx*2].tick_params(axis='both', labelsize=55)
    ax[idx*2].set_ylim(ylims)
    ax[idx*2].set_xlim([start_date, end_date])
    ax[idx*2].set_title("%s, Lag = 7 \n(Model retrained every %d days)"%(titles[signal], test_w), fontsize=70)
    ax[idx*2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        
    ax2 = ax[idx*2].twinx()
    ax2.plot([],[])
    ax2.set_yticks(list(map(re_to_wis, yticks)))
    ax2.set_yticklabels([str(x)+"%" for x in yticks])
    ax2.set_ylim((ax[idx*2].get_ylim()[0], ax[idx*2].get_ylim()[1]))
    ax2.tick_params(axis='both', labelsize=55)  
    ax2.set_ylabel('Absolute\nRelative\nError', fontsize=60, rotation=270, labelpad=200)
    
    target_df = dfs[signal].loc[(dfs[signal]["tw"] == 180)
                                & (dfs[signal]["lag"] == 7)
                                ].groupby("reference_date").mean().reset_index()
    full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    df_full = pd.DataFrame({'reference_date': full_date_range})
    df_full = pd.merge(df_full,  target_df, on='reference_date', how='left')
    sns.heatmap(np.exp(df_full[[target_cols[signal]]].T), cmap='Reds', cbar=False, ax=ax[idx*2+1], 
                xticklabels=False, yticklabels=False)
    xticks = df_full[df_full["reference_date"].dt.day == 1].index.tolist()
    ax[idx*2+1].text(1.05, 0.08, subtitles[signal], 
                fontsize=55, va='center', ha='left', rotation=0, transform=ax[idx*2+1].transAxes)    
        
ax[idx*2+1].set_xlabel('Reference Date', fontsize=60)
ax[idx*2+1].set_xticks(xticks)
ax[idx*2+1].set_xticklabels(df_full['reference_date'].dt.date[xticks], fontsize=50, rotation=45, ha='right')
plt.tight_layout()
plt.suptitle("Fraction Forecasting", fontsize=90, y=1.12)
labels_handles = {
  label: handle for ax in fig.axes for handle, label in zip(*ax.get_legend_handles_labels())
}

fig.legend(
  labels_handles.values(),
  labels_handles.keys(),
  loc = "upper center",
  bbox_to_anchor = (0.5, -0.35),
  bbox_transform = plt.gcf().transFigure,
  ncol = 2,
  fontsize=55
)
# Adjust only the space between the second and third subplot
fig.subplots_adjust(hspace=0.5)  # This applies to all, but we will override it next
ax[2].set_position([ax[2].get_position().x0, ax[2].get_position().y0 - 0.15, 
                    ax[2].get_position().width, ax[2].get_position().height])

ax[3].set_position([ax[3].get_position().x0, ax[3].get_position().y0 - 0.20, 
                    ax[3].get_position().width, ax[3].get_position().height])


plt.savefig(fig_dir + "experiment_fraction_result_time_series.pdf", bbox_inches = 'tight')
