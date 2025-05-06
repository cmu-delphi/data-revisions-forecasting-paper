#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregated evl results by reference date per location

@author: jingjingtang
"""

import os
from datetime import datetime
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

### Read results
dfs = {}
dfs["COVID-19 cases in MA"] = read_experimental_results(madph_config, "DelphiRF")
dfs["CHNG Outpatient Count"] = read_experimental_results(chng_count_config, "DelphiRF")
dfs["Insurance claims"] = read_experimental_results(chng_fraction_config, "DelphiRF")
dfs["Antigen tests"] = read_experimental_results(quidel_config, "DelphiRF")


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

colors = {
    180: "tab:orange",
    365: "tab:green"}

plt.style.use('default')


for signal in ["COVID-19 cases in MA", "CHNG Outpatient Count",
                    "Insurance claims", "Antigen tests"]:  
    if signal == "COVID-19 cases in MA":
        yticks = [0, 5, 10]
        ylims = [0, 0.1]
        test_w = 7
    else:
        yticks = [0, 20, 50]
        ylims = [0, 0.5]
        test_w = 30
    for state in dfs[signal]["geo_value"].unique():
        ### generate fig for one signal
        plt.figure(figsize=(30, 8))
        yticks = [0, 20, 50]
        ylims = [0, 0.5]
        signal = "Insurance claims"
        fig, ax = plt.subplots(2, 1, figsize=(30, 8), 
                               gridspec_kw={'height_ratios': [2.6, 0.8], 'hspace': 0.2})
        delphi_result = dfs[signal].loc[(dfs[signal]["lag"] == 7)
                                        & (dfs[signal]["geo_value"] == state)].groupby(["tw", "reference_date"]).agg(
            mean=('wis', 'mean'),
            sem=('wis', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
            quantile10=('wis', lambda x: np.quantile(x, 0.1)),         # Median (50th percentile)
            quantile90=('wis', lambda x: np.quantile(x, 0.9))
            ).reset_index()
        for tw in [180, 365]:
            delphi_subdf = delphi_result.loc[delphi_result["tw"] == tw].sort_values("reference_date")
            ax[0].plot(delphi_subdf["reference_date"], delphi_subdf["mean"], color =  colors[tw], alpha=0.8,
                           label="Training window: %d days"%(tw), linewidth = 5)
            ax[0].fill_between(
                delphi_subdf["reference_date"],
                delphi_subdf["quantile10"],
                delphi_subdf["quantile90"],
                alpha=0.2, color=colors[tw])  
        ax[0].grid(True)
        ax[0].set_ylabel("WIS" ,fontsize=50)
        ax[0].tick_params(axis='both', labelsize=45)
        ax[0].set_ylim(ylims)
        ax[0].set_xlim([start_date, end_date])
        ax[0].set_title("%s, Lag = 7, %s \n(Model retrained every %d days)"%(titles[signal], state.upper(), test_w), fontsize=65)    
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
        xticks = df_full[df_full["reference_date"].dt.day == 1].index.tolist()
        ax[1].text(1.01, 0.1, subtitles[signal], 
                    fontsize=45, va='center', ha='left', rotation=0, transform=ax[1].transAxes)    
        
        # sns.heatmap(np.exp(df_full[[target_cols[signal]]].T), cmap='Reds', cbar=False, ax=ax[1], 
        #             xticklabels=False, yticklabels=False)
        # ax[1].text(1, 0.5, "%Positive COVID-19\nAntigen Tests\n(At Target Lag)", 
        #             fontsize=35, va='center', ha='left', rotation=0, transform=ax[1].transAxes)
        # Apply the same x-ticks and labels to ax[1]
        xticks = df_full[df_full["reference_date"].dt.day == 1].index.tolist()
        ax[1].set_xlabel('Reference Date', fontsize=50)
        ax[1].set_xticks(xticks)
        ax[1].set_xticklabels(df_full['reference_date'].dt.date[xticks], fontsize=40, rotation=45, ha='right')
        plt.savefig(fig_dir + "appendix_%s_experiment_result_%s_time_series.pdf"%(state, "_".join(signal.split(" "))), bbox_inches = 'tight')
    
                