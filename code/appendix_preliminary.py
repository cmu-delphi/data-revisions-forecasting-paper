#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Figs for preliminary in Appendix

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

dfs_raw = {}
dfs_raw["CHNG Outpatient"] = read_chng_outpatient()
dfs_raw["MA-DPH"] = read_ma_dph()
dfs_raw["Quidel"] = read_quidel() 

####################################
### Fig 5: heatmap of largest L
####################################
date1 = datetime(2021, 8, 1)
date2 = datetime(2022, 1, 1)
date3 = datetime(2022, 6, 1)

plt.figure(figsize=(18, 8))
plt.subplot(2, 1, 1)
# Find the smallest lag where the revision of y_it is finalized
maxL_df = df.sort_values(["geo_value", "time_value", "value_covid", "lag"], 
                         ascending=[True, True, False, True])
maxL_df = maxL_df.groupby(["geo_value", "time_value"]).first().reset_index()
# Sanity Check
# subdf = maxL_df.loc[maxL_df["geo_value"] == "ca"]
# plt.plot(subdf["time_value"], subdf["value_covid"])


subdf = maxL_df.loc[maxL_df["time_value"].isin([date1, date2, date3])].sort_values(["time_value", "lag"])
subdf["Reference Date"] = subdf["time_value"].dt.date.astype(str)
subdf["State"] = subdf["geo_value"].apply(lambda x: x.upper())
sns.barplot(subdf, x="State", y="lag", hue="Reference Date")
plt.grid(True)
plt.xticks(rotation=90, fontsize=10)
plt.ylabel(r'$L_{it}$', fontsize=15)
plt.xlabel("State", fontsize=15)
plt.title("COVID-19 Claims", fontsize=20)

plt.subplot(2, 1, 2)
# Find the smallest lag where the revision of y_it is finalized
maxL_df = df.sort_values(["geo_value", "time_value", "value_total", "lag"], 
                         ascending=[True, True, False, True])
maxL_df = maxL_df.groupby(["geo_value", "time_value"]).first().reset_index()
# Sanity Check
# subdf = maxL_df.loc[maxL_df["geo_value"] == "ca"]
# plt.plot(subdf["time_value"], subdf["value_covid"])

subdf = maxL_df.loc[maxL_df["time_value"].isin([date1, date2, date3])].sort_values(["time_value", "lag"])
subdf["Reference Date"] = subdf["time_value"].dt.date.astype(str)
subdf["State"] = subdf["geo_value"].apply(lambda x: x.upper())
sns.barplot(subdf, x="State", y="lag", hue="Reference Date")
plt.grid(True)
plt.xticks(rotation=90, fontsize=10)
plt.ylabel(r'$L_{it}$', fontsize=15)
plt.xlabel("State", fontsize=15)
plt.title("Total Claims", fontsize=20)
plt.tight_layout()
plt.savefig("/Users/jingjingtang/Downloads/figs/Lit_examples.pdf", bbox_inches = 'tight')



