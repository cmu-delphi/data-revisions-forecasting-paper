#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 12:53:33 2025

@author: jingjingtang
"""

# Import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from delphi_epidata import Epidata
from epiweeks import Week

Epidata.auth = ('epidata', "55f506d86b0bc")
# Fetch data
pdList = []
for lag in range(0, 105):
    print(lag)
    res = Epidata.fluview(['nat'], [Epidata.range(201101, 201752)], lag=lag)
    subdf = pd.DataFrame(res["epidata"])
    pdList.append(subdf)
df = pd.concat(pdList)
df.index = list(range(0, df.shape[0]))
# enddates are saturdays
df["issue_date"] = df['issue'].apply(lambda x: Week.fromstring(str(x)).enddate())
df["time_value"] = df['epiweek'].apply(lambda x: Week.fromstring(str(x)).enddate())
df["lag"] = (df["issue_date"] - df["time_value"]).dt.days

# Keep only rows where `issue_date` is max within each `time_value` group
df_newest = df.loc[df.groupby('time_value')['lag'].idxmax()]
df_initial = df.loc[df.groupby('time_value')['lag'].idxmin()]

plt.figure(figsize=(15, 4))
plt.plot(df_newest["time_value"], df_newest["num_ili"])
plt.plot(df_initial["time_value"], df_initial["num_ili"])

df.to_csv("num_ili_national.csv", index=False)