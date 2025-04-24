#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 14:28:27 2025

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


nobBS_dfs={}
nobBS_dfs["CHNG Outpatient Count"] = pd.read_csv("/Users/jingjingtang/Documents/backfill-recheck/nobBS_result/chng_outpatient_covid/combined_result.csv",
                           parse_dates=["onset_date", "issue_date"])
nobBS_dfs["CHNG Outpatient Count"]["lag"] = [(x-y).days for x, y in zip(nobBS_dfs["CHNG Outpatient Count"]["issue_date"], nobBS_dfs["CHNG Outpatient Count"]["onset_date"])]
nobBS_dfs["CHNG Outpatient Count"]["time_value"] = nobBS_dfs["CHNG Outpatient Count"]["onset_date"] 
nobBS_dfs["CHNG Outpatient Count"]["wis"] = abs(np.log(nobBS_dfs["CHNG Outpatient Count"]["estimate"] + 1) - np.log(nobBS_dfs["CHNG Outpatient Count"]["value_target"] + 1))

nobBS_dfs["COVID-19 cases"] = pd.read_csv("/Users/jingjingtang/Documents/backfill-recheck/nobBS_result/ma-dph/ma_daily.csv",
                           parse_dates=["onset_date", "issue_date"])
nobBS_dfs["COVID-19 cases"]["lag"] = [(x-y).days for x, y in zip(nobBS_dfs["COVID-19 cases"]["issue_date"], nobBS_dfs["COVID-19 cases"]["onset_date"])]
# nobBS_dfs["COVID-19 cases"]["time_value"] = nobBS_dfs["CHNG Outpatient Count"]["onset_date"] 
# nobBS_dfs["COVID-19 cases"]["wis"] = abs(np.log(nobBS_dfs["COVID-19 cases"]["estimate"] + 1) - np.log(nobBS_dfs["COVID-19 cases"]["value_target"] + 1))
nobBS_dfs["COVID-19 cases"] = nobBS_dfs["COVID-19 cases"].loc[nobBS_dfs["COVID-19 cases"]["lag"] >= 1]

nobBS_dfs["dengue"] = pd.read_csv("/Users/jingjingtang/Documents/backfill-recheck/nobBS_result/dengue/combined_result_weekly.csv", parse_dates=["issue_date", "onset_date"])
nobBS_dfs["dengue"]["lag"] = [(x-y).days for x, y in zip(nobBS_dfs["dengue"]["issue_date"], nobBS_dfs["dengue"]["onset_date"])]
nobBS_dfs["dengue"]["time_value"] = nobBS_dfs["dengue"]["onset_date"] 

nobBS_dfs["ilinat"]= pd.read_csv("/Users/jingjingtang/Documents/backfill-recheck/nobBS_result/ilicases/ilinat_weekly.csv", parse_dates=["issue_date", "onset_date"])
nobBS_dfs["ilinat"]["lag"] = [(x-y).days for x, y in zip(nobBS_dfs["ilinat"]["issue_date"], nobBS_dfs["ilinat"]["onset_date"])]
nobBS_dfs["ilinat"]["time_value"] = nobBS_dfs["ilinat"]["onset_date"] 


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
epinowcast_dfs["ilinat"]["issue_date"] = epinowcast_dfs["ilinat"]["report_date"] 
#############################
# Plot for runtime comparison
#############################
preprocessing_df = {}
input_dir = "/Users/jingjingtang/Documents/backfill-recheck/paper/data/raw/chng_outpatient_input/pre-processed-state-lag60/"
pdList = []
for fn in os.listdir(input_dir):
    if "_daily" in fn:
        subdf = pd.read_csv(input_dir + fn, parse_dates=["time_value", "issue_date"])
        subdf["geo_value"] = fn[:2]
        pdList.append(subdf)
preprocessing_df["CHNG Outpatient Count"] = pd.concat(pdList)[["issue_date", "geo_value", "elapsed_time"]].drop_duplicates()
# preprocessing_df["COVID-19 cases"] = preprocessing_df["CHNG Outpatient Count"]
# preprocessing_df["dengue"] = preprocessing_df["CHNG Outpatient Count"]

rt_summary = pd.DataFrame(columns = ["method", "data", "mean", "std"])
idx = 0
for signal in ["COVID-19 cases", "CHNG Outpatient Count", "dengue", "ilinat"]:
    epinowcast_df = epinowcast_dfs[signal][["report_date", "geo_value", "elapsed_time"]].drop_duplicates()
    mean_elapsed_time = epinowcast_df["elapsed_time"].mean()
    std_elapsed_time = epinowcast_df["elapsed_time"].std(ddof=1)
    sem_elapsed_time = std_elapsed_time / np.sqrt(len(epinowcast_df))
    rt_summary.loc[idx] = ["epinowcast", signal, mean_elapsed_time, sem_elapsed_time]
    idx += 1

    nobBS_df = nobBS_dfs[signal][["issue_date", "geo_value", "elapsed_time"]].drop_duplicates()
    mean_elapsed_time = nobBS_df["elapsed_time"].mean()
    std_elapsed_time = nobBS_df["elapsed_time"].std(ddof=1)
    sem_elapsed_time = std_elapsed_time / np.sqrt(len(nobBS_df))
    rt_summary.loc[idx] = ["nobBS", signal, mean_elapsed_time, sem_elapsed_time]
    idx += 1

    delphi_df = dfs[signal].loc[(dfs[signal]["lag"].isin(
        list(range(0, 15)) + [21, 35, 51] 
        ))
        & (dfs[signal]["tw"] == 365)].groupby(["issue_date", "geo_value"])["elapsed_time"].sum().reset_index()
    if signal == "CHNG Outpatient Count":
        delphi_df = delphi_df.merge(
            preprocessing_df[signal], on=["issue_date", "geo_value"], suffixes=["_pre", "_run"])
        delphi_df["elapsed_time"] = delphi_df["elapsed_time_pre"] + delphi_df["elapsed_time_run"]
    elif signal == "COVID-19 cases":
        delphi_df["elapsed_time"] = delphi_df["elapsed_time"] + 17
    elif signal == "dengue":
        delphi_df["elapsed_time"] = delphi_df["elapsed_time"] + 0.41
    else:
        delphi_df["elapsed_time"] = delphi_df["elapsed_time"] + 0.31
    mean_elapsed_time = delphi_df["elapsed_time"].mean()
    std_elapsed_time = delphi_df["elapsed_time"].std(ddof=1)
    sem_elapsed_time = std_elapsed_time / np.sqrt(len(delphi_df))
    
    rt_summary.loc[idx] = [ "Delphi-Training", signal, mean_elapsed_time, sem_elapsed_time]
    idx += 1
rt_summary    

    
    
    