#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper Functions

@author: jingjingtang
"""
import os
from matplotlib import pyplot as plt
from datetime import datetime

import numpy as np
import pandas as pd
from constants import data_dir, filtered_states, MA_POP

## constants providing information for each data set

file_dir = "./data/"

madph_config = {
    "data_dir": file_dir,
    "indicator": "ma-dph",
    "log_value_col": "log_value_7dav",
    "log_target_col": "log_value_target",
    "start_date": datetime(2021, 7, 1),
    "end_date": datetime(2022, 6, 24)
    }

quidel_config = {
    "data_dir": file_dir,
    "indicator": "quidel",
    "log_value_col": "log_value_7dav",
    "log_target_col": "log_value_target_7dav",
    "start_date": datetime(2021, 5, 18),
    "end_date": datetime(2022, 12, 12)
    }

chng_count_config = {
    "data_dir": file_dir,
    "indicator": "chng_outpatient_count",
    "log_value_col": "log_value_7dav",
    "log_target_col": "log_value_target",
    "start_date": datetime(2021,6, 1),
    "end_date": datetime(2023, 1, 10)
    }

chng_fraction_config = {
    "data_dir": file_dir,
    "indicator": "chng_outpatient_fraction",
    "log_value_col": "log_value_7dav",
    "log_target_col": "log_value_target_7dav",
    "start_date": datetime(2021, 6, 1),
    "end_date": datetime(2023, 1, 10)
    }


dengue_config = {
    "data_dir": file_dir,
    "indicator": "dengue",
    "log_value_col": "log_value_7dav",
    "log_target_col": "log_value_target_7dav",
    "start_date": datetime(1991, 12, 23),
    "end_date": datetime(2010, 11,29)
    }

ilicases_config = {
    "data_dir": file_dir,
    "indicator": "ilicases",
    "log_value_col": "log_value_7dav",
    "log_target_col": "log_value_target_7dav",
    "start_date": datetime(2014, 6, 30),
    "end_date": datetime(2017, 9, 25)
    }


####################################
### Read Data ######################
####################################

# def read_chng_outpatient(): 
#     df = pd.read_csv("/Users/jingjingtang/Documents/backfill-recheck/paper/data/raw/chng_outpatient_input/CHNG_outpatient_state_combined_df_until20230218.csv",
#                      parse_dates=["time_value", "issue_date"])
#     df = df.loc[(df["time_value"] <= datetime(2023, 1, 10))
#                 &(df["time_value"] >= datetime(2021, 4, 1))
#                 & (~df["geo_value"].isin(set(filtered_states)))].sort_values(["issue_date", "time_value"])  
#     pdList = []
#     for geo in df["geo_value"].unique():
#         subdf = df[df["geo_value"] == geo]
#         subdf["value_7dav_covid"] = subdf["value_covid"].groupby(subdf["issue_date"]).shift(1).rolling(7).mean()
#         subdf["value_7dav_total"] = subdf["value_total"].groupby(subdf["issue_date"]).shift(1).rolling(7).mean()
#         pdList.append(subdf)
#     df = pd.concat(pdList)
    
#     df["7dav_frac"] = (df["value_7dav_covid"]+1)/(df["value_7dav_total"]+1)
#     return df
    
# df = read_chng_outpatient()

def read_ma_dph(data_dir, path):
    """
    Read the raw ma-dph data
    """
    file_dir = data_dir + path
    ma_df = pd.read_csv(file_dir, parse_dates=["test_date", "issue_date"])
    ma_df.rename({"confirmed_case_7d_avg": "value_covid", "test_date": "time_value"}, axis=1, inplace=True)
    ma_df["lag"] = [(x- y).days for x, y in zip(ma_df["issue_date"], ma_df["time_value"])]
    ma_df["geo_value"] = "ma"
    ma_df["7dav_frac"] = ma_df["value_covid"] / MA_POP    
    return ma_df

def read_quidel(data_dir, path):
    file_dir = data_dir + path
    df = pd.read_csv(file_dir, parse_dates=["time_value", "issue_date"])    
    df = df.loc[~df["geo_value"].isin(set(filtered_states))].sort_values(["issue_date", "time_value"])

    pdList = []
    for geo in df["geo_value"].unique():
        subdf = df[df["geo_value"] == geo]
        subdf["value_7dav_covid"] = subdf["value_covid"].groupby(subdf["issue_date"]).shift(1).rolling(7).mean()
        subdf["value_7dav_total"] = subdf["value_total"].groupby(subdf["issue_date"]).shift(1).rolling(7).mean()
        pdList.append(subdf)
    df = pd.concat(pdList)
    
    df["7dav_frac"] = (df["value_7dav_covid"]+1)/(df["value_7dav_total"]+1)
    
    return df


def read_experimental_results(config, method):
    """
    Read experimental results
    config contains information about:
        indicator, start_date, end_date,
        log_value_col and log_target_col (for baseline)
    """

    
    path = os.path.join(config["data_dir"], "results", "%s_result"%method, config["indicator"])
    pdList = []
    for fn in os.listdir(path):
        if ".csv" not in fn:
            continue
        # Get necessary info from the filename                 
        parts = fn.split("_")
        indicator = parts[0]
        geo = parts[1]
        temporal_resol = parts[2]
        training_window = int(parts[3].split("window")[1])
        testing_window = int(parts[4].split("window")[1])
        ref_lag = int(parts[5].split("lag")[1])
        
        if method == "DelphiRF":            
            report_date_col = "report_date"
            refd_col = "reference_date"
        elif method == "Epinowcast":
            refd_col = "reference_date"
            report_date_col = "report_date"
        elif method == "NobBS":
            refd_col = "onset_date"
            report_date_col = "issue_date"
        
        df = pd.read_csv(os.path.join(path, fn),
                         parse_dates=[report_date_col, refd_col]
                         ).sort_values([report_date_col, refd_col])   
        df["tw"] = training_window
        df["lag"] = [(x-y).days for x, y in zip(df[report_date_col], df[refd_col])]
        df = df.loc[(df[report_date_col] >= config["start_date"])
                    & (df[report_date_col] <= config["end_date"])]
        
        if indicator == "ilicases":
            df["geo_value"] = "nat"
        elif indicator == "madph":
           df["geo_value"] = "ma"
           df = df.loc[df["lag"]>0]
        elif indicator == "dengue":
            df["geo_value"] = "pr"

        
        
        for col in df.columns:
            if "predicted_tau" in col:
                df[col] = np.exp(df[col])
        
        if method == "DelphiRF":
            df["mae"] = abs(df[config["log_value_col"]] - df[config["log_target_col"]])
        
        pdList.append(df)

    return pd.concat(pdList)

### helper functions   
def wis_to_re(x):
    """
    Convert the WIS score to relative error
    re = exp(wis) - 1
    """
    y = (np.exp(x) -1 ) * 100
    re = str(y.round(decimals=2)) + "%"
    return re  

def re_to_wis(x):
    """
    Conver the relative error to the WIS score
    wis = np.log(re + 1)
    """
    y = np.log(x / 100 + 1)
    return y

def ratio_to_deviation(x):
    return "%+d"%(round(x-1, ndigits=2)*100) + "%"



### Data visualization for EAD
def create_pivot_table_for_heatmap(subdf, value_type, melt=False, target="newest"):
    """
    Create pivot table to clearly display the data revision process
    for a singla location

    Parameters
    ----------
    subdf : DataFrame
        Data revision records for a signal in a single location.
    value_type : string
        Indicate which value is chosen for analysis.
    melt : bool, optional
        Return a pivot table or a melt one. The default is False.
    Target : string
        The definition of the target. If target == "newest", we use the newest
        report for target, otherwise, we take the reports on lag int(target) 
        as the target.
        The default is "newest".


    """
    pivot_table = subdf.copy().pivot_table(index="time_value", columns="lag", values=value_type)
    pivot_table.index = pivot_table.index.astype(str)
    
    pivot_table["newest_report"] = pivot_table.ffill(axis=1).iloc[:, -1] 
    
    pivot_table.loc[pivot_table["newest_report"] == subdf["lag"].min()] = \
        pivot_table.loc[pivot_table["newest_report"] == subdf["lag"].min()] + 1
    
    for lag in range(subdf["lag"].min(), subdf["lag"].max() + 1):
        pivot_table[lag] = pivot_table[lag] / pivot_table["newest_report"]
    
    pivot_table.drop("newest_report", axis=1, inplace=True)
    
    unpivot = pd.melt(pivot_table.reset_index(), id_vars=["time_value"], var_name = ["lag"], value_name = value_type)
    unpivot["time_value"] = unpivot["time_value"].astype("datetime64[ns]")
    # unpivot = unpivot.loc[unpivot["time_value"] <= datetime(2022, 9, 15)]
    
    if melt:
        return unpivot
    else:
        pivot_table = unpivot.pivot_table(index="lag", columns="time_value", values=value_type)   
        return pivot_table

