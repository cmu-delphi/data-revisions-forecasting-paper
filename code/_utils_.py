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
from .constants import data_dir, filtered_states, MA_POP
####################################
### Read Data ######################
####################################

def read_chng_outpatient(): 
    df = pd.read_csv("/Users/jingjingtang/Documents/backfill-recheck/paper/data/raw/chng_outpatient_input/CHNG_outpatient_state_combined_df_until20230218.csv",
                     parse_dates=["time_value", "issue_date"])
    df = df.loc[(df["time_value"] <= datetime(2023, 1, 10))
                &(df["time_value"] >= datetime(2021, 4, 1))
                & (~df["geo_value"].isin(set(filtered_states)))].sort_values(["issue_date", "time_value"])  
    pdList = []
    for geo in df["geo_value"].unique():
        subdf = df[df["geo_value"] == geo]
        subdf["value_7dav_covid"] = subdf["value_covid"].groupby(subdf["issue_date"]).shift(1).rolling(7).mean()
        subdf["value_7dav_total"] = subdf["value_total"].groupby(subdf["issue_date"]).shift(1).rolling(7).mean()
        pdList.append(subdf)
    df = pd.concat(pdList)
    
    df["7dav_frac"] = (df["value_7dav_covid"]+1)/(df["value_7dav_total"]+1)
    return df
    
# df = read_chng_outpatient()

def read_ma_dph():
    file_dir = data_dir + "raw/MA-DPH-covid-alldata.csv"
    ma_df = pd.read_csv(file_dir, parse_dates=["test_date", "issue_date"])
    ma_df.rename({"confirmed_case_7d_avg": "value_covid", "test_date": "time_value"}, axis=1, inplace=True)
    ma_df["lag"] = [(x- y).days for x, y in zip(ma_df["issue_date"], ma_df["time_value"])]
    ma_df["geo_value"] = "ma"
    ma_df["7dav_frac"] = ma_df["value_covid"] / MA_POP
    
    return ma_df

def read_quidel():
    file_dir = data_dir + "raw/quidel_allages_state_combined_df_until20230216.csv"
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


def read_proj(path):
    """
    Read correction results stored in a single file.
    """
    df = pd.read_csv(path,
             parse_dates=["issue_date", "time_value"]).sort_values(["issue_date", "time_value"])    
    # new_cols = []
    for col in df.columns:
        if "predicted_tau" in col:
            df[col] = np.exp(df[col])
            #new_col = col + "_7dav"
            #df[new_col] = df[col].groupby(df["issue_date"]).shift(1).rolling(7).mean()
    return df

def read_folder(path):
    """
    Read the correction results (per file per location) stored in a folder
    """
    pdList = []
    for fn in os.listdir(path):
        if ("coef" not in fn) & (".csv" in fn) & ("addsmallvalues." in fn): #& ("noyitl" in fn)
            try:
                df = read_proj(path+fn)
                pdList.append(df)
            except:
                pass
    return pd.concat(pdList)


def read_chng_outpatient_result():
    pdList = []
    def read_folder(path, tw):
        """
        Read the correction results (per file per location) stored in a folder
        """
        pdList = []
        for fn in os.listdir(path):
            if ("coef" not in fn) & (".csv" in fn) &(str(tw) in fn) & ("0.1" in fn):#& ("noyitl" in fn) 
                try:
                    df = read_proj(path+fn)
                    pdList.append(df)
                except:
                    pass
        return pd.concat(pdList)
    for tw in [180, 365]:
        df = read_folder("./data/results/chng_outpatient_fraction_7dav/", tw)
        df = df.loc[(df["time_value"] <= datetime(2023, 1, 10))
                    &(df["time_value"] >= datetime(2021, 6, 1))
                    & (~df["geo_value"].isin(set(filtered_states)))].sort_values(["issue_date", "time_value"])
        df["mae"] = abs(df["log_value_7dav"]-df["log_value_target_7dav"])
        df["tw"] = tw
        pdList.append(df)
    
    return pd.concat(pdList)

def read_quidel_result():
    def read_folder(path):
        """
        Read the correction results (per file per location) stored in a folder
        """
        pdList = []
        for fn in os.listdir(path):
            if ("coef" not in fn) & (".csv" in fn) & ("180" in fn) & ("0.1" in fn): #& ("noyitl" in fn) # noyilt should be removed
                try:
                    df = read_proj(path+fn)
                    pdList.append(df)
                except:
                    pass
        return pd.concat(pdList)
    
    df = read_folder("./data/results/quidel_fraction_7dav/")
    # df = read_folder("/Users/jingjingtang/Downloads/quidel_fraction_7dav/")
    df = df.loc[(df["time_value"] <= datetime(2022, 12, 12))
                &(df["time_value"] >= datetime(2021, 5, 18))
                & (~df["geo_value"].isin(set(filtered_states)))].sort_values(["issue_date", "time_value"])
    df["mae"] = abs(df["log_value_7dav"]-df["log_value_target_7dav"])
    df["tw"] = 180
    
    
    def read_folder(path):
        """
        Read the correction results (per file per location) stored in a folder
        """
        pdList = []
        for fn in os.listdir(path):
            if ("coef" not in fn) & (".csv" in fn) & ("365" in fn) & ("0.1" in fn):
                try:
                    df = read_proj(path+fn)
                    pdList.append(df)
                except:
                    pass
        return pd.concat(pdList)
    
    # df2 = read_folder("./data/results/quidel_fraction_7dav_addsmallvalues/")
    df2 = read_folder("./data/results/quidel_fraction_7dav/")
    # df = read_folder("/Users/jingjingtang/Downloads/quidel_fraction_7dav/")
    df2 = df2.loc[(df2["time_value"] <= datetime(2022, 12, 12))
                &(df2["time_value"] >= datetime(2021, 5, 18))
                & (~df2["geo_value"].isin(set(filtered_states)))].sort_values(["issue_date", "time_value"])
    df2["mae"] = abs(df2["log_value_7dav"]-df2["log_value_target_7dav"])
    df2["tw"] = 365
    
    return pd.concat([df, df2])
        
def read_ma_dph_result():
    def read_folder(path, tw):
        """
        Read the correction results (per file per location) stored in a folder
        """
        pdList = []
        for fn in os.listdir(path):
            if ("coef" not in fn) & (".csv" in fn) & (str(tw) in fn): #& ("noyitl" in fn) # noyilt should be removed
                try:
                    df = read_proj(path+fn)
                    pdList.append(df)
                except:
                    pass
        return pd.concat(pdList)
    pdList = []
    plt.figure()
    for tw in [180, 365]:
        # df = read_proj("~/Downloads/ma-dph-testwindow7/ma_daily_tw%d.csv"%tw)
        df = read_proj(data_dir + "results/ma-dph/ma_daily_tw%d.csv"%tw)
        df = df.loc[(df["time_value"] <= datetime(2022, 6, 26))
                    &(df["time_value"] >= datetime(2021, 9, 1))
                    & (~df["geo_value"].isin(set(filtered_states)))].sort_values(["issue_date", "time_value"])
        df["mae"] = abs(df["log_value_7dav"]-df["log_value_target"])
        df["tw"] = tw
        
        subdf = df.loc[df["lag"] == 1]
        plt.plot(subdf["time_value"], subdf["wis"], label="tw=%d"%tw)
        pdList.append(df)    
    plt.legend()        
    return pd.concat(pdList)

def read_chng_outpatient_count_result():
    def read_folder(path, tw):
        """
        Read the correction results (per file per location) stored in a folder
        """
        pdList = []
        for fn in os.listdir(path):
            if ("coef" not in fn) & (".csv" in fn) & (str(tw) in fn): #& ("noyitl" in fn) # noyilt should be removed
                try:
                    df = read_proj(path+fn)
                    pdList.append(df)
                except:
                    pass
        return pd.concat(pdList)
    
    pdList = []
    for tw in [180, 365]:
        # df= read_folder("/Users/jingjingtang/Downloads/chng_outpatient_count_covid_ref60/", tw)
        df = read_folder("./data/results/chng_outpatient_count_covid_ref60/", tw)
        df = df.loc[(df["time_value"] <= datetime(2023, 1, 10))
                    &(df["time_value"] >= datetime(2021, 6, 1))
                    & (~df["geo_value"].isin(set(filtered_states)))].sort_values(["issue_date", "time_value"])
        df["mae"] = abs(df["log_value_7dav"] - df["log_value_target"])
        # df["wis_median"] =  abs(df["predicted_tau0.5"]) - np.log(df["value_target"]))
        df["tw"] = tw
        pdList.append(df)    
    return pd.concat(pdList)
 
# def read_chng_outpatient_count_result():
#     df = pd.read_csv(data_dir + "results/20230501_CHNG_outpatient_count_corrected.csv",
#                       parse_dates = ["time_value", "issue_date"]) 
#     df = df.loc[(df["time_value"] <= datetime(2023, 1, 10))
#                 &(df["time_value"] >= datetime(2021, 6, 1))
#                 & (~df["geo_value"].isin(set(filtered_states)))].sort_values(["issue_date", "time_value"])
   
#     df["mae"] = abs(np.log(df["value_raw_covid"]+1) - np.log(df["value_target_covid"]+1))
#     df["wis_median"] =  abs(np.log(df["predicted_tau0.5_covid"]) - np.log(df["value_target_covid"]+1))
#     df["wis"] = df["wis_covid"]
#     return df


def get_data(df, state="ma"):
    
    subdf = df.loc[(df["geo_value"] == state)
                          & (df["time_value"] >= datetime(2021, 6, 1))
                          & (df["lag"] <= 240)]
    subdf.index = list(range(subdf.shape[0]))
    
    target_df = subdf.loc[subdf.groupby("time_value")["issue_date"].idxmax()][["time_value", "value_covid"]]
    
    subdf = subdf.merge(target_df, on=["time_value"], how="left", suffixes=["", "_target"])
    subdf["%reported"] = subdf["value_covid"] / subdf["value_covid_target"]
    subdf["mae_log"] = abs(np.log(subdf["value_covid"]+1) - np.log(subdf["value_covid_target"]+1))
    subdf["mape"] = abs(subdf["value_covid"] - subdf["value_covid_target"])/(subdf["value_covid_target"])
    subdf = subdf.loc[subdf["lag"] <= 60].dropna()
    
    mean_df = subdf.groupby("lag").mean().reset_index()
    q5_df = subdf.groupby("lag").quantile(0.05).reset_index()
    q95_df = subdf.groupby("lag").quantile(0.95).reset_index()
    
    return mean_df, q5_df, q95_df


