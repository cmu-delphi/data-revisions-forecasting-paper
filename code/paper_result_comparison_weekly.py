#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Count projection example

multicolumn{2}{c|}{\makecell{\textbf{WIS}\\(lag=7 (days), at log scale)}}         & & & &\\ \hline
\multirow{8}{*}{\rotatebox{90}{\textbf{Dataset}}} &\makecell{Insurance Claims\\(Daily, State)}     &$0.111 \pm  0.001$ &$0.108\pm  0.005$ & $0.295 \pm 0.010$&$0.635 \pm  0.013$\\
 &\makecell{Confirmed Cases\\(Daily, State, MA only)} &$0.014 \pm  0.001$ &$0.004\pm  0.002$ &$0.022 \pm  0.009$ &$0.006 \pm 0.001$\\
 &\makecell{Insurance Claims\\(Weekly, State)} &$0.092 \pm  0.002$ &$0.061 \pm 0.071$ &$0.137 \pm 0.184$ &$0.432\pm  0.242$\\
 &\makecell{Dengue Fever Cases\\(Weekly, State, PR only)} &$0.290 \pm  0.429$ &$0.067 \pm 0.040$ &$0.121 \pm  0.186$ &$0.693\pm  0.429$\\
\hline\hline


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
                        filtered_states, map_list)
from ._utils_ import (read_chng_outpatient_result, read_ma_dph_result, 
                      read_quidel_result, read_chng_outpatient_count_result, re_to_wis)
dfs = {}
pdList = []
for state in map_list:
    try:
        temp = pd.read_csv("/Users/jingjingtang/Documents/backfill-recheck/paper/data/results/chng_outpatient_count_covid_ref63/%s_weekly.csv"%state,
                                   parse_dates=["time_value", "issue_date"])
        temp["geo_value"] = state
        pdList.append(temp)
    except:
        pass
dfs["CHNG Outpatient Count"] = pd.concat(pdList)
dfs["dengue"] = pd.read_csv("/Users/jingjingtang/Documents/backfill-recheck/paper/data/results/denguedat_weekly_ref70/pr_weekly.csv",
                            parse_dates=["time_value", "issue_date"])
dfs["dengue"] = pd.read_csv("/Users/jingjingtang/Downloads/denguedat_weekly_ref70/pr_weekly.csv",
                            parse_dates=["time_value", "issue_date"])



nobBS_dfs={}
nobBS_dfs["CHNG Outpatient Count"] = pd.read_csv("/Users/jingjingtang/Documents/backfill-recheck/nobBS_result/chng_outpatient_covid/combined_result_weekly.csv",
                           parse_dates=["onset_date", "issue_date"])
nobBS_dfs["CHNG Outpatient Count"]["lag"] = [(x-y).days for x, y in zip(nobBS_dfs["CHNG Outpatient Count"]["issue_date"], nobBS_dfs["CHNG Outpatient Count"]["onset_date"])]
nobBS_dfs["CHNG Outpatient Count"]["time_value"] = nobBS_dfs["CHNG Outpatient Count"]["onset_date"] 
nobBS_dfs["CHNG Outpatient Count"]["wis"] = abs(np.log(nobBS_dfs["CHNG Outpatient Count"]["estimate"] + 1) - np.log(nobBS_dfs["CHNG Outpatient Count"]["value_target"] + 1))

nobBS_dfs["dengue"] = pd.read_csv("/Users/jingjingtang/Documents/backfill-recheck/nobBS_result/dengue/combined_result_weekly.csv", parse_dates=["issue_date", "onset_date"])
nobBS_dfs["dengue"]["lag"] = [(x-y).days for x, y in zip(nobBS_dfs["dengue"]["issue_date"], nobBS_dfs["dengue"]["onset_date"])]
nobBS_dfs["dengue"]["time_value"] = nobBS_dfs["dengue"]["onset_date"] 

epinowcast_dfs = {}
epinowcast_dfs["dengue"] = pd.read_csv("/Users/jingjingtang/Documents/backfill-recheck/epinowcast_result/dengue/pr_weekly.csv", parse_dates=["report_date", "reference_date"])
part2 = pd.read_csv("/Users/jingjingtang/Documents/backfill-recheck/epinowcast_result/dengue/pr_weekly_part2.csv", parse_dates=["report_date", "reference_date"])
epinowcast_dfs["dengue"] = pd.concat([epinowcast_dfs["dengue"], part2])
epinowcast_dfs["dengue"]["lag"] = [(x-y).days for x, y in zip(epinowcast_dfs["dengue"]["report_date"], epinowcast_dfs["dengue"]["reference_date"])]
epinowcast_dfs["dengue"]["time_value"] = epinowcast_dfs["dengue"]["reference_date"] 
epinowcast_dfs["dengue"]["issue_date"] = epinowcast_dfs["dengue"]["report_date"] 
pdList = []
for state in map_list:
    try:
        temp = pd.read_csv("/Users/jingjingtang/Documents/backfill-recheck/epinowcast_result/chng_outpatient_covid/%s_weekly.csv"%state,
                                   parse_dates=["reference_date", "report_date"])
        temp["geo_value"] = state
        pdList.append(temp)
    except:
        pass
epinowcast_dfs["CHNG Outpatient Count"] = pd.concat(pdList)
epinowcast_dfs["CHNG Outpatient Count"]["lag"] = [(x-y).days for x, y in zip(epinowcast_dfs["CHNG Outpatient Count"]["report_date"], epinowcast_dfs["CHNG Outpatient Count"]["reference_date"])]
epinowcast_dfs["CHNG Outpatient Count"]["time_value"] = epinowcast_dfs["CHNG Outpatient Count"]["reference_date"] 
epinowcast_dfs["CHNG Outpatient Count"]["issue_date"] = epinowcast_dfs["CHNG Outpatient Count"]["report_date"] 

####################################
### COUNT EXAMPLES
####################################

df = dfs["CHNG Outpatient Count"]
df = df.loc[(df["issue_date"] == df["issue_date"].isin(dfs["epinowcast"]["test_date"]))]
lag = 14
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


yticks = [0, 10, 50, 100, 150]
plt.style.use('default')
fig = plt.figure(figsize=(12, 10))
# for idx, signal in enumerate(["CHNG Outpatient Count", "dengue"]):
for idx, signal in enumerate(["dengue"]):
    delphi_df = dfs[signal].groupby(["lag"]).agg(
        mean=('wis', 'mean'),
        sem=('wis', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
        quantile_0_5=('wis', lambda x: np.quantile(x, 0.5)),         # Median (50th percentile)
        quantile_0_95=('wis', lambda x: np.quantile(x, 0.95))
        ).reset_index()
    
    # baseline = dfs[signal].groupby(["lag"]).agg(
    #     mean=('mae', 'mean'),
    #     sem=('mae', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
    #     quantile_0_5=('mae', lambda x: np.quantile(x, 0.5)),         # Median (50th percentile)
    #     quantile_0_95=('mae', lambda x: np.quantile(x, 0.95))
    #     ).reset_index()
       
    ax1 = plt.subplot(1, 1, idx+1)
    ax1.plot(delphi_df["lag"], delphi_df["mean"], label="Delphi", linewidth=3.0)    
    ax1.fill_between(delphi_df["lag"], 
                      delphi_df["quantile_0_5"], 
                      delphi_df["quantile_0_95"], alpha=0.1)    
    # ax1.plot(baseline["lag"],  baseline["mean"], label="Baseline",linewidth=3.0)
    try:
        nobBS_df = nobBS_dfs[signal].groupby(["lag"]).agg(
            mean=('wis', 'mean'),
            sem=('wis', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
            quantile_0_5=('wis', lambda x: np.quantile(x, 0.5)),         # Median (50th percentile)
            quantile_0_95=('wis', lambda x: np.quantile(x, 0.95))
            ).reset_index()
        
        ax1.plot(nobBS_df["lag"], nobBS_df["mean"], label="nobBS", linewidth=3.0)
        ax1.fill_between(nobBS_df["lag"], 
                          nobBS_df["quantile_0_5"], 
                          nobBS_df["quantile_0_95"], alpha=0.1)
    except:
        pass
    # ax1.plot(epinowcast_df["lag"], epinowcast_df["mean"], label="epinowcast", linewidth=3.0)
    try:
        epinowcast_df = epinowcast_dfs[signal].groupby(["lag"]).agg(
            mean=('wis', 'mean'),
            sem=('wis', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
            quantile_0_5=('wis', lambda x: np.quantile(x, 0.5)),         # Median (50th percentile)
            quantile_0_95=('wis', lambda x: np.quantile(x, 0.95))
            ).reset_index()
        ax1.plot(epinowcast_df["lag"], epinowcast_df["mean"], label="Epinowcast", linewidth=3.0)    
        ax1.fill_between(epinowcast_df["lag"], 
                          epinowcast_df["quantile_0_5"], 
                          epinowcast_df["quantile_0_95"], alpha=0.1)    
    except:
        pass
    
    
    ax1.grid(True)
    ax1.set_xlabel("Lag (Days)", fontsize=30)
    # ax1.set_ylabel("WIS", fontsize=30)
    if idx == 0:
        ax1.set_ylabel("WIS", fontsize=30)
    ax1.set_xlim((0, 57))
    ax1.set_ylim((0, 1))
    # ax1.axhline(np.log(1.1), linestyle="--", color="gray", label = "10% Absolute Relative Error", linewidth=3.0)
    # ax1.axhline(np.log(1.2), linestyle="-.", color="gray", label = "20% Absolute Relative Error", linewidth=3.0)
    ax1.set_xticks(np.arange(0, 57, 7))
    # ax1.set_yticks(np.append(ax1.get_yticks(), [0.1])) # Add tick for a specific absolute relative error
    ax1.tick_params(axis='both', labelsize=20)
    
    ax2 = ax1.twinx()
    ax2.plot([],[])
    ax2.set_yticks(list(map(re_to_wis, yticks)))
    ax2.set_yticklabels([str(x)+"%" for x in yticks])
    ax2.set_ylim((ax1.get_ylim()[0], ax1.get_ylim()[1]))
    ax2.tick_params(axis='both', labelsize=18)  
    ax2.grid(True)    
    # ax1.legend(bbox_to_anchor=(0.8, 0.85), fontsize=25)
    if signal == "CHNG Outpatient Count":
        ax1.set_title("Insurance claims", fontsize=35)
    else:
        ax1.set_title(signal, fontsize=35)
    ax2.set_ylabel('Absolute Relative Error', fontsize=30, rotation=270, labelpad=30)
    # if idx == 1:
    #     ax2.set_ylabel('Absolute Relative Error', fontsize=30, rotation=270, labelpad=30)
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
plt.suptitle("Count Forecasting, Weekly", fontsize=40, y=1.07)
plt.savefig(fig_dir + "experiment_count_result_evl_general.pdf", bbox_inches = 'tight')



#############################
# Plot for wis comparison
#############################

lag = 0

plt.figure(figsize=(20, 8))
plt.subplot(3, 1, 1)
signal = "dengue"
delphi_subdf =  dfs[signal].loc[(dfs[signal]["lag"] == lag)]
                           # & (dfs[signal]["time_value"] <= datetime(1998, 10, 16))
                           # & (dfs[signal]["time_value"] >= datetime(1994, 5, 31))]
plt.plot(delphi_subdf["time_value"], delphi_subdf["wis"], label="Delphi")
print(np.mean(delphi_subdf["wis"]))
nobBS_subdf =  nobBS_dfs[signal].loc[(nobBS_dfs[signal]["lag"] == lag)]
                   # & (nobBS_dfs[signal]["time_value"] <= datetime(1998, 10, 16))
                   # & (nobBS_dfs[signal]["time_value"] >= datetime(1994, 5, 31))]
plt.plot(nobBS_subdf["time_value"], nobBS_subdf["wis"], label="nobBS") 
epinowcast_subdf =  epinowcast_dfs[signal].loc[(epinowcast_dfs[signal]["lag"] == lag)].sort_values("time_value")
                                               # & (epinowcast_dfs[signal]["time_value"] <= datetime(1998, 10, 16))
                                               # & (epinowcast_dfs[signal]["time_value"] >= datetime(1994, 5, 31))
                                               
plt.plot(epinowcast_subdf["time_value"], epinowcast_subdf["wis"], label="epinowcast") 
plt.ylabel("WIS",fontsize=20)
plt.xticks(fontsize=20)
plt.xlabel("Reference Date", fontsize=20)
plt.legend(fontsize=20)
plt.title("Lag= %d"%lag, fontsize=30)
plt.ylim(0, 2)
print(np.mean(nobBS_subdf["wis"]))

plt.subplot(3, 1, 2)
plt.plot(delphi_subdf["time_value"], delphi_subdf["log_7dav_slope"])
plt.xticks(fontsize=20)
plt.ylabel("log 7dav slope", fontsize=20)
plt.xlabel("Reference Date", fontsize=20)
plt.legend(fontsize=20)


plt.subplot(3, 1, 3)
# plt.plot(delphi_subdf["time_value"], delphi_subdf["value_target"], label="target")
plt.plot(delphi_subdf["time_value"], delphi_subdf["value_7dav"], label="first release")
plt.xticks(fontsize=20)
plt.ylabel("# Dengue Fever Case", fontsize=20)
plt.xlabel("Reference Date", fontsize=20)
plt.legend(fontsize=20)
plt.tight_layout()


subdf =  epinowcast_dfs[signal].loc[epinowcast_dfs[signal]["lag"] == lag]
plt.plot(subdf["time_value"], subdf["wis"]) 
print(np.mean(subdf["wis"]))





#############################
# check coefficients
#############################
subdf =  dfs[signal].loc[(dfs[signal]["lag"] == lag)]
for col in [  'log_value_7dav', 'log_value_7dav_lag7',
'log_value_7dav_lag14',
'sqrty1', 'sqrty2', 'sqrty3']:
    plt.figure(figsize=(20, 4))
    plt.plot(subdf["time_value"], subdf[col])
    plt.xticks(fontsize=20)
    plt.title(col)
    
    
coefs = pd.read_csv("/Users/jingjingtang/Downloads/denguedat_weekly_ref70/pr_weekly_coefs.csv",
                    parse_dates=["training_end_date", "training_start_date"])
subdf = coefs.loc[(coefs["tau"] == 0.1)
                  & (coefs["test_lag"] == 7)]
for col in [ 'W1_issue_coef', 'W2_issue_coef',
'W3_issue_coef', 'log_value_7dav_coef', 'value_7dav_lag7_coef',
'value_7dav_lag14_coef', 'sqrty0_coef', 'sqrty1_coef', 'sqrty2_coef',
'sqrty3_coef']:
    plt.figure(figsize=(20, 4))
    plt.plot(subdf["training_end_date"], subdf[col])
    plt.title(col)
    

#############################
# plotly generation
#############################
selected_cols = ["time_value", "lag", "wis"]

combined = dfs["dengue"][selected_cols + ["log_value_7dav", "log_value_target", "log_value_7dav_lag7"]].merge(
    epinowcast_dfs["dengue"][selected_cols], on=["time_value", "lag"], suffixes=["_delphi", "_epinowcast"]).merge(
    nobBS_dfs["dengue"][selected_cols], on=["time_value", "lag"])
combined.rename({"wis": "wis_nobBS"}, axis=1, inplace=True)

df = combined   
import plotly.graph_objects as go

# Create a figure
fig = go.Figure()


# Add traces for val1, val2, and val3 for each lag
lags = df["lag"].unique()
buttons = []
visibility_template = []

# Initialize the trace index
trace_idx = 0

# Add val1 and val2 to the primary y-axis, grouped by 'lag'
for lag, group in combined.groupby("lag"):
    visibility_template.extend([True, True, True, True, True, True])  # All traces visible by default
    fig.add_trace(go.Scatter(
        x=group["time_value"],
        y=group["wis_delphi"],
        mode="lines",
        name=f"Delphi (lag={lag})",
        yaxis="y1"
    ))
    trace_idx += 1
    fig.add_trace(go.Scatter(
        x=group["time_value"],
        y=group["wis_epinowcast"],
        mode="lines",
        name=f"epinowcast (lag={lag})",
        yaxis="y1"
    ))
    trace_idx += 1
    fig.add_trace(go.Scatter(
        x=group["time_value"],
        y=group["wis_nobBS"],
        mode="lines",
        name=f"nobBS (lag={lag})",
        yaxis="y1"
    ))
    trace_idx += 1
    fig.add_trace(go.Scatter(
        x=group["time_value"],
        y=group["log_value_target"],
        mode="lines",
        name=f"log(Target+1) (lag=70 days )",
        yaxis="y2",
    ))
    trace_idx += 1
    fig.add_trace(go.Scatter(
        x=group["time_value"],
        y=group["log_value_7dav_lag7"],
        mode="lines",
        name=f"log(Prev Report+1) (lag=7 days )",
        yaxis="y2",
    ))
    trace_idx += 1
    fig.add_trace(go.Scatter(
        x=group["time_value"],
        y=group["log_value_7dav"],
        mode="lines",
        name=f"log(Current Report+1)(lag={lag} days)",
        yaxis="y2"
    ))
    trace_idx += 1
    
    
    # Create button for toggling this lag
    button_visibility = [False] * trace_idx
    button_visibility[-6:] = [True, True, True, True, True, True]  # Enable last 5 traces (this lag group)
    
    buttons.append(dict(
        label=f"Toggle Lag {lag}",
        method="update",
        args=[{"visible": button_visibility + [False] * (len(lags) * 6 - len(button_visibility))}]
    ))

# Add a button to reset (show all traces)
buttons.append(dict(
    label="Show All",
    method="update",
    args=[{"visible": visibility_template}]
))

# Update layout to include two y-axes
fig.update_layout(
    title="Forecast Performance Comparison",
    xaxis_title="Reference Date",
    yaxis=dict(
        title="WIS",
        titlefont=dict(color="blue"),
        tickfont=dict(color="blue")
    ),
    yaxis2=dict(
        title="log(#Dengue Fever Case+1)",
        titlefont=dict(color="red"),
        tickfont=dict(color="red"),
        overlaying="y",
        side="right"
    ),
    legend=dict(
        x=1.05,  # Move the legend slightly to the right
        y=1,  # Align it at the top
        xanchor="left",
        yanchor="top"
    ),
    legend_title="Legend",
    updatemenus=[
        dict(
            type="buttons",
            direction="left",
            buttons=buttons,
            showactive=True,
            x=0,  # Position to the right of the figure
            y=-0.2,
            xanchor="left",
            yanchor="top"
        )
    ],
    hovermode="x unified",  # Show hover information for all points at the same x-coordinate
    xaxis=dict(
        showspikes=True,  # Enable spike line
        spikemode="across",  # Spike line extends across the entire plot
        spikesnap="cursor",  # Spike line snaps to the cursor
        spikethickness=1,  # Thickness of the spike line
        spikecolor="gray",  # Color of the spike line
        spikedash="dot",  # Dash style of the spike line
        hoverformat="%Y-%m-%d", 
    )
)

# Save the figure as an HTML file
fig.write_html("troubleshooting.html")


fig.show()
 




epinowcast_dfs["dengue"]["predicted_tau0.5"] = np.log(epinowcast_dfs["dengue"]["q50"] + 1)
epinowcast_dfs["dengue"]["predicted_tau0.1"] = np.log(epinowcast_dfs["dengue"]["q10"] + 1)
epinowcast_dfs["dengue"]["predicted_tau0.9"] = np.log(epinowcast_dfs["dengue"]["q90"] + 1)
selected_cols = ["time_value", "lag", "wis", "predicted_tau0.5", "predicted_tau0.1", "predicted_tau0.9"]

combined = dfs["dengue"][selected_cols + ["log_value_7dav", "log_value_target"]].merge(
    epinowcast_dfs["dengue"][selected_cols], on=["time_value", "lag"], suffixes=["_delphi", "_epinowcast"])
combined.rename({"wis": "wis_nobBS"}, axis=1, inplace=True)

df = combined   
import plotly.graph_objects as go

color_map = {
    'wis_delphi': 'orange',
    'predicted_tau0.5_delphi': 'blue',
    'predicted_tau0.1_delphi': 'blue',
    'predicted_tau0.9_delphi': 'blue',
    'predicted_tau0.5_epinowcast': 'red',
    'predicted_tau0.1_epinowcast': 'red',
    'predicted_tau0.9_epinowcast': 'red',
    'log_value_7dav': 'gray',
    'log_value_7dav': 'black',
}


# Create a figure
fig = go.Figure()


# Add traces for val1, val2, and val3 for each lag
lags = df["lag"].unique()
buttons = []
visibility_template = []

# Initialize the trace index
trace_idx = 0

# Add val1 and val2 to the primary y-axis, grouped by 'lag'
for lag, group in combined.groupby("lag"):
    visibility_template.extend([True, True, True, True, True, True, True, True, True])  # All traces visible by default
    fig.add_trace(go.Scatter(
        x=group["time_value"],
        y=group["wis_delphi"],
        mode="lines",
        name=f"Delphi WIS(lag={lag})",
        yaxis="y1",
        line=dict(color=color_map.get("wis_delphi", 'gray'))
    ))
    trace_idx += 1
    fig.add_trace(go.Scatter(
        x=group["time_value"],
        y=group["predicted_tau0.5_delphi"],
        mode="lines",
        name=f"Delphi Predicted q50(lag={lag})",
        yaxis="y2",
        line=dict(color=color_map.get("predicted_tau0.5_delphi", 'gray'))
    ))
    trace_idx += 1
    fig.add_trace(go.Scatter(
        x=group["time_value"],
        y=group["predicted_tau0.5_epinowcast"],
        mode="lines",
        name=f"Epinowcast Predicted q50 (lag={lag})",
        yaxis="y2",
        line=dict(color=color_map.get("predicted_tau0.5_epinowcast", 'gray'))
    ))
    trace_idx += 1
    fig.add_trace(go.Scatter(
        x=group["time_value"],
        y=group["predicted_tau0.1_delphi"],
        mode="lines",
        name=f"Delphi Predicted Q10(lag={lag})",
        yaxis="y2",
        line=dict(color=color_map.get("predicted_tau0.1_delphi", 'gray'))
    ))
    trace_idx += 1
    fig.add_trace(go.Scatter(
        x=group["time_value"],
        y=group["predicted_tau0.1_epinowcast"],
        mode="lines",
        name=f"Epinowcast Predicted Q10(lag={lag})",
        yaxis="y2",
        line=dict(color=color_map.get("predicted_tau0.1_epinowcast", 'gray'))
    ))
    trace_idx += 1
    fig.add_trace(go.Scatter(
        x=group["time_value"],
        y=group["predicted_tau0.9_delphi"],
        mode="lines",
        name=f"Delphi Predicted Q90(lag={lag})",
        yaxis="y2",
        line=dict(color=color_map.get("predicted_tau0.9_delphi", 'gray'))
    ))
    trace_idx += 1
    fig.add_trace(go.Scatter(
        x=group["time_value"],
        y=group["predicted_tau0.9_epinowcast"],
        mode="lines",
        name=f"Epinowcast Predicted Q90(lag={lag})",
        yaxis="y2",
        line=dict(color=color_map.get("predicted_tau0.9_epinowcast", 'gray'))
    ))
    trace_idx += 1
    fig.add_trace(go.Scatter(
        x=group["time_value"],
        y=group["log_value_target"],
        mode="lines",
        name=f"log(Target+1) (lag=70 days )",
        yaxis="y2",
        line=dict(color=color_map.get("log_value_target", 'gray'))
    ))
    trace_idx += 1
    fig.add_trace(go.Scatter(
        x=group["time_value"],
        y=group["log_value_7dav"],
        mode="lines",
        name=f"log(Current Report+1)(lag={lag} days)",
        yaxis="y2",
        line=dict(color=color_map.get("log_value_7dav", 'gray'))
    ))
    trace_idx += 1
    
    
    # Create button for toggling this lag
    button_visibility = [False] * trace_idx
    button_visibility[-9:] = [True, True, True, True, True, True, True, True, True]  # Enable last 5 traces (this lag group)
    
    buttons.append(dict(
        label=f"Toggle Lag {lag}",
        method="update",
        args=[{"visible": button_visibility + [False] * (len(lags) * 9 - len(button_visibility))}]
    ))

# Add a button to reset (show all traces)
buttons.append(dict(
    label="Show All",
    method="update",
    args=[{"visible": visibility_template}]
))

# Update layout to include two y-axes
fig.update_layout(
    title="Forecast Performance Comparison",
    xaxis_title="Reference Date",
    yaxis=dict(
        title="WIS",
        titlefont=dict(color="blue"),
        tickfont=dict(color="blue")
    ),
    yaxis2=dict(
        title="log(#Dengue Fever Case+1)",
        titlefont=dict(color="red"),
        tickfont=dict(color="red"),
        overlaying="y",
        side="right"
    ),
    legend=dict(
        x=1.05,  # Move the legend slightly to the right
        y=1,  # Align it at the top
        xanchor="left",
        yanchor="top"
    ),
    legend_title="Legend",
    updatemenus=[
        dict(
            type="buttons",
            direction="left",
            buttons=buttons,
            showactive=True,
            x=0,  # Position to the right of the figure
            y=-0.2,
            xanchor="left",
            yanchor="top"
        )
    ],
    hovermode="x unified",  # Show hover information for all points at the same x-coordinate
    xaxis=dict(
        showspikes=True,  # Enable spike line
        spikemode="across",  # Spike line extends across the entire plot
        spikesnap="cursor",  # Spike line snaps to the cursor
        spikethickness=1,  # Thickness of the spike line
        spikecolor="gray",  # Color of the spike line
        spikedash="dot",  # Dash style of the spike line
        hoverformat="%Y-%m-%d", 
    )
)

# Save the figure as an HTML file
fig.write_html("troubleshooting_with_quantiles.html")


fig.show()


#############################
# Plot for runtime comparison
#############################
rt = pd.DataFrame(columns = ["unit",  "method", "data", "mean", "std"])


nobBS_df = nobBS_dfs[signal].groupby(["lag"]).agg(
    mean=('elapsed_time', 'mean'),
    std=('elapsed_time', lambda x: np.std(x, ddof=1)),
    sem=('elapsed_time', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))
    ).reset_index()

epinowcast_df = epinowcast_dfs[signal].groupby(["lag"]).agg(
    mean=('elapsed_time', 'mean'),
    std=('elapsed_time', lambda x: np.std(x, ddof=1)),
    sem=('elapsed_time', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))
    ).reset_index()

delphi_df = dfs[signal].loc[dfs[signal]["lag"].isin(
    list(range(0, 15)) + [21, 35, 51] 
    )].groupby(["issue_date", "geo_value"])["elapsed_time"].sum().reset_index()
mean_elapsed_time = delphi_df["elapsed_time"].mean()
std_elapsed_time = delphi_df["elapsed_time"].std(ddof=1)
sem_elapsed_time = std_elapsed_time / np.sqrt(len(delphi_df))


rt.loc[0] = ["weekly", "nobBS", "CHNG Outpatient Count", 1.637980, 0.105288]
rt.loc[1] = ["weekly", "Delphi-training", "CHNG Outpatient Count", 0.40787, 0.045776]
rt.loc[2] = ["weekly", "Delphi-testing", "CHNG Outpatient Count", 0.0069, 0.00072]

rt.loc[3] = ["weekly", "nobBS", "dengue", 8.337, 1.050]


plt.bar(rt["method"], rt["mean"], yerr=rt["std"])
plt.xticks(fontsize=15)
plt.title("Runtime(s)", fontsize=25)


epinowcast_df = epinowcast_dfs[signal].groupby(["lag"]).agg(
    mean=('wis', 'mean'),
    std=('wis', lambda x: np.std(x, ddof=1)),
    sem=('wis', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))
    ).reset_index()

nobBS_dfs[signal]["baseline"] = abs(np.log(nobBS_dfs[signal]["n.reported"].values+1) - np.log(nobBS_dfs[signal]["value_target"].values+1))
nobBS_df = nobBS_dfs[signal].groupby(["lag"]).agg(
    mean=('wis', 'mean'),
    std=('wis', lambda x: np.std(x, ddof=1)),
    sem=('wis', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))
    ).reset_index()

dfs[signal]["baseline"] = abs(np.log(dfs[signal]["value_raw"].values+1) - np.log(dfs[signal]["value_target"].values+1))
delphi_df = dfs[signal].groupby(["lag"]).agg(
    mean=('wis', 'mean'),
    std=('wis', lambda x: np.std(x, ddof=1)),
    sem=('wis', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))
    ).reset_index()
