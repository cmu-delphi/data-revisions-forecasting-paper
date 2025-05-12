#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Only use chng fraction

@author: jingjingtang
"""
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from _utils_ import *
from constants import filtered_state, map_list

# Use chng outpatient fraction project

df = read_experimental_results(chng_fraction_config, "DelphiRF")


df["target_date"] = df["reference_date"] + timedelta(60)
df["value_7dav"] = np.exp(df["log_value_7dav"])
df["value_target_7dav"] = np.exp(df["log_value_target_7dav"])
target_df = df[["geo_value", "target_date", "value_target_7dav"]].drop_duplicates()
target_df_prev = target_df.copy()
target_df_prev["target_date"] = target_df_prev["target_date"] + timedelta(days=7)
target_df_prev.rename({"value_target_7dav": "value_target_7dav_prev"}, axis=1, inplace=True)
target_df = target_df.merge(target_df_prev, on=["geo_value", "target_date"])
target_df.drop("value_target_7dav", axis=1, inplace=True)
df = df.merge(target_df, on=["geo_value", "target_date"])
df["target_7dav_slope"] = df["value_target_7dav"] / df["value_target_7dav_prev"] 


df["Trend"] = "Flat"
df.loc[df["target_7dav_slope"]>1.25, "Trend"] = "Up"
df.loc[df["target_7dav_slope"]<=0.75, "Trend"] = "Down"

df["Target Level"] = "Medium: otherwise"
df.loc[df["value_target_7dav"] <= df["value_target_7dav"].quantile(0.25), "Target Level"] = r"Low: $\leq$ 25% percentile"
df.loc[df["value_target_7dav"] > df["value_target_7dav"].quantile(0.75), "Target Level"] = r"High: $\geq$ 75% percentile"

yticks = [0, 10, 20, 30, 40, 50, 60]
plt.figure(figsize=(18, 6))
ax1 = plt.subplot(1, 2, 1)
sns.boxplot(df.loc[df["lag"] <= 14], x="lag", y="wis", hue="Trend", 
            showfliers=False, hue_order=["Up", "Flat", "Down"])
plt.xlabel("Lag (Days)", fontsize=30)
plt.ylim([0, 0.5])
plt.xticks(fontsize=20)
plt.yticks(fontsize=25) 
plt.ylabel("WIS", fontsize=30)
plt.legend(title='Trend', fontsize=25, title_fontsize=30)
ax2 = ax1.twinx()
ax2.plot([],[])
ax2.set_yticks(list(map(re_to_wis, yticks)))
ax2.set_yticklabels([str(x)+"%" for x in yticks])
ax2.set_ylim((ax1.get_ylim()[0], ax1.get_ylim()[1]))
ax2.tick_params(axis='both', labelsize=20)  

ax1 = plt.subplot(1, 2, 2)
sns.boxplot(df.loc[df["lag"] <= 14], x="lag", y="wis", hue="Target Level", 
            showfliers=False, hue_order=[r"High: $\geq$ 75% percentile", "Medium: otherwise", r"Low: $\leq$ 25% percentile"])
plt.xlabel("Lag (Days)", fontsize=30)
plt.ylabel("")
plt.ylim(0, 0.5)
plt.xticks(fontsize=20)
plt.yticks(fontsize=25)
plt.legend(title='Target Level', fontsize=25, title_fontsize=30)
ax2 = ax1.twinx()
ax2.plot([],[])
ax2.set_yticks(list(map(re_to_wis, yticks)))
ax2.set_yticklabels([str(x)+"%" for x in yticks])
ax2.set_ylim((ax1.get_ylim()[0], ax1.get_ylim()[1]))
ax2.tick_params(axis='both', labelsize=20)  
ax2.set_ylabel('Absolute Relative Error', fontsize=30, rotation=270, labelpad=30)

plt.tight_layout()
# plt.suptitle("Insurance claims", fontsize=40, y=1.1)
plt.savefig(fig_dir + "experiment_fraction_result_factors.pdf", bbox_inches = 'tight')



fig = plt.figure(figsize=(60, 35))
for i in range(len(map_list)):
    state = map_list[i]
    if state == '':
        continue
    
    subdf = df.loc[(df["geo_value"] == state)
                    & (df["lag"] == 0) # randomly select one, we only need the target
                    & (df["report_date"]<= datetime(2022, 3, 1))
                    & (df["report_date"] >= datetime(2021, 6, 1))]        
    ax = plt.subplot(7, 11, i+1)
    ax.set_title(state.upper(), fontsize=40)
    ax.plot(subdf["reference_date"], subdf["value_target_7dav"], label="Target (lag = 60)")
    
    ups = subdf.loc[subdf["Trend"] == "Up"]
    ax.scatter(ups["reference_date"], ups["value_target_7dav"], color="red", label="Up")
    
    downs = subdf.loc[subdf["Trend"] == "Down"]
    ax.scatter(downs["reference_date"], downs["value_target_7dav"], color="blue", label="Down")

    # Add x-ticks every 4 months
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    
    plt.yticks(fontsize=30)
    plt.ylim(-0.001, 0.071)
    plt.grid()
plt.suptitle("Temporal Annotation of Trend Categories", y = 1.01, fontsize=70)
labels_handles = {
  label: handle for ax in fig.axes for handle, label in zip(*ax.get_legend_handles_labels())
}

fig.legend(
    # handles,
    # labels,   
    labels_handles.values(),
    labels_handles.keys(),
  loc = "upper center",
  bbox_to_anchor = (0.5, 0),
  bbox_transform = plt.gcf().transFigure,
  ncol = 3,
  fontsize=50
)
plt.tight_layout()
plt.savefig(fig_dir + temporal_annotation_for_states.png", bbox_inches="tight")
