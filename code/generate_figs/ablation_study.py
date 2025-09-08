import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _utils_ import (madph_config, chng_count_config,
                     dengue_config, ilicases_config,
                     chng_fraction_config, quidel_config,
                     re_to_wis,
                     read_experimental_results, read_ablation_results)



# DelphiRF results
all_dfs = {}
all_dfs["COVID-19 cases in MA"] = read_experimental_results(madph_config, "DelphiRF")
all_dfs["CHNG Outpatient Count"] = read_experimental_results(chng_count_config, "DelphiRF")
all_dfs["Dengue fever cases"] = read_experimental_results(dengue_config, "DelphiRF")
all_dfs["ILI cases"] = read_experimental_results(ilicases_config, "DelphiRF")
all_dfs["Insurance claims"] = read_experimental_results(chng_fraction_config, "DelphiRF")
all_dfs["Antigen tests"] = read_experimental_results(quidel_config, "DelphiRF")

ablation_dfs = {}
ablation_dfs["COVID-19 cases in MA"] = read_ablation_results("madph")
ablation_dfs["Dengue fever cases"] = read_ablation_results("dengue")
ablation_dfs["ILI cases"] = read_ablation_results("ili")
ablation_dfs["CHNG Outpatient Count"] = read_ablation_results("chng_outpatient_count")
ablation_dfs["Insurance claims"] = read_ablation_results("chng_outpatient_fraction")
ablation_dfs["Antigen tests"] = read_ablation_results("quidel")

label_map = {
    "daily_issue": "Dropped: Day-of-Week (report date)",
    "daily_ref": "Dropped: Day-of-Week (reference date)",
    "week_issue": "Dropped: Week-of-Month (report date)",
    "value_lags": "Dropped: Inverse lag",
    "delta_lags": "Dropped: Revision magnitude",
    "y7dav": "Dropped: 7-day moving average",
    "all": "All features included"   # <-- if you want a baseline line
}

extra_info = {
    "COVID-19 cases in MA": "(Daily, State, MA only)",
    "CHNG Outpatient Count": "(Daily, State, All states)",
    "Dengue fever cases": "(Weekly, State, PR only)",
    "ILI cases": "(Weekly, National)",
    "Insurance claims": "(Daily, State, All states)",
    "Antigen tests": "(Daily, State, All states)"   
}

color_map = {
    "daily_issue": "tab:blue",
    "daily_ref": "tab:orange",
    "week_issue": "tab:green",
    "value_lags": "tab:red",
    "delta_lags": "tab:purple",
    "y7dav": "tab:brown",
    "all": "black"   # baseline
}
  

####################################
### Ablation study for covid daily counts
####################################

yticks = [0, 10, 20, 50]
plt.style.use('default')
fig = plt.figure(figsize=(15, 6))
for idx, signal in enumerate(["COVID-19 cases in MA", "CHNG Outpatient Count"]):
        
    aggregated = ablation_dfs[signal][["dropped_feature", "lag", "wis"]].groupby(
        ["dropped_feature", "lag"]
    ).agg(
        mean=('wis', 'mean'),  # pandas mean() already ignores NaN by default
        sem=('wis', lambda x: np.nanstd(x, ddof=1) / np.sqrt(np.sum(~np.isnan(x)))),
        quantile10=('wis', lambda x: np.nanquantile(x, 0.1)),
        quantile90=('wis', lambda x: np.nanquantile(x, 0.9)),
    ).reset_index()
            
    aggregated_all = all_dfs[signal].groupby(
        ["lag"]).agg(
            mean=('wis', 'mean'),
            sem=('wis', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
            quantile10=('wis', lambda x: np.quantile(x, 0.1)),         # Median (50th percentile)
            quantile90=('wis', lambda x: np.quantile(x, 0.9))
            ).reset_index()
    
    ax1 = plt.subplot(1, 2, idx+1)
    
    for dropped_feature in aggregated["dropped_feature"].unique():
        subdf = aggregated.loc[aggregated["dropped_feature"] == dropped_feature]
        # ax1.plot(subdf["lag"], subdf["mean"], label="-%s"%dropped_feature, linewidth=3.0)
        # ax1.fill_between(subdf["lag"], 
        #                  subdf["quantile10"], 
        #                  subdf["quantile90"], alpha=0.1)
        ax1.errorbar(
            subdf["lag"], subdf["mean"], 
            yerr=subdf["sem"], 
            label="%s" % label_map[dropped_feature],
            linewidth=3.0, 
            capsize=4,   # adds little caps at ends of error bars
            color = color_map[dropped_feature]
        )
    # ax1.plot(aggregated_all["lag"], aggregated_all["mean"], label="all", linewidth=3.0)
    # ax1.fill_between(aggregated_all["lag"], 
    #                  aggregated_all["quantile10"], 
    #                  aggregated_all["quantile90"], alpha=0.1)
    ax1.errorbar(aggregated_all["lag"], 
                 aggregated_all["mean"], 
                 yerr = aggregated_all["sem"],
                 label="All features included", linewidth=3.0, color = "black")
 

    ax1.grid(True)
    ax1.set_xlabel("Lag (Days)", fontsize=30)
    if idx == 0:
        ax1.set_ylabel("WIS", fontsize=30)
    ax1.set_xlim((0, 14))
    ax1.set_ylim((0, 0.3))
    ax1.set_xticks(np.arange(0, 15, 1))
    ax1.tick_params(axis='both', labelsize=20)
    
    ax2 = ax1.twinx()
    ax2.plot([],[])
    ax2.set_yticks(list(map(re_to_wis, yticks)))
    ax2.set_yticklabels([str(x)+"%" for x in yticks])
    ax2.set_ylim((ax1.get_ylim()[0], ax1.get_ylim()[1]))
    ax2.tick_params(axis='both', labelsize=18)  
    ax2.grid(True)
    # ax1.legend(bbox_to_anchor= (1.1, -0.3), fontsize=20)
    if signal == "CHNG Outpatient Count":
        ax1.set_title("Insurance claims\n%s"%(extra_info[signal]), fontsize=30, pad=13)
    else:
        ax1.set_title("%s\n%s"%(signal, extra_info[signal]), fontsize=30, pad=13)
    if idx == 1:
        ax2.set_ylabel('Absolute Relative Error', fontsize=25, rotation=270, labelpad=30)
labels_handles = {
  label: handle for ax in fig.axes for handle, label in zip(*ax.get_legend_handles_labels())
}

labels = ["All features included"] + [label_map.get(x, x) for x in aggregated["dropped_feature"].unique()]
handles = [labels_handles[x] for x in labels]
fig.legend(
    handles,
    labels,   
    # labels_handles.values(),
    # labels_handles.keys(),
  loc = "upper center",
  bbox_to_anchor = (0.5, 0),
  bbox_transform = plt.gcf().transFigure,
  ncol = 2,
  fontsize=20
)
plt.tight_layout()
plt.suptitle("Count Forecasting", fontsize=35, y=1.08)



####################################
### Ablation study for covid daily fractions
####################################
yticks = [0, 10, 20, 50]
plt.style.use('default')
fig = plt.figure(figsize=(15, 6))
for idx, signal in enumerate(["Insurance claims", "Antigen tests"]):
        
    aggregated = ablation_dfs[signal][["dropped_feature", "lag", "wis"]].groupby(
        ["dropped_feature", "lag"]
    ).agg(
        mean=('wis', 'mean'),  # pandas mean() already ignores NaN by default
        sem=('wis', lambda x: np.nanstd(x, ddof=1) / np.sqrt(np.sum(~np.isnan(x)))),
        quantile10=('wis', lambda x: np.nanquantile(x, 0.1)),
        quantile90=('wis', lambda x: np.nanquantile(x, 0.9)),
    ).reset_index()
            
    aggregated_all = all_dfs[signal].groupby(
        ["lag"]).agg(
            mean=('wis', 'mean'),
            sem=('wis', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
            quantile10=('wis', lambda x: np.quantile(x, 0.1)),         # Median (50th percentile)
            quantile90=('wis', lambda x: np.quantile(x, 0.9))
            ).reset_index()
    
    ax1 = plt.subplot(1, 2, idx+1)
    
    for dropped_feature in aggregated["dropped_feature"].unique():
        subdf = aggregated.loc[aggregated["dropped_feature"] == dropped_feature]
        # ax1.plot(subdf["lag"], subdf["mean"], label="-%s"%dropped_feature, linewidth=3.0)
        # ax1.fill_between(subdf["lag"], 
        #                  subdf["quantile10"], 
        #                  subdf["quantile90"], alpha=0.1)
        ax1.errorbar(
            subdf["lag"], subdf["mean"], 
            yerr=subdf["sem"], 
            label="%s" % label_map[dropped_feature],
            linewidth=3.0, 
            capsize=4,   # adds little caps at ends of error bars
            color = color_map[dropped_feature]
        )
    # ax1.plot(aggregated_all["lag"], aggregated_all["mean"], label="all", linewidth=3.0)
    # ax1.fill_between(aggregated_all["lag"], 
    #                  aggregated_all["quantile10"], 
    #                  aggregated_all["quantile90"], alpha=0.1)
    ax1.errorbar(aggregated_all["lag"], 
                 aggregated_all["mean"], 
                 yerr = aggregated_all["sem"],
                 label="All features included", linewidth=3.0, color="black")
 

    ax1.grid(True)
    ax1.set_xlabel("Lag (Days)", fontsize=30)
    if idx == 0:
        ax1.set_ylabel("WIS", fontsize=30)
    ax1.set_xlim((0, 14))
    ax1.set_ylim((0, 0.25))
    ax1.set_xticks(np.arange(0, 15, 1))
    ax1.tick_params(axis='both', labelsize=20)
    
    ax2 = ax1.twinx()
    ax2.plot([],[])
    ax2.set_yticks(list(map(re_to_wis, yticks)))
    ax2.set_yticklabels([str(x)+"%" for x in yticks])
    ax2.set_ylim((ax1.get_ylim()[0], ax1.get_ylim()[1]))
    ax2.tick_params(axis='both', labelsize=18)  
    ax2.grid(True)
    # ax1.legend(bbox_to_anchor= (1.1, -0.3), fontsize=20)
    ax1.set_title("%s\n%s"%(signal, extra_info[signal]), fontsize=30, pad=13)
    plt.grid(True)
    if idx == 1:
        ax2.set_ylabel('Absolute Relative Error', fontsize=25, rotation=270, labelpad=30)
labels_handles = {
  label: handle for ax in fig.axes for handle, label in zip(*ax.get_legend_handles_labels())
}

labels = ["All features included"] + [label_map.get(x, x) for x in aggregated["dropped_feature"].unique()]
handles = [labels_handles[x] for x in labels]
fig.legend(
    handles,
    labels,   
    # labels_handles.values(),
    # labels_handles.keys(),
  loc = "upper center",
  bbox_to_anchor = (0.5, 0),
  bbox_transform = plt.gcf().transFigure,
  ncol = 2,
  fontsize=20
)
plt.tight_layout()
plt.suptitle("Fraction Forecasting", fontsize=35, y=1.08)



####################################
### Ablation study for covid weekly counts
####################################
yticks = [0, 10, 20, 50]
plt.style.use('default')
fig = plt.figure(figsize=(15, 6))
for idx, signal in enumerate(["Dengue fever cases", "ILI cases"]):
        
    aggregated = ablation_dfs[signal][["dropped_feature", "lag", "wis"]].groupby(
        ["dropped_feature", "lag"]
    ).agg(
        mean=('wis', 'mean'),  # pandas mean() already ignores NaN by default
        sem=('wis', lambda x: np.nanstd(x, ddof=1) / np.sqrt(np.sum(~np.isnan(x)))),
        quantile10=('wis', lambda x: np.nanquantile(x, 0.1)),
        quantile90=('wis', lambda x: np.nanquantile(x, 0.9)),
    ).reset_index()
            
    aggregated_all = all_dfs[signal].groupby(
        ["lag"]).agg(
            mean=('wis', 'mean'),
            sem=('wis', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
            quantile10=('wis', lambda x: np.quantile(x, 0.1)),         # Median (50th percentile)
            quantile90=('wis', lambda x: np.quantile(x, 0.9))
            ).reset_index()
    
    ax1 = plt.subplot(1, 2, idx+1)
    
    for dropped_feature in aggregated["dropped_feature"].unique():
        subdf = aggregated.loc[aggregated["dropped_feature"] == dropped_feature]
        # ax1.plot(subdf["lag"], subdf["mean"], label="-%s"%dropped_feature, linewidth=3.0)
        # ax1.fill_between(subdf["lag"], 
        #                  subdf["quantile10"], 
        #                  subdf["quantile90"], alpha=0.1)
        ax1.errorbar(
            subdf["lag"] // 7, subdf["mean"], 
            yerr=subdf["sem"], 
            label="%s" % label_map[dropped_feature],
            linewidth=3.0, 
            capsize=4,   # adds little caps at ends of error bars
            color=color_map[dropped_feature]
        )
    # ax1.plot(aggregated_all["lag"], aggregated_all["mean"], label="all", linewidth=3.0)
    # ax1.fill_between(aggregated_all["lag"], 
    #                  aggregated_all["quantile10"], 
    #                  aggregated_all["quantile90"], alpha=0.1)
    ax1.errorbar(aggregated_all["lag"] // 7, 
                 aggregated_all["mean"], 
                 yerr = aggregated_all["sem"],
                 label="All features included", linewidth=3.0, color="black")
 

    ax1.grid(True)
    ax1.set_xlabel("Lag (Weeks)", fontsize=30)
    if idx == 0:
        ax1.set_ylabel("WIS", fontsize=30)
    ax1.set_xlim((0, 9))
    ax1.set_ylim((0, 0.25))
    ax1.set_xticks(np.arange(0, 9, 1))
    ax1.tick_params(axis='both', labelsize=20)
    
    ax2 = ax1.twinx()
    ax2.plot([],[])
    ax2.set_yticks(list(map(re_to_wis, yticks)))
    ax2.set_yticklabels([str(x)+"%" for x in yticks])
    ax2.set_ylim((ax1.get_ylim()[0], ax1.get_ylim()[1]))
    ax2.tick_params(axis='both', labelsize=18)  
    ax2.grid(True)
    # ax1.legend(bbox_to_anchor= (1.1, -0.3), fontsize=20)
    ax1.set_title("%s\n%s"%(signal,extra_info[signal]), fontsize=30, pad=13)
    plt.grid(True)
    if idx == 1:
        ax2.set_ylabel('Absolute Relative Error', fontsize=25, rotation=270, labelpad=30)
labels_handles = {
  label: handle for ax in fig.axes for handle, label in zip(*ax.get_legend_handles_labels())
}

labels = ["All features included"] + [label_map.get(x, x) for x in aggregated["dropped_feature"].unique()]
handles = [labels_handles[x] for x in labels]
fig.legend(
    handles,
    labels,   
    # labels_handles.values(),
    # labels_handles.keys(),
  loc = "upper center",
  bbox_to_anchor = (0.5, 0),
  bbox_transform = plt.gcf().transFigure,
  ncol = 2,
  fontsize=20
)
plt.tight_layout()
plt.suptitle("Count Forecasting", fontsize=35, y=1.08)


