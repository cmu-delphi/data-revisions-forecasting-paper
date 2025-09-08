import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

from _utils_ import (madph_config, chng_count_config,
                     dengue_config, ilicases_config,
                     chng_fraction_config, quidel_config,
                     re_to_wis, read_hyperparams_analysis_results,
                     read_experimental_results)



pdList = []
configs = [
    (madph_config, "madph", 0),
    (chng_count_config, "chng_outpatient_count", 1),
    (dengue_config, "dengue", 0),
    (ilicases_config, "ili", 0),
    (chng_fraction_config, "chng_outpatient_fraction", 1),
    (quidel_config, "quidel_fraction", 1),
]
for config, signal, lagpad in configs:
    df = read_experimental_results(config, "DelphiRF")
    df["lambda"] = 0.1
    df["gamma"] = 0.1
    df["lagpad"] = lagpad
    df["signal"] = signal
    pdList.append(df)
all_df = pd.concat(pdList, ignore_index=True)


source_to_label = {
    "madph": "COVID-19 cases in MA",
    "dengue": "Dengue fever cases",
    "ili": "ILI cases",
    "chng_outpatient_count": "Insurance claims (counts)",
    "chng_outpatient_fraction": "Insurance claims (fractions)",
    "quidel_fraction": "Antigen test  (fractions)",
}

hyperparams_dfs = {}
hyperparams_dfs["gamma"] = read_hyperparams_analysis_results("gamma")
hyperparams_dfs["lambda"] = read_hyperparams_analysis_results("lambda")
hyperparams_dfs["lagpad"] = read_hyperparams_analysis_results("lagpad")



extra_info = {
    "COVID-19 cases in MA": "(Daily, State, MA only)",
    "CHNG Outpatient Count": "(Daily, State, All states)",
    "Dengue fever cases": "(Weekly, State, PR only)",
    "ILI cases": "(Weekly, National)",
    "Insurance claims": "(Daily, State, All states)",
    "Antigen tests": "(Daily, State, All states)"   
}

palette = {
    0.01: "tab:blue",     # blue
    0.1: "tab:orange",    # orange
    1.0: "tab:green"      # green (instead of black)
}

lag_pad_palette = {
    0: "tab:blue",
    1: "tab:purple",
    2: "tab:orange",
    7: "tab:green",
    14: "tab:red"
}


signal_order = [source_to_label[k] for k in [
    "dengue",
    "ili",
    "madph",
    "chng_outpatient_count",
    "chng_outpatient_fraction",
    "quidel_fraction",
]]


param_cfg = {
    "lambda": {
        "title": r"Regularization strength ($\lambda$)",
        "legend_title":r"$\lambda$",
        "hue_order": [0.01, 0.1, 1.0],
        "palette": palette,
    },
    "gamma": {
        "title": r"Decay parameter ($\gamma$)",
        "legend_title":r"$\gamma$",
        "hue_order": [0.01, 0.1, 1.0],
        "palette": palette,
    },
    "lagpad": {
        "title": r"Lag padding ($c$)",
        "legend_title":r"$c$",
        "hue_order": [0, 1, 2, 7, 14],
        "palette": lag_pad_palette,
    },
}



# ---- put these near your imports/setup ----
# Build per-signal minimum available lag across your dataframes
def compute_min_lag_by_signal(*dfs):
    cat = pd.concat([df[["signal", "lag"]] for df in dfs], ignore_index=True)
    return cat.groupby("signal")["lag"].min().to_dict()

min_lag_by_signal = compute_min_lag_by_signal(all_df, *hyperparams_dfs.values())

def filter_by_lag_per_signal(df, lag, min_lag_map):
    """If lag==0, pick each signal's min available lag; else pick exact lag."""
    if lag > 0:
        return df.loc[df["lag"] == lag]
    # For the 'ℓ_min' panel: for each signal, choose its min lag
    ml = df["signal"].map(min_lag_map)
    return df.loc[df["lag"] == ml]

yticks_pct = [0, 10, 20, 30, 40, 50, 60]
lags_to_show = [0, 7, 14]

for param in ["lambda", "gamma", "lagpad"]:
    cfg = param_cfg[param]

    fig = plt.figure(figsize=(18, 4))
    handles, labels = None, None
    for idx, lag in enumerate(lags_to_show):
        ax1 = plt.subplot(1, 3, idx + 1)

        # --- Build plotting dataframe: tuned + baseline (using ℓ_min logic) ---
        tuned_src = hyperparams_dfs[param][["signal", "lag", param, "wis"]].dropna().copy()
        tuned = filter_by_lag_per_signal(tuned_src, lag, min_lag_by_signal)

        base_src = all_df.loc[:, ["signal", "lag", param, "wis"]].dropna().copy()
        base = filter_by_lag_per_signal(base_src, lag, min_lag_by_signal)

        plot_df = pd.concat([tuned, base], ignore_index=True)

        # map signal to pretty labels
        plot_df["signal"] = plot_df["signal"].map(source_to_label).fillna(plot_df["signal"])

        # ensure numeric hue for ordering
        if plot_df[param].dtype == object:
            plot_df[param] = pd.to_numeric(plot_df[param], errors="coerce")

        sns.boxplot(
            data=plot_df,
            x="signal", y="wis", hue=param,
            order=signal_order,
            showfliers=False,
            hue_order=cfg["hue_order"],
            palette=cfg["palette"],
            ax=ax1,
        )

        ax1.set_ylim([0, 0.5])
        ax1.tick_params(axis="x", labelsize=20, rotation=90)
        ax1.tick_params(axis="y", labelsize=25)

        # ---- Title: use ℓ_min label for the first panel ----
        if lag == 0:
            ax1.set_title(r"Lag = $l_\min$ days", fontsize=35)
        else:
            ax1.set_title(f"Lag = {lag} days", fontsize=35)

        ax1.set_xlabel("")

        if idx == 0:
            ax1.set_ylabel("WIS", fontsize=30)
        else:
            ax1.set_ylabel("")
            ax1.tick_params(labelleft=False)

        # if idx < len(lags_to_show) - 1 and ax1.legend_ is not None:
        #     ax1.legend_.remove()
        # elif idx == len(lags_to_show) - 1:
        #     ax1.legend(fontsize=20, title=cfg["legend_title"], title_fontsize=25)
        
        # Capture legend handles once
        if handles is None and ax1.get_legend() is not None:
            handles, labels = ax1.get_legend_handles_labels()

        # remove subplot legends
        if ax1.legend_ is not None:
            ax1.legend_.remove()

        ax1.grid(True)

        # Secondary y-axis
        ax2 = ax1.twinx()
        ax2.plot([], [])
        ax2.set_yticks(list(map(re_to_wis, yticks_pct)))
        ax2.set_yticklabels([f"{x}%" for x in yticks_pct])
        ax2.set_ylim(ax1.get_ylim())
        ax2.tick_params(axis="both", labelsize=20)

        if idx == len(lags_to_show) - 1:
            ax2.set_ylabel("Absolute Relative Error", fontsize=30, rotation=270, labelpad=30)
        else:
            ax2.set_ylabel("")
            ax2.tick_params(labelright=False)
    # Add shared legend below
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -1.2),
        ncol=len(cfg["hue_order"]),
        fontsize=20,
        title=cfg["legend_title"],
        title_fontsize=25,
        frameon=True
    )
    
    plt.tight_layout(rect=[0, 0.1, 1, 1])  # leave space at bottom for legend
    plt.suptitle(cfg["title"], fontsize=40, y=1.2)
    plt.show()
