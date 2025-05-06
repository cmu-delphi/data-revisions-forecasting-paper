#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregated evl results over lags per location

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

from constants import (signals, data_dir, fig_dir, taus,
                        filtered_states)
from _utils_ import (read_chng_outpatient_result, read_ma_dph_result, read_quidel_result,
                      read_chng_outpatient_count_result, re_to_wis)

### Read results
dfs = {}
dfs["COVID-19 cases in MA"] = read_experimental_results(madph_config, "DelphiRF")
dfs["CHNG Outpatient Count"] = read_experimental_results(chng_count_config, "DelphiRF")
dfs["Insurance claims"] = read_experimental_results(chng_fraction_config, "DelphiRF")
dfs["Antigen tests"] = read_experimental_results(quidel_config, "DelphiRF")



plt.style.use('default')
for lag in [3, 4, 5, 6, 7, 14, 21, 28]: # base lag
    print(lag)
    output = '../projection_results/wis_vs_wis/for_all_states_start_from_lag%d.pdf'%lag
    with PdfPages(os.path.join("./", output)) as pdf:
        for mid_lag in [7, 14, 30]:
            if lag > mid_lag:
                continue
            #state_lists = df["geo_value"].unique()
            fig = plt.figure(figsize=(60, 35))
            for i in range(len(map_list)):
                state = map_list[i]
                if state == '':
                    continue
                dfa = results.loc[(results["geo"] == state) & (results["ref_lag"] == 60) & (results["lag"] == mid_lag)]
                dfb = results.loc[(results["geo"] == state) & (results["ref_lag"] == mid_lag) & (results["lag"] == lag)]
                
                dfab = dfa.merge(dfb, on=["time_value"], suffixes=["_60", "_%d"%mid_lag])
                
                try:
                    plt.subplot(7, 11, i+1)
                    alphas = (dfab["time_value"] - dfab["time_value"].min()).dt.days
                    plt.plot(np.arange(0, 1.5, 0.01), np.arange(0, 1.5, 0.01), linestyle="--", color="r")
                    plt.scatter(dfab["wis_60"], dfab["wis_%d"%mid_lag], alpha=alphas.values/alphas.max())
                    plt.xlabel("Lag %d -> Target Lag = 60"%mid_lag, fontsize=20)
                    plt.ylabel("Lag %d -> Target Lag = %d"%(lag, mid_lag), fontsize=20) 
                    cor_s = spearmanr(dfab["wis_60"],dfab["wis_%d"%mid_lag])
                    cor_p = pearsonr(dfab["wis_60"],dfab["wis_%d"%mid_lag])
                    
                    plt.title("%s\nSpearman Cor=%.6f\nPearson Cor=%.6f"%(state.upper(), cor_s[0], cor_p[0]), fontsize=25)
                    plt.ylim(0, 1.5)
                    plt.xlim(0, 1.5)
                    plt.grid()
                except: 
                    pass
            plt.suptitle("WIS Score(log scale)", fontsize=35, y = 1.05)
            plt.tight_layout()
                               
                            
            pdf.savefig(fig)
            plt.close()  
            