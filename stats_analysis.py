# HIT140 Statistical Analysis Script
# Spearman correlation and group comparisons
# Code by Mafuja Akhtar

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load cleaned merged data
merged = pd.read_csv("merged_clean.csv")

# Spearman correlation
rho, pval = stats.spearmanr(
    merged["bat_landing_to_food"], merged["seconds_after_rat_arrival"], nan_policy="omit"
)
print("Spearman rho:", rho, "p-value:", pval)

# T-test & Mann-Whitney (latency vs risk groups)
log0 = merged.loc[merged["risk"]==0, "log_latency"].dropna()
log1 = merged.loc[merged["risk"]==1, "log_latency"].dropna()
t_stat, t_p = stats.ttest_ind(log0, log1, equal_var=False)
u_stat, u_p = stats.mannwhitneyu(log0, log1, alternative="two-sided")
print("Welch t-test:", t_stat, "p=", t_p)
print("Mann-Whitney U:", u_stat, "p=", u_p)

# Save results
with open("statistical_analysis.txt", "w") as f:
    f.write(f"Spearman rho: {rho}, p-value: {pval}\n")
    f.write(f"Welch t-test: t={t_stat}, p={t_p}\n")
    f.write(f"Mann-Whitney U: U={u_stat}, p={u_p}\n")   
print("Statistical analysis results saved in 'statistical_analysis.txt'.")

