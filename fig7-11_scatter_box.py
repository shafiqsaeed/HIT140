# HIT140 Figures 7-11: Scatter and Box Plots
# Code by Mafuja Akhtar and Shafiq Rahman

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load cleaned merged data
merged = pd.read_csv("merged_clean.csv")

# Output directory for plots
plots = "plots_all"
os.makedirs(plots, exist_ok=True)

# Fig7. Scatter — latency vs seconds since rat arrival
plt.figure(figsize=(12, 5))
plt.scatter(merged["seconds_after_rat_arrival"], merged["bat_landing_to_food"], s=10, alpha=0.6, color="orange")
plt.xlabel("Seconds after rat arrival (s)")
plt.ylabel("Latency to approach food (s)")
plt.title("Latency vs seconds since rat arrival")
plt.savefig(f"{plots}/fig7_scatter_seconds_vs_latency.png", bbox_inches="tight", dpi=300)
plt.show()
plt.close()

# Fig8. Box — latency by risk group
data0 = merged.loc[merged["risk"]==0, "bat_landing_to_food"].dropna()
data1 = merged.loc[merged["risk"]==1, "bat_landing_to_food"].dropna()
plt.figure(figsize=(12, 5))
plt.boxplot([data0, data1], tick_labels=["Avoidance (0)", "Risk-taking (1)"], showfliers=False)
plt.ylabel("Latency (s)")
plt.title("Latency by risk group")
plt.savefig(f"{plots}/fig8_box_latency_by_risk.png", bbox_inches="tight", dpi=300)
plt.show()
plt.close()


# Fig9. Scatter — risk (jittered) vs rat minutes
rng = np.random.default_rng(1)
risk_j = merged["risk"].astype(float) + rng.uniform(-0.05, 0.05, size=len(merged))
plt.figure(figsize=(12, 5))
plt.scatter(merged["rat_minutes"], risk_j, s=10, alpha=0.5, color="orange")
plt.xlabel("Rat minutes in 30-min window")
plt.ylabel("Risk (0/1; jittered)")
plt.title("Risk-taking vs rat minutes")
plt.savefig(f"{plots}/fig9_scatter_risk_vs_ratminutes.png", bbox_inches="tight", dpi=300)
plt.show()
plt.close()

# Fig10. Box — log latency by reward outcome
data_nr = merged.loc[merged["reward"]==0, "log_latency"].dropna()
data_r  = merged.loc[merged["reward"]==1, "log_latency"].dropna()
plt.figure(figsize=(12, 5))
plt.boxplot([data_nr, data_r], tick_labels=["No reward", "Reward"], showfliers=False)
plt.ylabel("Log latency")
plt.title("Latency by reward outcome")
plt.savefig(f"{plots}/fig10_box_loglat_by_reward.png", bbox_inches="tight", dpi=300)
plt.show()
plt.close()


# Fig11. Scatter — latency vs rat minutes, split by risk
mask0 = merged["risk"]==0
mask1 = merged["risk"]==1
plt.figure(figsize=(12, 5))
plt.scatter(merged.loc[mask0, "rat_minutes"], merged.loc[mask0, "bat_landing_to_food"], s=10, alpha=0.6, label="Risk=0", color="purple")
plt.scatter(merged.loc[mask1, "rat_minutes"], merged.loc[mask1, "bat_landing_to_food"], s=10, alpha=0.6, label="Risk=1", color="orange")
plt.xlabel("Rat minutes (per 30-min period)")
plt.ylabel("Latency (s)")
plt.title("Latency vs rat minutes by risk group")
plt.legend()
plt.savefig(f"{plots}/fig11_scatter_latency_vs_ratmin_by_risk.png", bbox_inches="tight", dpi=300)
plt.show()
plt.close()


print("Figures 7-11 saved in 'plots_all' directory.")
