# HIT140 Figures 12-13: Bar and Line Plots
# Code by Shafiq Rahman

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load cleaned merged data
merged = pd.read_csv("merged_clean.csv")

# Output directory for plots
plots = "plots_all"
os.makedirs(plots, exist_ok=True)

# Fig12. Bar — reward rate (proportion) by risk group
reward_rate = merged.groupby("risk")["reward"].mean()
plt.figure(figsize=(12, 5))
plt.bar(reward_rate.index.astype(str), reward_rate.values, color="orange")
plt.xlabel("Risk group (0/1)")
plt.ylabel("Proportion rewarded")
plt.title("Reward rate by risk group")
plt.savefig(f"{plots}/fig12_bar_reward_rate_by_risk.png", bbox_inches="tight", dpi=300)
plt.show()
plt.close()

# Fig13. Line — mean latency vs hours after sunset
merged2 = merged.copy()
if "hours_after_sunset_x" in merged2.columns:
    merged2["hour_bin"] = merged2["hours_after_sunset_x"].round()
else:
    merged2["hour_bin"] = merged2["hours_after_sunset"].round()
latency_by_hour = merged2.groupby("hour_bin", dropna=True)["bat_landing_to_food"].mean()

plt.figure(figsize=(12, 5))
plt.plot(latency_by_hour.index, latency_by_hour.values, marker="o", color="orange")
plt.xlabel("Hours after sunset (binned)")
plt.ylabel("Mean latency (s)")
plt.title("Mean latency vs time of night")
plt.savefig(f"{plots}/fig13_line_latency_vs_timeofnight.png", bbox_inches="tight", dpi=300)
plt.show()
plt.close()


print("Figures 12-13 saved in 'plots_all' directory.")
