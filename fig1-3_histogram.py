# HIT140 Figures 1-3: Histograms
# Code by Mafuja Akhtar

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load cleaned merged data
merged = pd.read_csv("merged_clean.csv")

# Output directory for plots
plots = "plots_all"
os.makedirs(plots, exist_ok=True)

# Fig1. Histogram — latency (seconds)
plt.figure(figsize=(12, 5))
plt.hist(merged["bat_landing_to_food"].dropna(), bins=30, color="orange")
plt.xlabel("Latency to approach food (s)")
plt.ylabel("Frequency")
plt.title("Distribution of latency (seconds)")
plt.savefig(f"{plots}/fig1_hist_latency.png", bbox_inches="tight", dpi=300)
plt.show()
plt.close()

# Fig2. Histogram — log-transformed latency
plt.figure(figsize=(12, 5))
plt.hist(merged["log_latency"].dropna(), bins=30, color="orange")
plt.xlabel("Log latency to approach food (log(1+seconds))")
plt.ylabel("Frequency")
plt.title("Distribution of log-transformed latency")
plt.savefig(f"{plots}/fig2_hist_log_latency.png", bbox_inches="tight", dpi=300)
plt.show()
plt.close()

# Fig3. Histogram — seconds since rat arrival
plt.figure(figsize=(12, 5))
plt.hist(merged["seconds_after_rat_arrival"].dropna(), bins=30, color="orange")
plt.xlabel("Seconds after rat arrival (s)")
plt.ylabel("Frequency")
plt.title("Distribution: seconds after rat arrival at landing")
plt.savefig(f"{plots}/fig3_hist_seconds_after_rat.png", bbox_inches="tight", dpi=300)
plt.show()
plt.close()

print("Figures 1-3 saved in 'plots_all' directory.")
