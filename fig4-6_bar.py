# HIT140 Figures 4-6: Bar Plots
# Code by Shfiq Rahman

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load cleaned merged data
merged = pd.read_csv("merged_clean.csv")

# Output directory for plots
plots = "plots_all"
os.makedirs(plots, exist_ok=True)

# Fig4. Bar — risk-taking vs avoidance (counts)
risk_counts = merged["risk"].value_counts().sort_index()
plt.figure(figsize=(12, 5))
plt.bar(risk_counts.index.astype(str), risk_counts.values, color="orange")
plt.xlabel("Risk (0=avoidance, 1=risk-taking)")
plt.ylabel("Count")
plt.title("Risk-taking vs avoidance counts")
plt.savefig(f"{plots}/fig4_bar_risk.png", bbox_inches="tight", dpi=300)
plt.show()
plt.close()

# Fig5. Bar — reward outcome (counts)
reward_counts = merged["reward"].value_counts().sort_index()
plt.figure(figsize=(12, 5))
plt.bar(reward_counts.index.astype(str), reward_counts.values, color="orange")
plt.xlabel("Reward (0=no, 1=yes)")
plt.ylabel("Count")
plt.title("Reward outcome counts")
plt.savefig(f"{plots}/fig5_bar_reward.png", bbox_inches="tight", dpi=300)
plt.show()
plt.close()

# Fig6. Bar — landings by season (if available)
if "season" in merged.columns:
    season_counts = merged["season"].value_counts()
    plt.figure(figsize=(12, 5))
    plt.bar(season_counts.index.astype(str), season_counts.values, color="orange")
    plt.xlabel("Season")
    plt.ylabel("Count of landings")
    plt.title("Landings by season")
    plt.savefig(f"{plots}/fig6_bar_season.png", bbox_inches="tight", dpi=300)
    plt.show()
    plt.close()


print("Figures 4-6 saved in 'plots_all' directory.")
