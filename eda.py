# HIT140 Exploratory Data Analysis (EDA) Script
# Code by Mafuja Akhtar

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load cleaned merged data
merged = pd.read_csv("merged_clean.csv")

# Descriptive statistics
desc_stats = merged[["bat_landing_to_food","log_latency",
                     "seconds_after_rat_arrival","rat_minutes"]].describe()
print(desc_stats)

# Risk & reward proportions
risk_counts = merged["risk"].value_counts(normalize=True)
reward_counts = merged["reward"].value_counts(normalize=True)
print("Risk proportions:\n", risk_counts)
print("Reward proportions:\n", reward_counts)

# Save descriptive statistics
with open("descriptive_statistics.txt", "w") as f:
    f.write(desc_stats.to_string())
    f.write("\n\nRisk proportions:\n")
    f.write(risk_counts.to_string())
    f.write("\n\nReward proportions:\n")
    f.write(reward_counts.to_string())  
print("Descriptive statistics saved in 'descriptive_statistics.txt'.")
