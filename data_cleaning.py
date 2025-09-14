# HIT140 Data Cleaning Script
# This script cleans and merges two datasets: dataset1.csv and dataset2.csv
# Code by Shafiq Rahman

import pandas as pd
import numpy as np

# Load data
d1 = pd.read_csv("dataset1.csv")
d2 = pd.read_csv("dataset2.csv")

# Parse datetime columns
for col in ["start_time", "rat_period_start", "rat_period_end", "sunset_time"]:
    d1[col] = pd.to_datetime(d1[col], dayfirst=True, errors="coerce")
d2["time"] = pd.to_datetime(d2["time"], dayfirst=True, errors="coerce")

# Convert numerics
num_cols_d1 = ["bat_landing_to_food", "seconds_after_rat_arrival",
               "risk", "reward", "month", "hours_after_sunset", "season"]
d1[num_cols_d1] = d1[num_cols_d1].apply(pd.to_numeric, errors="coerce")

num_cols_d2 = ["month", "hours_after_sunset", "bat_landing_number",
               "food_availability", "rat_minutes", "rat_arrival_number"]
d2[num_cols_d2] = d2[num_cols_d2].apply(pd.to_numeric, errors="coerce")

# Drop duplicates
d1 = d1.drop_duplicates()
d2 = d2.drop_duplicates()

# Feature engineering
d1["log_latency"] = np.log1p(d1["bat_landing_to_food"].clip(lower=0))
d2["rat_presence_ratio"] = d2["rat_minutes"] / 30

# Merge bat landings into 30-min observation windows
merged = pd.merge_asof(
    d1.sort_values("start_time"), 
    d2.sort_values("time"),
    left_on="start_time", right_on="time",
    direction="backward", tolerance=pd.Timedelta("30min")
)
merged = merged[merged["start_time"] < (merged["time"] + pd.Timedelta("30min"))]

# Drop rows with NaNs (after cleaning)
d1 = d1.dropna()
d2 = d2.dropna()
merged = merged.dropna()

# Save cleaned datasets
d1.to_csv("dataset1_clean.csv", index=False)
d2.to_csv("dataset2_clean.csv", index=False)
merged.to_csv("merged_clean.csv", index=False)

print("Cleaned data saved: dataset1_clean.csv, dataset2_clean.csv, merged_clean.csv")
