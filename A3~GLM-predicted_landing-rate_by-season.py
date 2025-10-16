# 05_GLM_landing-rate_report.py
# Negative Binomial GLM for landing counts (handles overdispersion better than Poisson)

import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
d1 = pd.read_csv("dataset1.csv")
d2 = pd.read_csv("dataset2.csv")

# Parse times
for col in ["start_time","rat_period_start","rat_period_end","sunset_time"]:
    if col in d1.columns:
        d1[col] = pd.to_datetime(d1[col], dayfirst=True, errors="coerce")
if "time" in d2.columns:
    d2["time"] = pd.to_datetime(d2["time"], dayfirst=True, errors="coerce")

# Numeric conversions
def to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

to_num(d1, ["bat_landing_to_food","seconds_after_rat_arrival","risk","reward","month","hours_after_sunset","season"])
to_num(d2, ["month","hours_after_sunset","bat_landing_number","food_availability","rat_minutes","rat_arrival_number"])

# Features / helpers
d1["log_latency"] = np.log1p(d1["bat_landing_to_food"].clip(lower=0))
d2["rat_present"] = (d2["rat_minutes"] > 0).astype(int)

# Season mapping for dataset2 if needed
if "season" not in d2.columns:
    season_map = {12:"Summer",1:"Summer",2:"Summer",
                  3:"Autumn",4:"Autumn",
                  5:"Winter",6:"Winter",7:"Winter",
                  8:"Spring",9:"Spring",10:"Spring",11:"Summer"}
    d2["season"] = d2["month"].map(season_map)

# Output directory
os.makedirs("figs", exist_ok=True)
