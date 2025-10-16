# 01_load_clean_merge.py
# Load, clean, feature engineer, and time-merge dataset1 (bat landings) with dataset2 (30-min windows)

import os
import pandas as pd
import numpy as np

# ---- Paths ----
DATA1 = "dataset1.csv"
DATA2 = "dataset2.csv"
OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)

def to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def main():
    # Load
    d1 = pd.read_csv(DATA1)
    d2 = pd.read_csv(DATA2)

    # Datetime parse
    for col in ["start_time","rat_period_start","rat_period_end","sunset_time"]:
        if col in d1.columns:
            d1[col] = pd.to_datetime(d1[col], dayfirst=True, errors="coerce")
    if "time" in d2.columns:
        d2["time"] = pd.to_datetime(d2["time"], dayfirst=True, errors="coerce")

    # Numerics
    to_num(d1, ["bat_landing_to_food","seconds_after_rat_arrival","risk","reward","month","hours_after_sunset","season"])
    to_num(d2, ["month","hours_after_sunset","bat_landing_number","food_availability","rat_minutes","rat_arrival_number"])

    # Feature engineering
    d1["log_latency"] = np.log1p(d1["bat_landing_to_food"].clip(lower=0))
    d2["rat_presence_ratio"] = d2["rat_minutes"] / 30.0
    d2["rat_present"] = (d2["rat_minutes"] > 0).astype(int)

    # Time-window merge (asof backward within 30 min), then ensure landing within 30-min window
    merged = pd.merge_asof(
        d1.sort_values("start_time"),
        d2.sort_values("time"),
        left_on="start_time", right_on="time",
        direction="backward", tolerance=pd.Timedelta("30min")
    )
    merged = merged[merged["start_time"] < (merged["time"] + pd.Timedelta("30min"))]

    # Save cleaned/merged
    d1.to_csv(os.path.join(OUTDIR, "dataset1_clean.csv"), index=False)
    d2.to_csv(os.path.join(OUTDIR, "dataset2_clean.csv"), index=False)
    merged.to_csv(os.path.join(OUTDIR, "merged.csv"), index=False)

    # Basic summaries
    desc = merged[["bat_landing_to_food","log_latency","seconds_after_rat_arrival","rat_minutes","rat_arrival_number"]].describe()
    desc.to_csv(os.path.join(OUTDIR, "descriptives.csv"))

    print("Saved cleaned data and merged table in:", OUTDIR)
    print("Shapes:", {"dataset1": d1.shape, "dataset2": d2.shape, "merged": merged.shape})

if __name__ == "__main__":
    main()
