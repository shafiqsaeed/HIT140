# 02_eda.py
# Exploratory Data Analysis – figures and simple tables
# Rules: matplotlib only, one plot per figure

import os
import pandas as pd
import matplotlib.pyplot as plt

IN_MERGED = "outputs/merged.csv"
OUTDIR = "figs"
os.makedirs(OUTDIR, exist_ok=True)

def main():
    df = pd.read_csv(IN_MERGED)

    # Histograms
    plt.figure()
    plt.hist(df["bat_landing_to_food"].dropna(), bins=30)
    plt.xlabel("Latency (s)"); plt.ylabel("Frequency"); plt.title("Latency distribution")
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "fig01_hist_latency.png"), dpi=200); plt.close()

    plt.figure()
    plt.hist(df["log_latency"].dropna(), bins=30)
    plt.xlabel("log(1+Latency)"); plt.ylabel("Frequency"); plt.title("Log-latency distribution")
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "fig02_hist_log_latency.png"), dpi=200); plt.close()

    plt.figure()
    plt.hist(df["seconds_after_rat_arrival"].dropna(), bins=30)
    plt.xlabel("Seconds after rat arrival"); plt.ylabel("Frequency"); plt.title("Seconds after rat arrival")
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "fig03_hist_sec_after_rat.png"), dpi=200); plt.close()

    # Proportions
    risk_prop = df["risk"].value_counts(normalize=True).sort_index()
    plt.figure()
    plt.bar(risk_prop.index.astype(str), risk_prop.values)
    plt.xlabel("Risk (0=avoid, 1=risk)"); plt.ylabel("Proportion"); plt.title("Risk proportions")
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "fig04_bar_risk_prop.png"), dpi=200); plt.close()

    reward_prop = df["reward"].value_counts(normalize=True).sort_index()
    plt.figure()
    plt.bar(reward_prop.index.astype(str), reward_prop.values)
    plt.xlabel("Reward (0/1)"); plt.ylabel("Proportion"); plt.title("Reward proportions")
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "fig05_bar_reward_prop.png"), dpi=200); plt.close()

    # Scatter latency vs seconds after rat
    plt.figure()
    plt.scatter(df["seconds_after_rat_arrival"], df["bat_landing_to_food"], s=10, alpha=0.6)
    plt.xlabel("Seconds after rat arrival"); plt.ylabel("Latency (s)"); plt.title("Latency vs seconds-after-rat")
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "fig06_scatter_sec_vs_latency.png"), dpi=200); plt.close()

    # Box latency by risk
    data0 = df.loc[df["risk"]==0, "bat_landing_to_food"].dropna()
    data1 = df.loc[df["risk"]==1, "bat_landing_to_food"].dropna()
    plt.figure()
    plt.boxplot([data0, data1], labels=["Avoidance (0)", "Risk-taking (1)"], showfliers=False)
    plt.ylabel("Latency (s)"); plt.title("Latency by risk group")
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "fig07_box_latency_by_risk.png"), dpi=200); plt.close()

    # Reward rate by risk
    reward_rate = df.groupby("risk")["reward"].mean()
    plt.figure()
    plt.bar(reward_rate.index.astype(str), reward_rate.values)
    plt.xlabel("Risk group"); plt.ylabel("Reward rate"); plt.title("Reward rate by risk group")
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "fig08_bar_reward_rate_by_risk.png"), dpi=200); plt.close()

    # Mean latency vs time of night
    hour_col = "hours_after_sunset_x" if "hours_after_sunset_x" in df.columns else "hours_after_sunset"
    df2 = df.copy(); df2["hour_bin"] = df2[hour_col].round()
    latency_by_hour = df2.groupby("hour_bin", dropna=True)["bat_landing_to_food"].mean()
    plt.figure()
    plt.plot(latency_by_hour.index, latency_by_hour.values, marker="o")
    plt.xlabel("Hours after sunset (binned)"); plt.ylabel("Mean latency (s)"); plt.title("Mean latency vs time of night")
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "fig09_line_latency_vs_night.png"), dpi=200); plt.close()

    print("EDA figures saved to:", OUTDIR)

if __name__ == "__main__":
    main()
