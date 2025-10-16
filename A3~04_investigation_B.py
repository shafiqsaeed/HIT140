# 04_investigation_B.py
# Investigation B: seasonal tests, moderation models, landing-rate GLMs (Poisson & NegBin)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

IN_MERGED = "outputs/merged.csv"
IN_D2 = "outputs/dataset2_clean.csv"
OUTDIR = "outputs"
FIGDIR = "figs"
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

def main():
    df = pd.read_csv(IN_MERGED)
    d2 = pd.read_csv(IN_D2)

    hour_col = "hours_after_sunset_x" if "hours_after_sunset_x" in df.columns else "hours_after_sunset"

    # Seasonal summaries & tests
    seasonal_summary = df.groupby("season")[["log_latency","risk","reward","rat_minutes","rat_arrival_number"]].agg(["mean","std","count"])
    seasonal_summary.to_csv(os.path.join(OUTDIR, "seasonal_summary.csv"))

    groups = [g.dropna().values for _, g in df.groupby("season")["log_latency"]]
    kw_stat, kw_p = stats.kruskal(*groups) if len(groups)>=2 else (np.nan, np.nan)

    ct_risk_season = pd.crosstab(df["season"], df["risk"])
    chi2_rs, chi_p_rs, *_ = stats.chi2_contingency(ct_risk_season)

    ct_rew_season = pd.crosstab(df["season"], df["reward"])
    chi2_rw, chi_p_rw, *_ = stats.chi2_contingency(ct_rew_season)

    groups_rm = [g.dropna().values for _, g in df.groupby("season")["rat_minutes"]]
    kw_rm, kw_rm_p = stats.kruskal(*groups_rm) if len(groups_rm)>=2 else (np.nan, np.nan)

    # Moderation models
    logit_int = smf.logit(f"risk ~ rat_minutes * C(season) + rat_arrival_number + {hour_col}", data=df).fit(disp=False)
    with open(os.path.join(OUTDIR, "logit_interaction_summary.txt"), "w") as f:
        f.write(logit_int.summary().as_text())

    ols_int = smf.ols(f"log_latency ~ rat_minutes * C(season) + rat_arrival_number + {hour_col}", data=df).fit(cov_type="HC3")
    with open(os.path.join(OUTDIR, "ols_interaction_summary.txt"), "w") as f:
        f.write(ols_int.summary().as_text())

    # Reward rate by season (Fig 16 style)
    reward_rate_season = df.groupby("season")["reward"].mean()
    plt.figure(); plt.bar(reward_rate_season.index.astype(str), reward_rate_season.values)
    plt.xlabel("Season"); plt.ylabel("Reward rate"); plt.title("Reward rate by season")
    plt.tight_layout(); plt.savefig(os.path.join(FIGDIR, "fig16_bar_reward_rate_by_season.png"), dpi=200); plt.close()

    # Landing rate models (dataset2)
    # If no season in d2, map from month
    if "season" not in d2.columns:
        season_map = {12:"Summer",1:"Summer",2:"Summer",3:"Autumn",4:"Autumn",5:"Winter",6:"Winter",7:"Winter",8:"Spring",9:"Spring",10:"Spring",11:"Summer"}
        d2["season"] = d2["month"].map(season_map)

    pois = smf.glm("bat_landing_number ~ rat_present * C(season) + rat_minutes + hours_after_sunset",
                   data=d2, family=sm.families.Poisson()).fit()
    with open(os.path.join(OUTDIR, "glm_poisson.txt"), "w") as f:
        f.write(pois.summary().as_text())

    mean_y = d2["bat_landing_number"].mean()
    var_y = d2["bat_landing_number"].var()
    overdisp_ratio = var_y / mean_y if mean_y > 0 else np.nan

    nb = smf.glm("bat_landing_number ~ rat_present * C(season) + rat_minutes + hours_after_sunset",
                 data=d2, family=sm.families.NegativeBinomial()).fit()
    with open(os.path.join(OUTDIR, "glm_negbin.txt"), "w") as f:
        f.write(nb.summary().as_text())

    # Bar charts for mean landings by season and rat presence (Fig 17/18 style)
    grp = d2.groupby(["season","rat_present"])["bat_landing_number"].mean().unstack()
    # Rats ABSENT (0)
    plt.figure()
    vals0 = grp.get(0, pd.Series(index=grp.index)).reindex(grp.index, fill_value=np.nan).values
    plt.bar(grp.index.astype(str), vals0)
    plt.xlabel("Season"); plt.ylabel("Mean bat landings"); plt.title("Mean bat landings when rats ABSENT (0) by season")
    plt.tight_layout(); plt.savefig(os.path.join(FIGDIR, "fig17_bar_landings_rats_absent_by_season.png"), dpi=200); plt.close()
    # Rats PRESENT (1)
    if 1 in grp.columns:
        plt.figure()
        plt.bar(grp.index.astype(str), grp[1].values)
        plt.xlabel("Season"); plt.ylabel("Mean bat landings"); plt.title("Mean bat landings when rats PRESENT (1) by season")
        plt.tight_layout(); plt.savefig(os.path.join(FIGDIR, "fig18_bar_landings_rats_present_by_season.png"), dpi=200); plt.close()

    # Save a short text summary for the report body
    with open(os.path.join(OUTDIR, "seasonality_tests.txt"), "w") as f:
        f.write(f"Kruskal–Wallis on log_latency by season: H={kw_stat:.3f}, p={kw_p:.3g}\n")
        f.write(f"Chi-square Risk x Season: chi2={chi2_rs:.3f}, p={chi_p_rs:.3g}\n")
        f.write(f"Chi-square Reward x Season: chi2={chi2_rw:.3f}, p={chi_p_rw:.3g}\n")
        f.write(f"Kruskal–Wallis rat_minutes by season: H={kw_rm:.3f}, p={kw_rm_p:.3g}\n")
        f.write(f"Overdispersion ratio (dataset2 landing counts): {overdisp_ratio:.2f}\n")

    print("Investigation B results saved to:", OUTDIR, "and figures to:", FIGDIR)

if __name__ == "__main__":
    main()
