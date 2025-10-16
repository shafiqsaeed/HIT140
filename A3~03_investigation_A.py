# 03_investigation_A.py
# Investigation A: tests & models — risk-taking (logit), latency (OLS), and key figures

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf

IN_MERGED = "outputs/merged.csv"
OUTDIR = "outputs"
FIGDIR = "figs"
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

def main():
    df = pd.read_csv(IN_MERGED)
    hour_col = "hours_after_sunset_x" if "hours_after_sunset_x" in df.columns else "hours_after_sunset"

    # Tests
    rho, p = stats.spearmanr(df["bat_landing_to_food"], df["seconds_after_rat_arrival"], nan_policy="omit")

    log0 = df.loc[df["risk"]==0, "log_latency"].dropna()
    log1 = df.loc[df["risk"]==1, "log_latency"].dropna()
    t_stat, t_p = stats.ttest_ind(log0, log1, equal_var=False)
    u_stat, u_p = stats.mannwhitneyu(log0, log1, alternative="two-sided")

    ct = pd.crosstab(df["risk"], df["reward"])
    chi2, chi_p, dof, exp = stats.chi2_contingency(ct)

    res_tbl = pd.DataFrame({
        "Test":["Spearman (latency vs sec_after_rat)","Welch t on log latency (risk 0 vs 1)","Mann-Whitney on log latency","Chi-square risk x reward"],
        "Statistic":[rho, t_stat, u_stat, chi2],
        "p_value":[p, t_p, u_p, chi_p]
    })
    res_tbl.to_csv(os.path.join(OUTDIR, "investigationA_tests.csv"), index=False)

    # Logistic regression: risk ~ rat activity + time + season
    logit = smf.logit(f"risk ~ seconds_after_rat_arrival + rat_minutes + rat_arrival_number + {hour_col} + C(season)", data=df).fit(disp=False)
    with open(os.path.join(OUTDIR, "logit_summary.txt"), "w") as f:
        f.write(logit.summary().as_text())

    # Plot Odds Ratios (Fig 10 style)
    params = logit.params[["seconds_after_rat_arrival","rat_minutes","rat_arrival_number"]]
    conf = logit.conf_int().loc[["seconds_after_rat_arrival","rat_minutes","rat_arrival_number"]]
    or_vals = np.exp(params); or_low = np.exp(conf[0]); or_hi = np.exp(conf[1])
    xnames = ["sec_after_rat","rat_minutes","rat_arrivals"]
    plt.figure()
    plt.errorbar(xnames, or_vals, yerr=[or_vals-or_low, or_hi-or_vals], fmt="o", capsize=5, linewidth=2)
    plt.axhline(1, linestyle="--")
    plt.ylabel("Odds Ratio (risk-taking)"); plt.title("Logistic Regression: Odds Ratios (95% CI)")
    plt.tight_layout(); plt.savefig(os.path.join(FIGDIR, "fig10_logit_or.png"), dpi=200); plt.close()

    # OLS (robust SE): log_latency ~ rat activity + time + season
    ols = smf.ols(f"log_latency ~ seconds_after_rat_arrival + rat_minutes + rat_arrival_number + {hour_col} + C(season)", data=df).fit(cov_type="HC3")
    with open(os.path.join(OUTDIR, "ols_summary.txt"), "w") as f:
        f.write(ols.summary().as_text())

    # Coefficient plot (Fig 11 style)
    coef = ols.params[["seconds_after_rat_arrival","rat_minutes","rat_arrival_number"]]
    conf = ols.conf_int().loc[["seconds_after_rat_arrival","rat_minutes","rat_arrival_number"]]
    plt.figure()
    plt.errorbar(xnames, coef, yerr=[coef-conf[0], conf[1]-coef], fmt="o", capsize=5, linewidth=2)
    plt.axhline(0, linestyle="--")
    plt.ylabel("Coefficient on log latency"); plt.title("Linear Regression: Key Coefficients (robust SE)")
    plt.tight_layout(); plt.savefig(os.path.join(FIGDIR, "fig11_ols_coef.png"), dpi=200); plt.close()

    print("Investigation A results saved to:", OUTDIR, "and figures to:", FIGDIR)

if __name__ == "__main__":
    main()
