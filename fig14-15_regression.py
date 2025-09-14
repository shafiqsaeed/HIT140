# HIT140 Figures 14-15: Regression Analyses
# Code by Mafuja Akhtar

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import os

# Load cleaned merged data
merged = pd.read_csv("merged_clean.csv")

# Output directory for plots
plots = "plots_all"
os.makedirs(plots, exist_ok=True)


# Fig14 — Logistic Regression Odds Ratios (95% CI)

# --- Logistic regression ---
logit_model = smf.logit(
    "risk ~ seconds_after_rat_arrival + rat_minutes + rat_arrival_number",
    data=merged
).fit(disp=False)

# --- Extract odds ratios and CI ---
params = logit_model.params
conf = logit_model.conf_int()
conf.columns = ["2.5%", "97.5%"]

or_table = np.exp(pd.concat([params, conf], axis=1))
or_table.columns = ["OR", "2.5%", "97.5%"]

# --- Plot ---
plt.figure(figsize=(12, 5))
xnames = ["sec_after_rat", "rat_minutes", "rat_arrivals"]
vals = or_table.loc[["seconds_after_rat_arrival","rat_minutes","rat_arrival_number"]]

plt.errorbar(xnames, vals["OR"], 
             yerr=[vals["OR"] - vals["2.5%"], vals["97.5%"] - vals["OR"]],
             fmt="o", capsize=5, linewidth=2)
plt.axhline(1, color="orange", linestyle="--")
plt.ylabel("Odds Ratio (risk-taking)")
plt.title("Logistic Regression: Odds Ratios (95% CI)")
plt.tight_layout()
plt.savefig(f"{plots}/fig14_logit_or.png", dpi=300)
plt.show()
plt.close()



# Fig15 — Linear Regression Coefficients (robust SE)

# --- OLS regression with robust SE ---
ols_model = smf.ols(
    "log_latency ~ seconds_after_rat_arrival + rat_minutes + rat_arrival_number",
    data=merged
).fit(cov_type="HC3")

# --- Extract coefficients and robust SE ---
coef = ols_model.params
se = ols_model.bse
conf = ols_model.conf_int()
conf.columns = ["2.5%", "97.5%"]

coef_table = pd.concat([coef, conf], axis=1)
coef_table.columns = ["coef", "2.5%", "97.5%"]

# --- Plot ---
plt.figure(figsize=(12, 5))
xnames = ["sec_after_rat", "rat_minutes", "rat_arrivals"]
vals = coef_table.loc[["seconds_after_rat_arrival","rat_minutes","rat_arrival_number"]]

plt.errorbar(xnames, vals["coef"], 
             yerr=[vals["coef"] - vals["2.5%"], vals["97.5%"] - vals["coef"]],
             fmt="o", capsize=5, linewidth=2)
plt.axhline(0, color="orange", linestyle="--")
plt.ylabel("Coefficient on log latency")
plt.title("Linear Regression: Key Coefficients (robust SE)")
plt.tight_layout()
plt.savefig(f"{plots}/fig15_ols_coef.png", dpi=300)
plt.show()
plt.close()


print("Figures 14-15 saved in 'plots_all' directory.")
