# HIT140 Modeling Script
# Logistic regression for risk-taking
# Linear regression for log latency
# Code by Shafiq Rahman

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# Load cleaned merged data
merged = pd.read_csv("merged_clean.csv")

# Logistic regression: risk ~ rat activity
logit_model = smf.logit(
    "risk ~ seconds_after_rat_arrival + rat_minutes + rat_arrival_number + hours_after_sunset_x + C(season)",
    data=merged
).fit()
print(logit_model.summary())

# OLS regression: log latency ~ rat activity
ols_model = smf.ols(
    "log_latency ~ seconds_after_rat_arrival + rat_minutes + rat_arrival_number + hours_after_sunset_x + C(season)",
    data=merged
).fit(cov_type="HC3")
print(ols_model.summary())

# Save models
with open("logit_model_summary.txt", "w") as f:
    f.write(logit_model.summary().as_text())
with open("ols_model_summary.txt", "w") as f:
    f.write(ols_model.summary().as_text())
print("Model summaries saved: logit_model_summary.txt, ols_model_summary.txt")

