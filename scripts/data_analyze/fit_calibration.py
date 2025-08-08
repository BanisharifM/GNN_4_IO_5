#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# 1) load your two‐column CSV
df = pd.read_csv("scripts/data_analyze/calibration_V5.csv")
y_pred = df["y_pred"].values.reshape(-1,1)   # shape (N,1)
y_true = df["y_true"].values                  # shape (N,)

# 2) fit a simple linear regressor with no intercept regularization
lr = LinearRegression(fit_intercept=True)
lr.fit(y_pred, y_true)

a = lr.coef_[0]
b = lr.intercept_
print(f"Calibration: y_true ≃ {a:.6f}·y_pred + {b:.6f}")

# 3) (optional) measure residual error
y_cal = lr.predict(y_pred)
mse = np.mean((y_cal - y_true)**2)
print(f"Calibration MSE: {mse:.6e}")

# 4) save coefficients for later
pd.DataFrame({"a":[a], "b":[b]}).to_csv(
    "scripts/data_analyze/calibration_coeffs.csv", index=False
)
print("→ coefficients saved to calibration_coeffs.csv")
