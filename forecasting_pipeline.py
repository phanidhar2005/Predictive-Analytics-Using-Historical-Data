"""
Predictive Analytics Using Historical Data
============================================
End-to-end pipeline demonstrating:
  1. Data cleaning & preprocessing
  2. Feature engineering (lags, rolling stats, calendar features)
  3. Two forecasting approaches:
       a) Machine-learning regression (Linear Regression + Random Forest)
       b) Classical time-series model (Holt-Winters triple exponential smoothing,
          implemented from scratch since statsmodels isn't installed in this env)
  4. Model evaluation (MAE, RMSE, MAPE, R^2)
  5. Visualization of actual vs. predicted values and future forecast

To use with YOUR OWN data: replace `load_data()` with a call that reads your
CSV (must have a date column and a numeric target column) and update
DATE_COL / TARGET_COL below.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)

DATE_COL = "date"
TARGET_COL = "sales"
FORECAST_HORIZON = 30  # days into the future to forecast


# ----------------------------------------------------------------------
# 1. LOAD DATA  (swap this out for pd.read_csv("your_file.csv") in practice)
# ----------------------------------------------------------------------
def load_data():
    n_days = 730  # 2 years of daily history
    dates = pd.date_range(start="2024-01-01", periods=n_days, freq="D")

    t = np.arange(n_days)
    trend = 200 + 0.35 * t                                   # gradual upward trend
    weekly_season = 15 * np.sin(2 * np.pi * t / 7)            # weekly pattern
    annual_season = 40 * np.sin(2 * np.pi * t / 365.25)       # yearly seasonality
    noise = np.random.normal(0, 10, n_days)

    sales = trend + weekly_season + annual_season + noise
    sales = np.maximum(sales, 0)

    df = pd.DataFrame({DATE_COL: dates, TARGET_COL: sales})

    # Inject realistic messiness: missing values + a couple of outliers,
    # so the preprocessing step below has real work to do.
    missing_idx = np.random.choice(n_days, size=15, replace=False)
    df.loc[missing_idx, TARGET_COL] = np.nan
    outlier_idx = np.random.choice(n_days, size=5, replace=False)
    df.loc[outlier_idx, TARGET_COL] *= 2.5

    return df


# ----------------------------------------------------------------------
# 2. CLEAN & PREPROCESS
# ----------------------------------------------------------------------
def clean_and_preprocess(df):
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    # Ensure a complete daily calendar (fills any missing dates)
    full_range = pd.date_range(df[DATE_COL].min(), df[DATE_COL].max(), freq="D")
    df = df.set_index(DATE_COL).reindex(full_range).rename_axis(DATE_COL).reset_index()

    # Outlier capping via IQR
    q1, q3 = df[TARGET_COL].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df[TARGET_COL] = df[TARGET_COL].clip(lower, upper)

    # Fill missing values via time-based interpolation
    df[TARGET_COL] = df[TARGET_COL].interpolate(method="linear", limit_direction="both")

    return df


# ----------------------------------------------------------------------
# 3a. FEATURE ENGINEERING FOR ML REGRESSION
# ----------------------------------------------------------------------
def build_features(df):
    df = df.copy()
    df["day_of_week"] = df[DATE_COL].dt.dayofweek
    df["day_of_year"] = df[DATE_COL].dt.dayofyear
    df["month"] = df[DATE_COL].dt.month
    df["t"] = np.arange(len(df))  # linear time index captures trend

    for lag in [1, 7, 14]:
        df[f"lag_{lag}"] = df[TARGET_COL].shift(lag)

    df["rolling_mean_7"] = df[TARGET_COL].shift(1).rolling(7).mean()
    df["rolling_std_7"] = df[TARGET_COL].shift(1).rolling(7).std()

    df = df.dropna().reset_index(drop=True)
    return df


# ----------------------------------------------------------------------
# 3b. HOLT-WINTERS TRIPLE EXPONENTIAL SMOOTHING (from scratch)
# ----------------------------------------------------------------------
def holt_winters(series, season_len=7, alpha=0.3, beta=0.1, gamma=0.2, horizon=30):
    n = len(series)
    season = [series[i] - np.mean(series[:season_len]) for i in range(season_len)]
    level = np.mean(series[:season_len])
    trend = (np.mean(series[season_len:2 * season_len]) - np.mean(series[:season_len])) / season_len

    fitted = []
    for i in range(n):
        s = season[i % season_len]
        fitted.append(level + trend + s)
        val = series[i]
        last_level = level
        level = alpha * (val - s) + (1 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1 - beta) * trend
        season[i % season_len] = gamma * (val - level) + (1 - gamma) * s

    forecast = []
    for h in range(1, horizon + 1):
        s = season[(n + h - 1) % season_len]
        forecast.append(level + h * trend + s)

    return np.array(fitted), np.array(forecast)


# ----------------------------------------------------------------------
# 4. EVALUATION
# ----------------------------------------------------------------------
def evaluate(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-6))) * 100
    r2 = r2_score(y_true, y_pred)
    print(f"\n--- {label} ---")
    print(f"MAE:  {mae:8.2f}")
    print(f"RMSE: {rmse:8.2f}")
    print(f"MAPE: {mape:8.2f}%")
    print(f"R^2:  {r2:8.3f}")
    return {"model": label, "MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


# ----------------------------------------------------------------------
# MAIN PIPELINE
# ----------------------------------------------------------------------
def main():
    raw = load_data()
    print(f"Raw data: {len(raw)} rows, {raw[TARGET_COL].isna().sum()} missing values")

    clean = clean_and_preprocess(raw)
    print(f"After cleaning: {clean[TARGET_COL].isna().sum()} missing values remain")

    feat = build_features(clean)

    # Time-based train/test split (never shuffle time series data)
    split_idx = len(feat) - FORECAST_HORIZON
    train, test = feat.iloc[:split_idx], feat.iloc[split_idx:]

    feature_cols = ["t", "day_of_week", "day_of_year", "month",
                     "lag_1", "lag_7", "lag_14", "rolling_mean_7", "rolling_std_7"]
    X_train, y_train = train[feature_cols], train[TARGET_COL]
    X_test, y_test = test[feature_cols], test[TARGET_COL]

    # --- Model 1: Linear Regression ---
    lr = LinearRegression().fit(X_train, y_train)
    lr_pred = lr.predict(X_test)

    # --- Model 2: Random Forest Regression ---
    rf = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    # --- Model 3: Holt-Winters time-series model ---
    hw_train_series = clean[TARGET_COL].values[:split_idx + (len(clean) - len(feat))]
    # align series length: use the same test window as clean data tail
    hw_full_series = clean[TARGET_COL].values
    hw_test_actual = hw_full_series[-FORECAST_HORIZON:]
    hw_train_only = hw_full_series[:-FORECAST_HORIZON]
    hw_fitted, hw_forecast = holt_winters(hw_train_only, season_len=7, horizon=FORECAST_HORIZON)

    # --- Evaluate all models on the held-out test window ---
    results = []
    results.append(evaluate(y_test.values, lr_pred, "Linear Regression"))
    results.append(evaluate(y_test.values, rf_pred, "Random Forest"))
    results.append(evaluate(hw_test_actual, hw_forecast, "Holt-Winters (time series)"))

    results_df = pd.DataFrame(results)
    print("\n=== Model Comparison ===")
    print(results_df.to_string(index=False))
    results_df.to_csv("/home/claude/forecast_project/model_comparison.csv", index=False)

    # ------------------------------------------------------------------
    # VISUALIZATIONS
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(13, 10))

    # Plot 1: full history + all model predictions on test window
    ax = axes[0]
    ax.plot(clean[DATE_COL], clean[TARGET_COL], label="Historical (cleaned)", color="#888888", linewidth=1)
    ax.plot(test[DATE_COL], y_test, label="Actual (test)", color="black", linewidth=2)
    ax.plot(test[DATE_COL], lr_pred, label="Linear Regression", linestyle="--")
    ax.plot(test[DATE_COL], rf_pred, label="Random Forest", linestyle="--")
    ax.plot(test[DATE_COL], hw_forecast, label="Holt-Winters", linestyle="--")
    ax.set_title("Historical Data & Model Predictions on Held-Out Test Window")
    ax.set_xlabel("Date")
    ax.set_ylabel(TARGET_COL)
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2: model accuracy comparison (RMSE / MAE bar chart)
    ax = axes[1]
    x = np.arange(len(results_df))
    width = 0.35
    ax.bar(x - width/2, results_df["MAE"], width, label="MAE")
    ax.bar(x + width/2, results_df["RMSE"], width, label="RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["model"], rotation=10)
    ax.set_title("Model Accuracy Comparison (lower is better)")
    ax.set_ylabel("Error")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("/home/claude/forecast_project/evaluation_plots.png", dpi=150)
    print("\nSaved evaluation_plots.png")

    # ------------------------------------------------------------------
    # FUTURE FORECAST (beyond available data) using the best model (Random Forest here)
    # ------------------------------------------------------------------
    best_model_name = results_df.sort_values("RMSE").iloc[0]["model"]
    print(f"\nBest model by RMSE: {best_model_name}")

    # Recursive forecast using Random Forest (works for any horizon beyond known data)
    history = clean.copy()
    future_dates = pd.date_range(history[DATE_COL].max() + pd.Timedelta(days=1), periods=FORECAST_HORIZON, freq="D")
    forecast_vals = []
    ext = history.copy()

    rf_full = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42)
    feat_full = build_features(clean)
    rf_full.fit(feat_full[feature_cols], feat_full[TARGET_COL])

    for d in future_dates:
        row = {
            DATE_COL: d,
            "t": len(ext),
            "day_of_week": d.dayofweek,
            "day_of_year": d.dayofyear,
            "month": d.month,
            "lag_1": ext[TARGET_COL].iloc[-1],
            "lag_7": ext[TARGET_COL].iloc[-7],
            "lag_14": ext[TARGET_COL].iloc[-14],
            "rolling_mean_7": ext[TARGET_COL].iloc[-7:].mean(),
            "rolling_std_7": ext[TARGET_COL].iloc[-7:].std(),
        }
        pred = rf_full.predict(pd.DataFrame([row])[feature_cols])[0]
        forecast_vals.append(pred)
        ext = pd.concat([ext, pd.DataFrame([{DATE_COL: d, TARGET_COL: pred}])], ignore_index=True)

    future_df = pd.DataFrame({DATE_COL: future_dates, "forecast": forecast_vals})
    future_df.to_csv("/home/claude/forecast_project/future_forecast.csv", index=False)

    fig2, ax2 = plt.subplots(figsize=(13, 5))
    ax2.plot(clean[DATE_COL].iloc[-120:], clean[TARGET_COL].iloc[-120:], label="Recent history", color="#444444")
    ax2.plot(future_df[DATE_COL], future_df["forecast"], label=f"{FORECAST_HORIZON}-day forecast", color="crimson", linewidth=2)
    ax2.axvline(clean[DATE_COL].max(), color="gray", linestyle=":", label="Forecast start")
    ax2.set_title(f"Future Forecast — Next {FORECAST_HORIZON} Days (Random Forest)")
    ax2.set_xlabel("Date")
    ax2.set_ylabel(TARGET_COL)
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("/home/claude/forecast_project/future_forecast.png", dpi=150)
    print("Saved future_forecast.png")

    return results_df, future_df


if __name__ == "__main__":
    main()
