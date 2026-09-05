"""
Run this to sanity-check feature quality before assuming "more data" is the answer.
Specifically checks for outliers in derived features (common cause of models
performing worse than a naive baseline) and how each feature actually correlates
with the targets.
"""
import sys
import os

import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CITY_NAME
from db.db_utils import get_features
from training_pipeline.train import get_feature_cols, HORIZONS

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)


def main():
    df = get_features(CITY_NAME)
    print(f"Loaded {len(df)} rows.\n")

    print("=== Describe: key derived features ===")
    check_cols = ["aqi_change_rate", "aqi_lag_1h", "aqi_lag_24h", "aqi_rolling_mean_6h", "pm25"]
    print(df[check_cols].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T)
    print()

    print("=== Extreme values check (top 10 |aqi_change_rate|) ===")
    extreme = df.reindex(df["aqi_change_rate"].abs().sort_values(ascending=False).index)
    print(extreme[["ts", "aqi_lag_1h", "target_aqi", "aqi_change_rate"]].head(10))
    print()

    print("=== NaN / Inf check across base feature columns ===")
    base_check_cols = get_feature_cols("24h")  # base cols overlap across horizons; 24h's set covers them
    for col in base_check_cols:
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            print(f"  {col}: {n_nan} NaN")
    print()

    print("=== Correlation of each feature with each target (Pearson) ===")
    for horizon_name, target_col in HORIZONS.items():
        feature_cols = get_feature_cols(horizon_name)
        data = df.dropna(subset=feature_cols + [target_col])
        corrs = data[feature_cols + [target_col]].corr()[target_col].drop(target_col)
        corrs = corrs.sort_values(key=abs, ascending=False)
        print(f"\n--- {horizon_name} ---")
        print(corrs.head(10))


if __name__ == "__main__":
    main()