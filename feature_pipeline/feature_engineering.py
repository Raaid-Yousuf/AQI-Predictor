"""
Turns raw readings into model-ready features:
- time-based features (hour, day of week, month, weekend flag)
- derived features (lags, rolling means, AQI change rate)
- forward-shifted targets (t+24h, t+48h, t+72h) for the 3-day forecast
"""
import pandas as pd
import numpy as np


def add_time_features(df: pd.DataFrame, ts_col: str = "ts") -> pd.DataFrame:
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    df["hour"] = df[ts_col].dt.hour
    df["day_of_week"] = df[ts_col].dt.dayofweek
    df["day_of_month"] = df[ts_col].dt.day
    df["month"] = df[ts_col].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6])
    return df


def add_derived_features(df: pd.DataFrame, aqi_col: str = "us_aqi") -> pd.DataFrame:
    """Assumes df is sorted ascending by ts and is a single city's hourly series."""
    df = df.copy()
    df = df.sort_values("ts").reset_index(drop=True)

    df["aqi_lag_1h"] = df[aqi_col].shift(1)
    df["aqi_lag_24h"] = df[aqi_col].shift(24)
    df["aqi_rolling_mean_6h"] = df[aqi_col].rolling(window=6, min_periods=1).mean()

    # AQI change rate = (current - previous) / previous, guarding div-by-zero
    prev = df["aqi_lag_1h"].replace(0, np.nan)
    df["aqi_change_rate"] = (df[aqi_col] - df["aqi_lag_1h"]) / prev
    df["aqi_change_rate"] = df["aqi_change_rate"].fillna(0)

    return df


def add_forecast_targets(df: pd.DataFrame, aqi_col: str = "us_aqi") -> pd.DataFrame:
    """Shift AQI backward to create 'future value at this row's timestamp' targets."""
    df = df.copy()
    df["target_aqi"] = df[aqi_col]
    df["target_aqi_24h"] = df[aqi_col].shift(-24)
    df["target_aqi_48h"] = df[aqi_col].shift(-48)
    df["target_aqi_72h"] = df[aqi_col].shift(-72)
    return df


def build_feature_frame(raw_df: pd.DataFrame, city: str) -> pd.DataFrame:
    """
    Full pipeline: raw hourly readings -> model-ready feature rows.
    raw_df must have columns: ts, pm25, pm10, o3, no2, so2, co, us_aqi,
    temperature_c, humidity_pct, wind_speed_ms, pressure_hpa
    """
    df = add_time_features(raw_df)
    df = add_derived_features(df)
    df = add_forecast_targets(df)
    df["city"] = city

    cols = [
        "city", "ts", "hour", "day_of_week", "day_of_month", "month", "is_weekend",
        "pm25", "pm10", "o3", "no2", "so2", "co",
        "temperature_c", "humidity_pct", "wind_speed_ms", "pressure_hpa",
        "aqi_lag_1h", "aqi_lag_24h", "aqi_rolling_mean_6h", "aqi_change_rate",
        "target_aqi", "target_aqi_24h", "target_aqi_48h", "target_aqi_72h",
    ]
    return df[cols]
