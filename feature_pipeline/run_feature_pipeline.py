"""
Runs every hour (via GitHub Actions):
1. Fetch current raw readings from Open-Meteo + AQICN
2. Store raw readings in the feature store
3. Recompute the feature row for "now" using recent history
4. Check for hazardous AQI and fire an alert if needed
"""
import sys
import os
from datetime import datetime, timedelta, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CITY_NAME, HAZARD_AQI_THRESHOLD
from feature_pipeline.fetch_data import fetch_open_meteo_current, fetch_aqicn_current
from feature_pipeline.feature_engineering import build_feature_frame
from db.db_utils import upsert_raw_reading, upsert_feature_row, get_raw_readings
from alerts.notifier import send_alert
import pandas as pd


def store_current_raw_readings():
    om = fetch_open_meteo_current()
    if om and om.get("ts"):
        upsert_raw_reading({**om, "city": CITY_NAME})

    aq = fetch_aqicn_current()
    if aq and aq.get("ts"):
        upsert_raw_reading({**aq, "city": CITY_NAME})

    # Use Open-Meteo as the primary series for feature engineering since it
    # has full historical continuity; AQICN readings serve as a cross-check.
    return om


def recompute_recent_features(lookback_hours: int = 24 * 10):
    """Pull recent history from Open-Meteo-sourced raw readings and recompute
    features for the most recent rows (lags/rolling windows need history)."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=lookback_hours)

    raw = get_raw_readings(CITY_NAME, start=start, end=end)
    raw = raw[raw["source"] == "open_meteo"].copy()
    if raw.empty:
        print("No raw history yet — run backfill first.")
        return None

    feature_df = build_feature_frame(raw, CITY_NAME)

    # Only upsert the most recent row(s) — older rows are already stored,
    # and forward-shifted targets for the very latest rows will be NULL
    # until future data arrives (expected and fine).
    latest_rows = feature_df.tail(3)
    for _, row in latest_rows.iterrows():
        row_dict = row.where(pd.notnull(row), None).to_dict()
        upsert_feature_row(row_dict)

    return feature_df.tail(1)


def check_and_alert(latest_row):
    if latest_row is None or latest_row.empty:
        return
    current_aqi = latest_row.iloc[0].get("target_aqi")
    if current_aqi is not None and current_aqi >= HAZARD_AQI_THRESHOLD:
        send_alert(
            f"⚠️ Hazardous AQI Alert for {CITY_NAME}: "
            f"current AQI is {current_aqi:.0f} (threshold: {HAZARD_AQI_THRESHOLD:.0f})"
        )


def main():
    print(f"[{datetime.now(timezone.utc)}] Running feature pipeline for {CITY_NAME}...")
    store_current_raw_readings()
    latest = recompute_recent_features()
    check_and_alert(latest)
    print("Done.")


if __name__ == "__main__":
    main()
