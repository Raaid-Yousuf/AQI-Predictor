"""
Run once (and re-run periodically to extend history) to build a training dataset.
Usage:
    python backfill/backfill_historical.py --days 90
"""
import sys
import os
import argparse
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CITY_NAME
from feature_pipeline.fetch_data import fetch_open_meteo_historical
from feature_pipeline.feature_engineering import build_feature_frame
from db.db_utils import upsert_raw_reading, upsert_feature_row
import pandas as pd


def backfill(days: int, chunk_days: int = 30):
    """Open-Meteo archive is queried in chunks to keep requests reasonably sized."""
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=days)

    all_rows = []
    cursor = start_date
    while cursor < end_date:
        chunk_end = min(cursor + timedelta(days=chunk_days), end_date)
        print(f"Fetching {cursor} -> {chunk_end} ...")
        rows = fetch_open_meteo_historical(cursor.isoformat(), chunk_end.isoformat())
        all_rows.extend(rows)
        cursor = chunk_end + timedelta(days=1)

    if not all_rows:
        print("No data returned — check API limits or date range.")
        return

    raw_df = pd.DataFrame(all_rows)
    raw_df = raw_df.dropna(subset=["ts"])

    print(f"Storing {len(raw_df)} raw hourly readings...")
    for _, row in raw_df.iterrows():
        row_dict = row.where(pd.notnull(row), None).to_dict()
        upsert_raw_reading({**row_dict, "city": CITY_NAME})

    print("Computing features...")
    feature_df = build_feature_frame(raw_df, CITY_NAME)

    print(f"Storing {len(feature_df)} feature rows...")
    for _, row in feature_df.iterrows():
        row_dict = row.where(pd.notnull(row), None).to_dict()
        upsert_feature_row(row_dict)

    print("Backfill complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90, help="Days of history to backfill")
    args = parser.parse_args()
    backfill(args.days)
