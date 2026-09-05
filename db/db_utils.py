"""
Thin Postgres helper layer. Acts as our lightweight "feature store" client.
Swap DATABASE_URL to point at any Postgres instance without changing callers.
"""
import pandas as pd
from sqlalchemy import create_engine, text
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATABASE_URL

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        if not DATABASE_URL:
            raise RuntimeError("DATABASE_URL is not set. Check your .env file.")
        _engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    return _engine


def upsert_raw_reading(row: dict):
    """Insert a raw reading, ignoring duplicates on (city, ts, source)."""
    engine = get_engine()
    cols = ", ".join(row.keys())
    placeholders = ", ".join(f":{k}" for k in row.keys())
    sql = f"""
        INSERT INTO raw_readings ({cols})
        VALUES ({placeholders})
        ON CONFLICT (city, ts, source) DO NOTHING
    """
    with engine.begin() as conn:
        conn.execute(text(sql), row)


def upsert_feature_row(row: dict):
    """Insert or update a computed feature row, keyed on (city, ts)."""
    engine = get_engine()
    cols = list(row.keys())
    insert_cols = ", ".join(cols)
    placeholders = ", ".join(f":{c}" for c in cols)
    update_cols = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c not in ("city", "ts"))
    sql = f"""
        INSERT INTO features ({insert_cols})
        VALUES ({placeholders})
        ON CONFLICT (city, ts) DO UPDATE SET {update_cols}
    """
    with engine.begin() as conn:
        conn.execute(text(sql), row)


def get_raw_readings(city: str, start=None, end=None) -> pd.DataFrame:
    engine = get_engine()
    sql = "SELECT * FROM raw_readings WHERE city = :city"
    params = {"city": city}
    if start:
        sql += " AND ts >= :start"
        params["start"] = start
    if end:
        sql += " AND ts <= :end"
        params["end"] = end
    sql += " ORDER BY ts ASC"
    return pd.read_sql(text(sql), engine, params=params)


def get_features(city: str, start=None, end=None) -> pd.DataFrame:
    engine = get_engine()
    sql = "SELECT * FROM features WHERE city = :city"
    params = {"city": city}
    if start:
        sql += " AND ts >= :start"
        params["start"] = start
    if end:
        sql += " AND ts <= :end"
        params["end"] = end
    sql += " ORDER BY ts ASC"
    return pd.read_sql(text(sql), engine, params=params)


def get_latest_features(city: str, n_rows: int = 168) -> pd.DataFrame:
    """Pull the most recent N rows (default: last 7 days hourly) for inference."""
    engine = get_engine()
    sql = """
        SELECT * FROM features
        WHERE city = :city
        ORDER BY ts DESC
        LIMIT :n
    """
    df = pd.read_sql(text(sql), engine, params={"city": city, "n": n_rows})
    return df.sort_values("ts").reset_index(drop=True)


def insert_prediction(row: dict):
    engine = get_engine()
    cols = ", ".join(row.keys())
    placeholders = ", ".join(f":{k}" for k in row.keys())
    sql = f"INSERT INTO predictions ({cols}) VALUES ({placeholders})"
    with engine.begin() as conn:
        conn.execute(text(sql), row)
