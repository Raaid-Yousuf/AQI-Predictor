-- Migration: add per-horizon future-weather columns to the existing features table.
-- Safe to run even though the table already has data — existing rows just get NULL
-- in these new columns until you re-run the backfill/feature pipeline.

ALTER TABLE features
    ADD COLUMN IF NOT EXISTS temp_future_24h        DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS temp_future_48h        DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS temp_future_72h        DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS humidity_future_24h    DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS humidity_future_48h    DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS humidity_future_72h    DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS wind_speed_future_24h  DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS wind_speed_future_48h  DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS wind_speed_future_72h  DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS pressure_future_24h    DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS pressure_future_48h    DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS pressure_future_72h    DOUBLE PRECISION;