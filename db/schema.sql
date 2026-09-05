-- Pearls AQI Predictor — feature store schema (Postgres / Supabase)

CREATE TABLE IF NOT EXISTS raw_readings (
    id              BIGSERIAL PRIMARY KEY,
    city            TEXT NOT NULL,
    ts              TIMESTAMPTZ NOT NULL,       -- timestamp of the reading
    source          TEXT NOT NULL,              -- 'open_meteo' | 'aqicn'
    pm25            DOUBLE PRECISION,
    pm10            DOUBLE PRECISION,
    o3              DOUBLE PRECISION,
    no2             DOUBLE PRECISION,
    so2             DOUBLE PRECISION,
    co              DOUBLE PRECISION,
    us_aqi          DOUBLE PRECISION,           -- reported AQI (target source)
    temperature_c   DOUBLE PRECISION,
    humidity_pct    DOUBLE PRECISION,
    wind_speed_ms   DOUBLE PRECISION,
    wind_direction  DOUBLE PRECISION,
    pressure_hpa    DOUBLE PRECISION,
    fetched_at      TIMESTAMPTZ DEFAULT now(),
    UNIQUE (city, ts, source)
);

CREATE TABLE IF NOT EXISTS features (
    id                  BIGSERIAL PRIMARY KEY,
    city                TEXT NOT NULL,
    ts                  TIMESTAMPTZ NOT NULL,
    hour                INT,
    day_of_week         INT,
    day_of_month        INT,
    month               INT,
    is_weekend          BOOLEAN,
    pm25                DOUBLE PRECISION,
    pm10                DOUBLE PRECISION,
    o3                  DOUBLE PRECISION,
    no2                 DOUBLE PRECISION,
    so2                 DOUBLE PRECISION,
    co                  DOUBLE PRECISION,
    temperature_c       DOUBLE PRECISION,
    humidity_pct        DOUBLE PRECISION,
    wind_speed_ms       DOUBLE PRECISION,
    pressure_hpa        DOUBLE PRECISION,
    aqi_lag_1h          DOUBLE PRECISION,
    aqi_lag_24h         DOUBLE PRECISION,
    aqi_rolling_mean_6h DOUBLE PRECISION,
    aqi_change_rate     DOUBLE PRECISION,       -- derived: (aqi_t - aqi_t-1) / aqi_t-1
    target_aqi          DOUBLE PRECISION,       -- current AQI (label for "now")
    target_aqi_24h      DOUBLE PRECISION,       -- AQI 24h ahead (for day+1 model)
    target_aqi_48h      DOUBLE PRECISION,       -- AQI 48h ahead (for day+2 model)
    target_aqi_72h      DOUBLE PRECISION,       -- AQI 72h ahead (for day+3 model)
    created_at          TIMESTAMPTZ DEFAULT now(),
    UNIQUE (city, ts)
);

CREATE TABLE IF NOT EXISTS predictions (
    id              BIGSERIAL PRIMARY KEY,
    city            TEXT NOT NULL,
    predicted_for   TIMESTAMPTZ NOT NULL,   -- date/time being forecast
    horizon_days    INT NOT NULL,           -- 1, 2, or 3
    predicted_aqi   DOUBLE PRECISION NOT NULL,
    model_name      TEXT NOT NULL,
    model_version   TEXT,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_features_city_ts ON features (city, ts DESC);
CREATE INDEX IF NOT EXISTS idx_raw_city_ts ON raw_readings (city, ts DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_city_created ON predictions (city, created_at DESC);
