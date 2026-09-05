# Pearls AQI Predictor 🌍💨

An end-to-end, **serverless** ML system that forecasts Air Quality Index (AQI)
for the next 3 days for a given city — built as a real product, not a class demo.

## Architecture

```
Open-Meteo API  ─┐
                  ├─► feature_pipeline/ ─► Supabase Postgres (feature store)
AQICN API       ─┘                              │
                                                 ▼
                              training_pipeline/ (RF, Ridge, LSTM)
                                                 │
                                        MLflow (model registry)
                                                 │
                                                 ▼
                                   dashboard/ (Streamlit on HF Spaces)
                                                 │
                                     SHAP explainability + alerts
```

Automation: GitHub Actions runs the feature pipeline **hourly** and the
training pipeline **daily** (see `.github/workflows/`).

## Why this stack (not the "default" course stack)

| Layer | Chosen | Instead of | Why |
|---|---|---|---|
| Data | Open-Meteo (primary, free, no key) + AQICN (secondary, ground truth) | single-source | redundancy, richer features |
| Feature store | Supabase Postgres | Hopsworks/Vertex | zero lock-in, real SQL, doubles as app DB |
| Orchestration | GitHub Actions | Airflow | free, no infra to run, enough for hourly/daily cadence |
| Model registry | MLflow (self-hosted on Supabase) | managed platforms | free, portable, no vendor lock-in |
| Dashboard | Streamlit on Hugging Face Spaces | Streamlit Community Cloud | more RAM/CPU on free tier |

## Repo layout

```
pearls-aqi-predictor/
├── config.py                      # central config: city, API keys, DB URL
├── db/
│   ├── schema.sql                 # Postgres schema (feature store)
│   └── db_utils.py                # connection + insert/query helpers
├── feature_pipeline/
│   ├── fetch_data.py              # Open-Meteo + AQICN fetchers
│   ├── feature_engineering.py     # time features + derived features
│   └── run_feature_pipeline.py    # orchestrates fetch -> engineer -> store
├── backfill/
│   └── backfill_historical.py     # loops fetch over a historical date range
├── training_pipeline/
│   ├── evaluate.py                # RMSE / MAE / R2
│   └── train.py                   # trains RF, Ridge, LSTM; logs to MLflow
├── alerts/
│   └── notifier.py                # Telegram / email alert on hazardous AQI
├── dashboard/
│   └── app.py                     # Streamlit forecast dashboard + SHAP
├── .github/workflows/
│   ├── feature_pipeline.yml       # cron: every hour
│   └── training_pipeline.yml      # cron: daily
├── requirements.txt
└── .env.example
```

## Setup

1. **Create a free Supabase project** → grab the Postgres connection string.
2. **Get an AQICN token** (free): https://aqicn.org/data-platform/token/
3. Copy `.env.example` → `.env` and fill in values.
4. `pip install -r requirements.txt`
5. Run the schema once: `psql "$DATABASE_URL" -f db/schema.sql`
6. Backfill history: `python backfill/backfill_historical.py --days 90`
7. Train: `python training_pipeline/train.py`
8. Run dashboard locally: `streamlit run dashboard/app.py`
9. Push to GitHub, add the same secrets in repo Settings → Secrets → Actions,
   and the two workflows will start running automatically.

## Notes on scaling later

- Swap `db/db_utils.py`'s connection string for any Postgres (RDS, Neon, self-hosted) — nothing else changes.
- MLflow tracking URI is just a Postgres table + a local `mlruns/` artifact
  folder — point it at S3/GCS later for durability without touching training code.
- If you outgrow GitHub Actions' free minutes, swap the trigger for Prefect/Dagster
  without changing the pipeline scripts themselves — they're plain Python functions.
