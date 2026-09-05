"""
Central configuration for Pearls AQI Predictor.
Loads from environment variables (populated via .env locally, or via
GitHub Actions Secrets in CI).
"""
import os
from dotenv import load_dotenv

load_dotenv(override=True)  # .env always wins over stray shell/session env vars

# --- City ---
CITY_NAME = os.getenv("CITY_NAME", "Lahore")
CITY_LAT = float(os.getenv("CITY_LAT", "31.5497"))
CITY_LON = float(os.getenv("CITY_LON", "74.3436"))

# --- Database ---
DATABASE_URL = os.getenv("DATABASE_URL")

# --- APIs ---
AQICN_TOKEN = os.getenv("AQICN_TOKEN")
OPEN_METEO_AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
OPEN_METEO_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
AQICN_URL_TEMPLATE = "https://api.waqi.info/feed/geo:{lat};{lon}/?token={token}"

# --- MLflow ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT_NAME = "pearls-aqi-forecaster"

# --- Alerts ---
HAZARD_AQI_THRESHOLD = float(os.getenv("HAZARD_AQI_THRESHOLD", "150"))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
ALERT_EMAIL_FROM = os.getenv("ALERT_EMAIL_FROM")
ALERT_EMAIL_TO = os.getenv("ALERT_EMAIL_TO")
ALERT_EMAIL_APP_PASSWORD = os.getenv("ALERT_EMAIL_APP_PASSWORD")

# --- Forecast horizon ---
FORECAST_DAYS = 3
