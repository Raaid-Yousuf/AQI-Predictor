"""
Training pipeline:
1. Fetch (features, targets) from the feature store
2. Train + evaluate several models per forecast horizon (24h/48h/72h):
   - Ridge Regression (statistical baseline)
   - Random Forest (tree-based)
   - Small LSTM (deep learning, TensorFlow/Keras)
3. Log every run's params/metrics/artifacts to MLflow
4. Register the best model per horizon in the MLflow Model Registry
   and also drop a plain pickle in models/ for the dashboard to load directly
   (keeps the dashboard decoupled from needing a live MLflow server).
"""
import sys
import os
import pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
# Deliberately NOT importing mlflow.keras: that submodule forces mlflow to import
# Keras/TensorFlow through its own internal path, which is what triggers the
# protobuf version collision between mlflow-skinny and TensorFlow. We log Keras
# models manually instead (see train_horizon) — this avoids the conflict
# entirely, on Colab, Kaggle, or anywhere else, regardless of package versions.
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CITY_NAME, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
from db.db_utils import get_features
from training_pipeline.evaluate import compute_metrics

BASE_FEATURE_COLS = [
    "hour", "day_of_week", "day_of_month", "month", "is_weekend",
    "pm25", "pm10", "o3", "no2", "so2", "co",
    "temperature_c", "humidity_pct", "wind_speed_ms", "pressure_hpa",
    "aqi_lag_1h", "aqi_lag_24h", "aqi_rolling_mean_6h", "aqi_change_rate",
]
HORIZONS = {"24h": "target_aqi_24h", "48h": "target_aqi_48h", "72h": "target_aqi_72h"}


def get_feature_cols(horizon_name: str) -> list:
    """Base features + that horizon's future-weather features (the key fix for
    48h/72h — the model needs to know what weather is coming, not just current
    conditions, since weather is the physical driver of AQI change)."""
    future_weather = [
        f"temp_future_{horizon_name}",
        f"humidity_future_{horizon_name}",
        f"wind_speed_future_{horizon_name}",
        f"pressure_future_{horizon_name}",
    ]
    return BASE_FEATURE_COLS + future_weather

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_training_data() -> pd.DataFrame:
    df = get_features(CITY_NAME)
    df["is_weekend"] = df["is_weekend"].astype(int)
    return df


def build_lstm(input_dim: int):
    from tensorflow import keras
    model = keras.Sequential([
        keras.layers.Input(shape=(1, input_dim)),
        keras.layers.LSTM(32, activation="tanh"),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_horizon(df: pd.DataFrame, horizon_name: str, target_col: str):
    feature_cols = get_feature_cols(horizon_name)
    data = df.dropna(subset=feature_cols + [target_col]).copy()
    if len(data) < 50:
        print(f"[{horizon_name}] Not enough data yet ({len(data)} rows) — skipping. "
              f"Run the backfill script with more history.")
        return None

    X = data[feature_cols].values
    y = data[target_col].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}

    with mlflow.start_run(run_name=f"ridge_{horizon_name}"):
        ridge = Ridge(alpha=1.0).fit(X_train_s, y_train)
        preds = ridge.predict(X_test_s)
        metrics = compute_metrics(y_test, preds)
        mlflow.log_params({"model": "ridge", "alpha": 1.0, "horizon": horizon_name})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(ridge, "model")
        results["ridge"] = (ridge, metrics, scaler)
        print(f"[{horizon_name}] Ridge -> {metrics}")

    with mlflow.start_run(run_name=f"random_forest_{horizon_name}"):
        rf = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)  # tree models don't need scaling
        preds = rf.predict(X_test)
        metrics = compute_metrics(y_test, preds)
        mlflow.log_params({"model": "random_forest", "n_estimators": 300, "max_depth": 12, "horizon": horizon_name})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(rf, "model")
        results["random_forest"] = (rf, metrics, None)
        print(f"[{horizon_name}] RandomForest -> {metrics}")

    try:
        with mlflow.start_run(run_name=f"lstm_{horizon_name}"):
            X_train_lstm = X_train_s.reshape((X_train_s.shape[0], 1, X_train_s.shape[1]))
            X_test_lstm = X_test_s.reshape((X_test_s.shape[0], 1, X_test_s.shape[1]))
            lstm = build_lstm(X_train_s.shape[1])
            lstm.fit(X_train_lstm, y_train, epochs=30, batch_size=16, verbose=0)
            preds = lstm.predict(X_test_lstm, verbose=0).flatten()
            metrics = compute_metrics(y_test, preds)
            mlflow.log_params({"model": "lstm", "epochs": 30, "horizon": horizon_name})
            mlflow.log_metrics(metrics)
            # Save manually + log as a plain artifact instead of mlflow.keras.log_model,
            # to avoid the protobuf/TensorFlow import conflict described above.
            tmp_lstm_path = os.path.join(MODELS_DIR, f"_tmp_lstm_{horizon_name}.keras")
            lstm.save(tmp_lstm_path)
            mlflow.log_artifact(tmp_lstm_path, artifact_path="model")
            results["lstm"] = (lstm, metrics, scaler)
            print(f"[{horizon_name}] LSTM -> {metrics}")
    except Exception as e:
        print(f"[{horizon_name}] LSTM training skipped: {e}")

    # Pick best by RMSE
    best_name = min(results, key=lambda k: results[k][1]["rmse"])
    best_model, best_metrics, best_scaler = results[best_name]
    print(f"[{horizon_name}] Best model: {best_name} ({best_metrics})")

    bundle = {
        "model_name": best_name,
        "model": best_model,
        "scaler": best_scaler,
        "feature_cols": feature_cols,
        "metrics": best_metrics,
        "trained_at": datetime.utcnow().isoformat(),
    }
    out_path = os.path.join(MODELS_DIR, f"best_model_{horizon_name}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"[{horizon_name}] Saved best model bundle to {out_path}")

    # Clean up temp LSTM artifact file used only for the mlflow.log_artifact call
    tmp_lstm_path = os.path.join(MODELS_DIR, f"_tmp_lstm_{horizon_name}.keras")
    if os.path.exists(tmp_lstm_path):
        os.remove(tmp_lstm_path)

    return bundle


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    df = load_training_data()
    print(f"Loaded {len(df)} feature rows for {CITY_NAME}.")

    for horizon_name, target_col in HORIZONS.items():
        train_horizon(df, horizon_name, target_col)


if __name__ == "__main__":
    main()