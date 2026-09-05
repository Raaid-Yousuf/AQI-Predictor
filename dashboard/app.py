"""
Pearls AQI Predictor — live dashboard.
Run locally: streamlit run dashboard/app.py
Deploy: push this repo to a Hugging Face Space (Streamlit SDK).
"""
import sys
import os
import pickle

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import shap

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CITY_NAME, HAZARD_AQI_THRESHOLD, FORECAST_DAYS
from db.db_utils import get_latest_features, get_features

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
HORIZONS = {"24h": 1, "48h": 2, "72h": 3}

st.set_page_config(page_title="Pearls AQI Predictor", page_icon="🌍", layout="wide")


@st.cache_resource
def load_model_bundle(horizon_name: str):
    path = os.path.join(MODELS_DIR, f"best_model_{horizon_name}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_with_bundle(bundle, X_row: pd.DataFrame):
    X = X_row[bundle["feature_cols"]].values
    if bundle["scaler"] is not None:
        X = bundle["scaler"].transform(X)
    model = bundle["model"]
    model_name = bundle["model_name"]
    if "lstm" in model_name:
        X = X.reshape((X.shape[0], 1, X.shape[1]))
        pred = model.predict(X, verbose=0).flatten()[0]
    else:
        pred = model.predict(X)[0]

    if bundle.get("predict_delta"):
        anchor_value = X_row[bundle["anchor_col"]].values[0]
        pred = pred + anchor_value

    return float(pred)


def aqi_category(aqi: float):
    if aqi <= 50:
        return "Good", "#00e400"
    elif aqi <= 100:
        return "Moderate", "#ffff00"
    elif aqi <= 150:
        return "Unhealthy (Sensitive)", "#ff7e00"
    elif aqi <= 200:
        return "Unhealthy", "#ff0000"
    elif aqi <= 300:
        return "Very Unhealthy", "#8f3f97"
    else:
        return "Hazardous", "#7e0023"


def main():
    st.title(f"🌍 Pearls AQI Predictor — {CITY_NAME}")
    st.caption("3-day AQI forecast · dual-source data (Open-Meteo + AQICN) · explainable predictions")

    latest_features = get_latest_features(CITY_NAME, n_rows=1)
    if latest_features.empty:
        st.warning("No feature data yet. Run the backfill and feature pipeline scripts first.")
        return

    latest_row = latest_features.tail(1)
    current_aqi = latest_row.iloc[0].get("target_aqi") or latest_row.iloc[0].get("aqi_lag_1h")

    col1, col2, col3, col4 = st.columns(4)
    if current_aqi is not None:
        cat, color = aqi_category(current_aqi)
        col1.metric("Current AQI", f"{current_aqi:.0f}", cat)
    col2.metric("PM2.5", f"{latest_row.iloc[0].get('pm25', float('nan')):.1f} µg/m³")
    col3.metric("Temperature", f"{latest_row.iloc[0].get('temperature_c', float('nan')):.1f} °C")
    col4.metric("Humidity", f"{latest_row.iloc[0].get('humidity_pct', float('nan')):.0f} %")

    if current_aqi is not None and current_aqi >= HAZARD_AQI_THRESHOLD:
        st.error(f"⚠️ Hazardous AQI level detected ({current_aqi:.0f}). Limit outdoor exposure.")

    st.subheader(f"Next {FORECAST_DAYS}-Day Forecast")

    forecast_rows = []
    for horizon_name, day_offset in HORIZONS.items():
        bundle = load_model_bundle(horizon_name)
        if bundle is None:
            st.info(f"No trained model found for {horizon_name} yet — run training_pipeline/train.py")
            continue
        pred = predict_with_bundle(bundle, latest_row)
        forecast_rows.append({
            "horizon": f"Day +{day_offset}",
            "predicted_aqi": pred,
            "model": bundle["model_name"],
            "experimental": bundle.get("experimental", False),
        })

    if forecast_rows:
        fdf = pd.DataFrame(forecast_rows)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=fdf["horizon"], y=fdf["predicted_aqi"],
            marker_color=[aqi_category(v)[1] for v in fdf["predicted_aqi"]],
            text=[f"{v:.0f}" for v in fdf["predicted_aqi"]],
            textposition="outside",
        ))
        fig.update_layout(yaxis_title="Predicted US AQI", showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(fdf, use_container_width=True, hide_index=True)

        if fdf["experimental"].any():
            weak_horizons = ", ".join(fdf.loc[fdf["experimental"], "horizon"])
            st.caption(f"⚠️ {weak_horizons} forecast(s) have low model confidence (R² below 0.15 in "
                       f"backtesting) — treat as a rough directional estimate, not a precise prediction.")

    st.subheader("Recent AQI Trend")
    hist = get_features(CITY_NAME).tail(168)  # last 7 days hourly
    if not hist.empty:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=hist["ts"], y=hist["target_aqi"], mode="lines", name="AQI"))
        fig2.update_layout(yaxis_title="US AQI", height=300)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Why this forecast? (SHAP feature importance)")
    horizon_choice = st.selectbox("Explain horizon:", list(HORIZONS.keys()))
    bundle = load_model_bundle(horizon_choice)
    if bundle and ("random_forest" in bundle["model_name"] or "ridge" in bundle["model_name"]):
        try:
            background = get_features(CITY_NAME).dropna(subset=bundle["feature_cols"]).tail(200)
            X_bg = background[bundle["feature_cols"]]
            if bundle["scaler"] is not None:
                X_bg_arr = bundle["scaler"].transform(X_bg.values)
            else:
                X_bg_arr = X_bg.values

            explainer = shap.Explainer(bundle["model"], X_bg_arr)
            X_current = latest_row[bundle["feature_cols"]].values
            if bundle["scaler"] is not None:
                X_current = bundle["scaler"].transform(X_current)
            shap_values = explainer(X_current)

            shap_df = pd.DataFrame({
                "feature": bundle["feature_cols"],
                "impact": shap_values.values[0],
            }).sort_values("impact", key=abs, ascending=False).head(10)

            fig3 = go.Figure(go.Bar(
                x=shap_df["impact"], y=shap_df["feature"], orientation="h",
                marker_color=["#ff4d4d" if v > 0 else "#4d94ff" for v in shap_df["impact"]],
            ))
            fig3.update_layout(xaxis_title="Impact on predicted AQI", height=400)
            st.plotly_chart(fig3, use_container_width=True)
            st.caption("Red = pushes AQI higher, Blue = pushes AQI lower")
        except Exception as e:
            st.info(f"SHAP explanation unavailable: {e}")
    else:
        st.info("SHAP explanation is available for Ridge and Random Forest models (not LSTM in this MVP).")


if __name__ == "__main__":
    main()