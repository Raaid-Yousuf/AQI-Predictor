"""
Pearls AQI Predictor — live dashboard.
Run locally: streamlit run dashboard/app.py
Deploy: push this repo to Streamlit Community Cloud (share.streamlit.io).

Visual design: an environmental-monitor aesthetic (dark, instrument-panel feel)
rather than a generic SaaS card dashboard. One hero moment (the AQI ring), a
consistent severity color system borrowed from the standard AQI scale (so it
reads as trustworthy/familiar), and quiet, structured supporting panels.
"""
import sys
import os
import pickle
from datetime import datetime, timezone, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import shap

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CITY_NAME, HAZARD_AQI_THRESHOLD, FORECAST_DAYS
from db.db_utils import get_latest_features, get_features
from feature_pipeline.fetch_data import fetch_open_meteo_weather_forecast

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
HORIZONS = {"24h": 1, "48h": 2, "72h": 3}


def html(markup: str):
    """Render raw HTML safely. Strips leading whitespace from every line first —
    Markdown treats 4+ leading spaces as a preformatted code block, which is what
    was breaking the forecast cards (each concatenated chunk kept its Python
    source indentation, and Streamlit's markdown parser rendered it as literal text)."""
    cleaned = "\n".join(line.strip() for line in markup.strip().splitlines())
    st.markdown(cleaned, unsafe_allow_html=True)

# --- Design tokens -----------------------------------------------------------
INK = "#0F1115"
PANEL = "#171A21"
PANEL_ALT = "#1E222B"
HAIRLINE = "#262B36"
TEXT = "#ECE9E2"
TEXT_MUTED = "#8890A0"
ACCENT = "#4CC9C0"

AQI_SCALE = [
    (50, "Good", "#5FD98A", "Air quality is satisfactory."),
    (100, "Moderate", "#F5CB4E", "Acceptable, but sensitive groups may notice minor effects."),
    (150, "Unhealthy for Sensitive Groups", "#F5934E", "Sensitive groups should reduce prolonged outdoor exertion."),
    (200, "Unhealthy", "#F0615F", "Everyone may begin to experience health effects."),
    (300, "Very Unhealthy", "#B57EF0", "Health alert — everyone may experience more serious effects."),
    (10_000, "Hazardous", "#8A2E3B", "Health emergency — avoid all outdoor exertion."),
]

st.set_page_config(page_title="Pearls AQI", page_icon="🜁", layout="wide")


def inject_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {{
        font-family: 'IBM Plex Sans', sans-serif;
    }}
    .stApp {{
        background-color: {INK};
        color: {TEXT};
    }}
    #MainMenu, footer, header {{ visibility: hidden; }}
    .block-container {{ padding-top: 2rem; max-width: 1100px; }}

    h1, h2, h3, .pearls-display {{
        font-family: 'Space Grotesk', sans-serif;
        color: {TEXT};
        font-weight: 600;
    }}

    /* --- Header --- */
    .pearls-header {{
        display: flex; justify-content: space-between; align-items: baseline;
        border-bottom: 1px solid {HAIRLINE}; padding-bottom: 1rem; margin-bottom: 2rem;
    }}
    .pearls-wordmark {{ font-family: 'Space Grotesk', sans-serif; font-size: 1.4rem; font-weight: 700; color: {TEXT}; }}
    .pearls-wordmark span {{ color: {ACCENT}; }}
    .pearls-meta {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; color: {TEXT_MUTED}; text-align: right; }}

    /* --- Hero --- */
    .hero-wrap {{ display: flex; gap: 3rem; align-items: center; margin-bottom: 2.5rem; flex-wrap: wrap; }}
    .gauge {{
        width: 200px; height: 200px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        position: relative; flex-shrink: 0;
    }}
    .gauge::after {{
        content: ""; position: absolute; inset: 16px; background: {PANEL}; border-radius: 50%;
    }}
    .gauge-inner {{ position: relative; z-index: 1; text-align: center; }}
    .gauge-number {{ font-family: 'Space Grotesk', sans-serif; font-size: 3rem; font-weight: 700; line-height: 1; color: {TEXT}; }}
    .gauge-label {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: {TEXT_MUTED}; text-transform: none; margin-top: 4px; }}

    .stat-rail {{ display: flex; gap: 2.2rem; flex-wrap: wrap; }}
    .stat-item {{ padding-right: 2.2rem; border-right: 1px solid {HAIRLINE}; }}
    .stat-item:last-child {{ border-right: none; }}
    .stat-value {{ font-family: 'Space Grotesk', sans-serif; font-size: 1.6rem; font-weight: 600; color: {TEXT}; }}
    .stat-label {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: {TEXT_MUTED}; margin-top: 2px; }}
    .category-pill {{
        display: inline-block; padding: 3px 12px; border-radius: 20px;
        font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; margin-top: 10px;
    }}

    .alert-banner {{
        border-left: 3px solid #8A2E3B; background: rgba(138,46,59,0.12);
        padding: 0.9rem 1.2rem; border-radius: 4px; margin-bottom: 2rem;
        font-size: 0.92rem; color: {TEXT};
    }}

    /* --- Section headers --- */
    .section-head {{ font-family: 'Space Grotesk', sans-serif; font-size: 1.05rem; font-weight: 600;
        color: {TEXT}; margin: 2.2rem 0 1rem 0; }}
    .section-sub {{ font-family: 'IBM Plex Sans', sans-serif; font-size: 0.85rem; color: {TEXT_MUTED}; margin-top: -0.7rem; margin-bottom: 1rem; }}

    /* --- Forecast strip --- */
    .forecast-strip {{ display: flex; gap: 1rem; flex-wrap: wrap; }}
    .day-card {{
        flex: 1; min-width: 200px; background: {PANEL}; border: 1px solid {HAIRLINE};
        border-left-width: 4px; border-radius: 6px; padding: 1.1rem 1.3rem;
    }}
    .day-card-head {{ display: flex; justify-content: space-between; align-items: center; }}
    .day-label {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; color: {TEXT_MUTED}; }}
    .day-number {{ font-family: 'Space Grotesk', sans-serif; font-size: 2.1rem; font-weight: 700; color: {TEXT}; margin-top: 6px; }}
    .day-category {{ font-size: 0.82rem; margin-top: 2px; }}
    .day-trend {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; margin-top: 8px; }}
    .experimental-tag {{
        font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; color: {TEXT_MUTED};
        border-top: 1px dashed {HAIRLINE}; margin-top: 10px; padding-top: 8px;
    }}

    .footer-note {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: {TEXT_MUTED};
        border-top: 1px solid {HAIRLINE}; padding-top: 1rem; margin-top: 3rem; }}
    </style>
    """, unsafe_allow_html=True)


def get_aqi_info(aqi: float):
    """Returns (category, color, guidance) for a given AQI value."""
    for threshold, category, color, guidance in AQI_SCALE:
        if aqi <= threshold:
            return category, color, guidance
    return AQI_SCALE[-1][1], AQI_SCALE[-1][2], AQI_SCALE[-1][3]


@st.cache_resource
def load_model_bundle(horizon_name: str):
    path = os.path.join(MODELS_DIR, f"best_model_{horizon_name}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    if bundle.get("lstm_model_path"):
        from tensorflow import keras
        lstm_path = os.path.join(MODELS_DIR, bundle["lstm_model_path"])
        bundle["model"] = keras.models.load_model(lstm_path)
    return bundle


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


@st.cache_data(ttl=1800)  # forecast weather doesn't change minute to minute — cache 30 min
def get_weather_forecast():
    return fetch_open_meteo_weather_forecast(forecast_days=4)


def enrich_with_forecast_weather(latest_row: pd.DataFrame) -> pd.DataFrame:
    """
    Fills in the 'future weather' feature columns (temp_future_24h, etc.) using a
    REAL weather forecast — these are NaN in the stored feature row for the most
    recent timestamp, since actual future weather hasn't happened yet. At training
    time we approximate them with historical actuals; at live inference this is
    the only correct source: a real forecast for the hours ahead.
    """
    latest_row = latest_row.copy()
    forecast = get_weather_forecast()
    if not forecast:
        return latest_row  # fall back to whatever is already there (likely NaN)

    current_ts = pd.Timestamp(latest_row.iloc[0]["ts"])
    forecast_index = {pd.Timestamp(k): v for k, v in forecast.items()}

    field_map = {
        "temperature_c": "temp_future",
        "humidity_pct": "humidity_future",
        "wind_speed_ms": "wind_speed_future",
        "pressure_hpa": "pressure_future",
    }

    for horizon_hours, suffix in [(24, "24h"), (48, "48h"), (72, "72h")]:
        target_ts = current_ts + timedelta(hours=horizon_hours)
        # snap to the nearest available forecast hour
        nearest_ts = min(forecast_index.keys(), key=lambda t: abs((t - target_ts).total_seconds()),
                          default=None)
        if nearest_ts is None:
            continue
        values = forecast_index[nearest_ts]
        for source_field, prefix in field_map.items():
            col = f"{prefix}_{suffix}"
            if col in latest_row.columns and values.get(source_field) is not None:
                latest_row.at[latest_row.index[0], col] = values[source_field]

    return latest_row


def render_header(last_ts):
    if last_ts is not None:
        ts = pd.Timestamp(last_ts)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        age_min = int((pd.Timestamp.now(tz="UTC") - ts).total_seconds() / 60)
        updated = f"UPDATED {age_min}M AGO" if age_min < 60 else f"UPDATED {age_min // 60}H AGO"
    else:
        updated = "NO DATA"
    html(f"""
    <div class="pearls-header">
        <div class="pearls-wordmark">Pearl<span>s</span> AQI</div>
        <div class="pearls-meta">{CITY_NAME.upper()} · {updated}</div>
    </div>
    """)


def render_hero(current_aqi, pm25, temp, humidity, wind):
    category, color, guidance = get_aqi_info(current_aqi)
    pct = min(current_aqi / 400, 1.0) * 360

    html(f"""
    <div class="hero-wrap">
        <div class="gauge" style="background: conic-gradient({color} {pct}deg, {PANEL_ALT} 0deg);
                                   box-shadow: 0 0 40px -8px {color}66;">
            <div class="gauge-inner">
                <div class="gauge-number">{current_aqi:.0f}</div>
                <div class="gauge-label">US AQI NOW</div>
            </div>
        </div>
        <div>
            <span class="category-pill" style="background:{color}26; color:{color};">{category}</span>
            <div class="section-sub" style="margin-top:8px; max-width:280px;">{guidance}</div>
            <div class="stat-rail" style="margin-top:1rem;">
                <div class="stat-item"><div class="stat-value">{pm25:.0f}</div><div class="stat-label">PM2.5 µG/M³</div></div>
                <div class="stat-item"><div class="stat-value">{temp:.0f}°</div><div class="stat-label">TEMP C</div></div>
                <div class="stat-item"><div class="stat-value">{humidity:.0f}%</div><div class="stat-label">HUMIDITY</div></div>
                <div class="stat-item"><div class="stat-value">{wind:.0f}</div><div class="stat-label">WIND M/S</div></div>
            </div>
        </div>
    </div>
    """)

    if current_aqi >= HAZARD_AQI_THRESHOLD:
        html(f"""
        <div class="alert-banner">⚠ Hazardous AQI detected ({current_aqi:.0f}). Limit outdoor exposure.</div>
        """)


def render_forecast_strip(forecast_rows, current_aqi):
    cards_html = '<div class="forecast-strip">'
    for row in forecast_rows:
        category, color, _ = get_aqi_info(row["predicted_aqi"])
        delta = row["predicted_aqi"] - current_aqi
        arrow = "▲" if delta > 3 else ("▼" if delta < -3 else "→")
        trend_color = "#F0615F" if delta > 3 else ("#5FD98A" if delta < -3 else TEXT_MUTED)
        exp_tag = f'<div class="experimental-tag">low confidence · directional estimate only</div>' if row["experimental"] else ""
        cards_html += f"""
        <div class="day-card" style="border-left-color:{color};">
            <div class="day-card-head">
                <span class="day-label">{row['horizon'].upper()}</span>
            </div>
            <div class="day-number">{row['predicted_aqi']:.0f}</div>
            <div class="day-category" style="color:{color};">{category}</div>
            <div class="day-trend" style="color:{trend_color};">{arrow} {abs(delta):.0f} vs now</div>
            {exp_tag}
        </div>
        """
    cards_html += "</div>"
    html(cards_html)


def styled_layout(fig, height=320):
    fig.update_layout(
        height=height,
        paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        font=dict(family="IBM Plex Sans", color=TEXT_MUTED, size=12),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(gridcolor=HAIRLINE, zerolinecolor=HAIRLINE),
        yaxis=dict(gridcolor=HAIRLINE, zerolinecolor=HAIRLINE),
    )
    return fig


def main():
    inject_css()

    latest_features = get_latest_features(CITY_NAME, n_rows=1)
    if latest_features.empty:
        render_header(None)
        st.warning("No feature data yet. Run the backfill and feature pipeline scripts first.")
        return
    latest_features["is_weekend"] = latest_features["is_weekend"].astype(int)

    latest_row = latest_features.tail(1)
    latest_row = enrich_with_forecast_weather(latest_row)  # fill NaN future-weather with a real forecast
    row0 = latest_row.iloc[0]
    current_aqi = row0.get("target_aqi") or row0.get("aqi_lag_1h")

    render_header(row0.get("ts"))
    render_hero(
        current_aqi,
        row0.get("pm25", 0) or 0,
        row0.get("temperature_c", 0) or 0,
        row0.get("humidity_pct", 0) or 0,
        row0.get("wind_speed_ms", 0) or 0,
    )

    st.markdown(f'<div class="section-head">Next {FORECAST_DAYS}-Day Forecast</div>', unsafe_allow_html=True)

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
        render_forecast_strip(forecast_rows, current_aqi)

    st.markdown('<div class="section-head">7-Day Trend</div>', unsafe_allow_html=True)
    hist = get_features(CITY_NAME).tail(168)
    if not hist.empty:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=hist["ts"], y=hist["target_aqi"], mode="lines",
            line=dict(color=ACCENT, width=2),
            fill="tozeroy", fillcolor="rgba(76,201,192,0.08)", name="AQI",
        ))
        fig2.add_hline(y=HAZARD_AQI_THRESHOLD, line_dash="dot", line_color="#8A2E3B",
                        annotation_text="hazard threshold", annotation_font_color=TEXT_MUTED)
        st.plotly_chart(styled_layout(fig2), use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="section-head">Why this forecast?</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">SHAP feature attribution — what\'s driving the selected forecast</div>', unsafe_allow_html=True)
    horizon_choice = st.selectbox("Horizon", list(HORIZONS.keys()), label_visibility="collapsed")
    bundle = load_model_bundle(horizon_choice)
    if bundle and ("random_forest" in bundle["model_name"] or "ridge" in bundle["model_name"]):
        try:
            background = get_features(CITY_NAME).dropna(subset=bundle["feature_cols"]).tail(200).copy()
            background["is_weekend"] = background["is_weekend"].astype(int)
            X_bg = background[bundle["feature_cols"]]
            X_bg_arr = bundle["scaler"].transform(X_bg.values) if bundle["scaler"] is not None else X_bg.values

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
                marker_color=["#F0615F" if v > 0 else ACCENT for v in shap_df["impact"]],
            ))
            fig3.update_layout(xaxis_title="Impact on predicted AQI", yaxis=dict(autorange="reversed"))
            st.plotly_chart(styled_layout(fig3, height=380), use_container_width=True, config={"displayModeBar": False})
            st.markdown(f'<span style="color:#F0615F;">■</span> pushes AQI higher &nbsp;&nbsp; '
                        f'<span style="color:{ACCENT};">■</span> pushes AQI lower',
                        unsafe_allow_html=True)
        except Exception as e:
            st.info(f"SHAP explanation unavailable: {e}")
    else:
        st.info("SHAP explanation is available for Ridge and Random Forest models (not LSTM).")

    st.markdown(
        '<div class="footer-note">DATA: OPEN-METEO + AQICN &nbsp;·&nbsp; MODELS RETRAINED DAILY &nbsp;·&nbsp; '
        'FORECASTS BEYOND 24H ARE DIRECTIONAL ESTIMATES</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()