"""
Dual-source data fetchers:
- Open-Meteo: free, no API key, has both current + historical air quality & weather
- AQICN (WAQI): free with token, ground-truth station readings (current only on free tier)

Both fetchers return a plain dict keyed the same way, ready for feature_engineering.py
"""
import requests
import sys
import os
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    CITY_LAT, CITY_LON, AQICN_TOKEN,
    OPEN_METEO_AIR_QUALITY_URL, OPEN_METEO_WEATHER_URL,
    OPEN_METEO_ARCHIVE_URL, AQICN_URL_TEMPLATE,
)


def fetch_open_meteo_current():
    """Current air quality + weather from Open-Meteo. No API key needed."""
    aq_params = {
        "latitude": CITY_LAT,
        "longitude": CITY_LON,
        "current": "pm10,pm2_5,ozone,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,us_aqi",
        "timezone": "auto",
    }
    weather_params = {
        "latitude": CITY_LAT,
        "longitude": CITY_LON,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,pressure_msl",
        "timezone": "auto",
    }
    aq = requests.get(OPEN_METEO_AIR_QUALITY_URL, params=aq_params, timeout=20).json()
    wx = requests.get(OPEN_METEO_WEATHER_URL, params=weather_params, timeout=20).json()

    aq_cur = aq.get("current", {})
    wx_cur = wx.get("current", {})

    return {
        "ts": aq_cur.get("time"),
        "source": "open_meteo",
        "pm25": aq_cur.get("pm2_5"),
        "pm10": aq_cur.get("pm10"),
        "o3": aq_cur.get("ozone"),
        "no2": aq_cur.get("nitrogen_dioxide"),
        "so2": aq_cur.get("sulphur_dioxide"),
        "co": aq_cur.get("carbon_monoxide"),
        "us_aqi": aq_cur.get("us_aqi"),
        "temperature_c": wx_cur.get("temperature_2m"),
        "humidity_pct": wx_cur.get("relative_humidity_2m"),
        "wind_speed_ms": wx_cur.get("wind_speed_10m"),
        "wind_direction": wx_cur.get("wind_direction_10m"),
        "pressure_hpa": wx_cur.get("pressure_msl"),
    }


def fetch_open_meteo_historical(start_date: str, end_date: str):
    """
    Historical air quality + weather from Open-Meteo archive, hourly resolution.
    start_date/end_date format: 'YYYY-MM-DD'. Used by the backfill script.
    Returns a list of dicts, one per hour.
    """
    aq_params = {
        "latitude": CITY_LAT,
        "longitude": CITY_LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "pm10,pm2_5,ozone,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,us_aqi",
        "timezone": "auto",
    }
    weather_params = {
        "latitude": CITY_LAT,
        "longitude": CITY_LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,pressure_msl",
        "timezone": "auto",
    }
    aq = requests.get(OPEN_METEO_AIR_QUALITY_URL, params=aq_params, timeout=30).json()
    wx = requests.get(OPEN_METEO_ARCHIVE_URL, params=weather_params, timeout=30).json()

    aq_hourly = aq.get("hourly", {})
    wx_hourly = wx.get("hourly", {})
    times = aq_hourly.get("time", [])

    rows = []
    for i, t in enumerate(times):
        rows.append({
            "ts": t,
            "source": "open_meteo",
            "pm25": _safe_get(aq_hourly, "pm2_5", i),
            "pm10": _safe_get(aq_hourly, "pm10", i),
            "o3": _safe_get(aq_hourly, "ozone", i),
            "no2": _safe_get(aq_hourly, "nitrogen_dioxide", i),
            "so2": _safe_get(aq_hourly, "sulphur_dioxide", i),
            "co": _safe_get(aq_hourly, "carbon_monoxide", i),
            "us_aqi": _safe_get(aq_hourly, "us_aqi", i),
            "temperature_c": _safe_get(wx_hourly, "temperature_2m", i),
            "humidity_pct": _safe_get(wx_hourly, "relative_humidity_2m", i),
            "wind_speed_ms": _safe_get(wx_hourly, "wind_speed_10m", i),
            "wind_direction": _safe_get(wx_hourly, "wind_direction_10m", i),
            "pressure_hpa": _safe_get(wx_hourly, "pressure_msl", i),
        })
    return rows


def fetch_aqicn_current():
    """Current ground-truth station reading from AQICN/WAQI. Needs a free token."""
    if not AQICN_TOKEN:
        return None
    url = AQICN_URL_TEMPLATE.format(lat=CITY_LAT, lon=CITY_LON, token=AQICN_TOKEN)
    resp = requests.get(url, timeout=20).json()
    if resp.get("status") != "ok":
        return None
    data = resp["data"]
    iaqi = data.get("iaqi", {})
    return {
        "ts": data.get("time", {}).get("iso", datetime.now(timezone.utc).isoformat()),
        "source": "aqicn",
        "pm25": iaqi.get("pm25", {}).get("v"),
        "pm10": iaqi.get("pm10", {}).get("v"),
        "o3": iaqi.get("o3", {}).get("v"),
        "no2": iaqi.get("no2", {}).get("v"),
        "so2": iaqi.get("so2", {}).get("v"),
        "co": iaqi.get("co", {}).get("v"),
        "us_aqi": data.get("aqi"),
        "temperature_c": iaqi.get("t", {}).get("v"),
        "humidity_pct": iaqi.get("h", {}).get("v"),
        "wind_speed_ms": iaqi.get("w", {}).get("v"),
        "wind_direction": None,
        "pressure_hpa": iaqi.get("p", {}).get("v"),
    }


def _safe_get(hourly_dict, key, i):
    values = hourly_dict.get(key)
    if values is None or i >= len(values):
        return None
    return values[i]


if __name__ == "__main__":
    print("Open-Meteo current:", fetch_open_meteo_current())
    print("AQICN current:", fetch_aqicn_current())
