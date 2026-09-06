# 🌍 Pearls AQI Predictor

> **An end-to-end, production-oriented machine learning system for forecasting Air Quality Index (AQI) up to 72 hours ahead.**

Pearls AQI Predictor is a complete **serverless-style ML forecasting pipeline** designed to collect air-quality and weather data, engineer predictive features, train multiple machine-learning and deep-learning models, track experiments with MLflow, store features in PostgreSQL/Supabase, generate future AQI predictions, provide explainability through SHAP, and display the results through an interactive Streamlit dashboard.

Unlike a simple machine-learning notebook, this project is structured as a complete ML system with **automated data ingestion, feature engineering, model training, model persistence, monitoring-oriented alerts, and an interactive prediction interface**.

---

## ✨ Key Features

* 🌫️ **AQI forecasting up to 72 hours ahead**
* 🌦️ Integration of air-quality and weather data
* 🔄 Automated hourly feature/data pipeline
* 🤖 Multiple forecasting models:

  * Ridge Regression
  * Random Forest Regressor
  * LSTM Neural Network
* 🎯 Separate forecasting horizons:

  * 24 hours
  * 48 hours
  * 72 hours
* 📊 Multiple evaluation metrics:

  * RMSE
  * MAE
  * R²
* 🧪 Experiment tracking using **MLflow**
* 🗃️ PostgreSQL/Supabase feature store
* 🔮 Real weather forecasts used during live inference
* 🧠 SHAP-based model explainability

  
* 📈 Interactive Streamlit dashboard
* ⚙️ Automated GitHub Actions pipelines
* 💾 Persistent trained model bundles
* 🔙 Historical data backfilling
* 🧩 Modular and extensible architecture

---

# 🏗️ System Architecture

```text
                         ┌───────────────────────┐
                         │      Open-Meteo        │
                         │ Air Quality + Weather  │
                         └───────────┬───────────┘
                                     │
                                     │
                         ┌───────────▼───────────┐
                         │        AQICN           │
                         │ Ground-Truth AQI Data   │
                         └───────────┬───────────┘
                                     │
                                     ▼
                    ┌──────────────────────────────┐
                    │     Feature Pipeline          │
                    │                              │
                    │ • Data Fetching              │
                    │ • Time Features              │
                    │ • Lag Features               │
                    │ • Rolling Statistics         │
                    │ • AQI Change Rate            │
                    │ • Future Weather Features   │
                    │ • Forecast Targets           │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │     Supabase PostgreSQL       │
                    │        Feature Store          │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │      Training Pipeline       │
                    │                              │
                    │ Ridge Regression              │
                    │ Random Forest                │
                    │ LSTM                         │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────┴───────────────┐
                    │                              │
                    ▼                              ▼
          ┌──────────────────┐          ┌──────────────────┐
          │      MLflow      │          │  Saved Models    │
          │ Experiment       │          │                  │
          │ Tracking &       │          │ 24h / 48h / 72h │
          │ Registry         │          │                  │
          └──────────────────┘          └────────┬─────────┘
                                                  │
                                                  ▼
                                  ┌─────────────────────────┐
                                  │   Streamlit Dashboard   │
                                  │                         │
                                  │ • Current AQI           │
                                  │ • AQI Forecast          │
                                  │ • Weather Information   │
                                  │ • SHAP Explainability   │
                                  │ • Hazard Alerts         │
                                  └─────────────────────────┘
```

---

# 🔄 End-to-End Workflow

The system follows a complete ML lifecycle:

```text
Data Collection
      ↓
Data Validation
      ↓
Feature Engineering
      ↓
Feature Store
      ↓
Training Dataset
      ↓
Model Training
      ↓
Model Evaluation
      ↓
MLflow Experiment Tracking
      ↓
Best Model Selection
      ↓
Model Persistence
      ↓
Live Weather Enrichment
      ↓
AQI Prediction
      ↓
SHAP Explainability
      ↓
Dashboard / Alerts
```

---

# 📡 Data Sources

The project uses two primary data sources.

## 1. Open-Meteo

Open-Meteo provides both historical and current data without requiring an API key.

The system collects:

### Air-quality variables

* PM2.5
* PM10
* O₃
* NO₂
* SO₂
* CO
* US AQI

### Weather variables

* Temperature
* Relative humidity
* Wind speed
* Wind direction
* Atmospheric pressure

Open-Meteo is also used to obtain **future weather forecasts during live inference**.

---

## 2. AQICN / WAQI

AQICN provides current ground-truth station observations.

When an AQICN token is configured, the pipeline can retrieve:

* AQI
* PM2.5
* PM10
* O₃
* NO₂
* SO₂
* CO
* Temperature
* Humidity
* Wind
* Pressure

AQICN acts as a secondary data source and provides an additional real-world AQI measurement.

---

# 🧠 Feature Engineering

The feature engineering pipeline converts raw hourly observations into model-ready features.

The implementation generates several categories of features.

## Temporal Features

* `hour`
* `day_of_week`
* `day_of_month`
* `month`
* `is_weekend`

These features allow the model to capture recurring temporal patterns in air pollution.

---

## Historical AQI Features

The system creates:

```text
AQI lag 1 hour
AQI lag 24 hours
6-hour rolling AQI mean
AQI change rate
```

The AQI change rate is calculated as:

```text
AQI Change Rate =
(AQI_t - AQI_t-1) / AQI_t-1
```

with protection against division by zero.

---

# 🔮 Forecast Targets

Instead of training a single model for every possible future timestamp, the system creates separate forecasting targets:

```text
target_aqi_24h
target_aqi_48h
target_aqi_72h
```

These represent the AQI expected:

* 24 hours into the future
* 48 hours into the future
* 72 hours into the future

The targets are generated by shifting the AQI time series backward by the corresponding number of hours.

---

# 🌦️ Future Weather Features

One of the important aspects of the system is that it does not rely exclusively on the current weather conditions.

For each forecast horizon, future weather variables are included:

```text
Temperature
Humidity
Wind Speed
Pressure
```

For example:

```text
temp_future_24h
humidity_future_24h
wind_speed_future_24h
pressure_future_24h
```

and corresponding features for 48h and 72h.

During training, historical weather at the target timestamp is used as a proxy.

During live inference, these values are populated using an actual Open-Meteo weather forecast.

This gives the forecasting models information about the expected meteorological conditions when predicting future AQI.

---

# 🤖 Machine Learning Models

The training pipeline evaluates multiple model families for each forecast horizon.

## 1. Ridge Regression

Ridge Regression is used as a statistical baseline.

```text
Ridge Regression
α = 1.0
```

It provides a simple linear reference point against which the more complex models can be compared.

---

## 2. Random Forest

The project uses a Random Forest Regressor with:

```text
Estimators: 300
Maximum Depth: 12
Random State: 42
```

Random Forest is particularly useful for capturing nonlinear relationships between:

* pollutants
* weather
* temporal variables
* historical AQI
* future weather conditions

Tree-based models are trained without feature scaling.

---

## 3. LSTM

A lightweight LSTM architecture is included for deep-learning-based forecasting.

```text
Input
  ↓
LSTM(32)
  ↓
Dense(16, ReLU)
  ↓
Dense(1)
```

Training configuration:

```text
Optimizer: Adam
Loss: Mean Squared Error
Metric: MAE
Epochs: 30
Batch Size: 16
```

The LSTM provides a neural-network-based alternative to the classical regression and tree-based approaches.

---

# 🎯 Absolute vs Delta Forecasting

The training pipeline evaluates two prediction formulations:

### Absolute AQI

The model directly predicts:

```text
Future AQI
```

### Delta AQI

The model predicts:

```text
Future AQI - Current AQI
```

The predicted change is then reconstructed using the current AQI as the anchor.

```text
Predicted AQI =
Predicted Delta + Current AQI
```

The pipeline evaluates both approaches and selects the configuration with the lowest RMSE.

This is performed independently for each forecast horizon.

---

# 📊 Model Evaluation

Models are evaluated using:

| Metric | Description                  |
| ------ | ---------------------------- |
| RMSE   | Root Mean Squared Error      |
| MAE    | Mean Absolute Error          |
| R²     | Coefficient of Determination |

The best model for each forecasting horizon is selected according to **RMSE**.

The resulting models are stored separately for:

```text
24-hour forecast
48-hour forecast
72-hour forecast
```

---

# 🧪 MLflow Experiment Tracking

MLflow is integrated into the training pipeline to track experiments.

For every model run, the system records information such as:

* Model type
* Forecast horizon
* Target formulation
* Hyperparameters
* RMSE
* MAE
* R²
* Model artifacts

The training pipeline also registers the best-performing model for each horizon.

This makes it possible to compare different model configurations and maintain a reproducible training workflow.

---

# 💾 Model Persistence

The trained models are stored under:

```text
models/
```

Typical model artifacts include:

```text
models/
├── best_model_24h.pkl
├── best_model_48h.pkl
├── best_model_72h.pkl
└── best_model_24h_lstm.keras
```

The model bundles contain information such as:

* model
* model name
* scaler
* feature columns
* prediction formulation
* anchor column
* evaluation metrics
* training timestamp
* experimental/low-confidence flag

LSTM models are saved using the native Keras format rather than attempting to pickle the TensorFlow model.

---

# 🗃️ Database & Feature Store

The project uses **PostgreSQL**, with Supabase serving as the hosted database/feature-store layer.

The database contains three main tables.

## `raw_readings`

Stores the original data retrieved from external APIs.

```text
raw_readings
├── city
├── timestamp
├── source
├── PM2.5
├── PM10
├── O3
├── NO2
├── SO2
├── CO
├── AQI
├── weather variables
└── fetched_at
```

---

## `features`

Stores engineered, model-ready features.

This includes:

* temporal features
* pollutant measurements
* weather variables
* AQI lags
* rolling statistics
* AQI change rate
* future weather features
* forecast targets

---

## `predictions`

Stores generated predictions.

```text
predictions
├── city
├── predicted_for
├── horizon_days
├── predicted_aqi
├── model_name
├── model_version
└── created_at
```

The schema also defines indexes for efficient city/time-based querying.

---

# 🔙 Historical Data Backfilling

The project includes a dedicated backfill pipeline:

```text
backfill/
└── backfill_historical.py
```

This allows historical Open-Meteo data to be retrieved over a configurable date range and inserted into the feature-store workflow.

Example:

```bash
python backfill/backfill_historical.py --days 90
```

Historical data is particularly useful for generating sufficient training examples for the 24h, 48h, and 72h forecasting horizons.

---

# 📈 Streamlit Dashboard

The project includes an interactive Streamlit dashboard.

Run it locally with:

```bash
streamlit run dashboard/app.py
```

The dashboard provides an environmental monitoring interface containing:

### Current AQI

Displays:

* current US AQI
* AQI category
* PM2.5
* temperature
* humidity
* wind speed

### AQI Forecast

Displays predictions for:

```text
Day 1 → 24 hours
Day 2 → 48 hours
Day 3 → 72 hours
```

### AQI Severity

The dashboard categorizes AQI using familiar severity levels:

```text
Good
Moderate
Unhealthy for Sensitive Groups
Unhealthy
Very Unhealthy
Hazardous
```

### Forecast Confidence

Models with weak R² values are marked as experimental/low-confidence rather than presenting all predictions as equally reliable.

---

# 🧠 Explainability with SHAP

The dashboard integrates **SHAP** for model explainability.

The purpose is to help answer:

> **Why did the model make this AQI prediction?**

SHAP can be used to investigate the contribution of model inputs such as:

* PM2.5
* PM10
* historical AQI
* temperature
* humidity
* wind
* pressure
* temporal variables
* future weather conditions

This makes the system more interpretable than a black-box prediction interface.

---

# ⚙️ Automation

The project uses **GitHub Actions** for automated ML workflows.

```text
.github/
└── workflows/
    ├── feature_pipeline.yml
    └── training_pipeline.yml
```

The intended workflow is:

```text
Every Hour
    ↓
Fetch latest data
    ↓
Feature pipeline
    ↓
Update feature store


Every Day
    ↓
Load feature store
    ↓
Train models
    ↓
Evaluate models
    ↓
Track experiments
    ↓
Update best models
```

This allows the project to operate as an automated ML pipeline rather than requiring manual execution of every stage.

---

# 📁 Project Structure

```text
AQI-Predictor/
│
├── .github/
│   └── workflows/
│       ├── feature_pipeline.yml
│       └── training_pipeline.yml
│
├── alerts/
│   ├── __init__.py
│   └── notifier.py
│
├── backfill/
│   ├── __init__.py
│   └── backfill_historical.py
│
├── dashboard/
│   └── app.py
│
├── db/
│   ├── __init__.py
│   ├── db_utils.py
│   ├── migration_001_future_weather.sql
│   └── schema.sql
│
├── feature_pipeline/
│   ├── __init__.py
│   ├── fetch_data.py
│   ├── feature_engineering.py
│   └── run_feature_pipeline.py
│
├── models/
│   ├── .gitkeep
│   ├── best_model_24h.pkl
│   ├── best_model_24h_lstm.keras
│   ├── best_model_48h.pkl
│   └── best_model_72h.pkl
│
├── training_pipeline/
│   ├── __init__.py
│   ├── data_quality_check.py
│   ├── evaluate.py
│   └── train.py
│
├── config.py
│
├── requirements.txt
├── requirements-colab.txt
├── requirements-kaggle.txt
├── requirements-pipeline.txt
├── requirements-training.txt
└── README.md
```

---

# 🛠️ Technology Stack

| Component              | Technology         |
| ---------------------- | ------------------ |
| Programming Language   | Python             |
| Data Processing        | Pandas, NumPy      |
| Machine Learning       | Scikit-learn       |
| Deep Learning          | TensorFlow / Keras |
| Experiment Tracking    | MLflow             |
| Database               | PostgreSQL         |
| Hosted Database        | Supabase           |
| Data Sources           | Open-Meteo, AQICN  |
| Explainability         | SHAP               |
| Dashboard              | Streamlit          |
| Visualization          | Plotly, Matplotlib |
| Automation             | GitHub Actions     |
| Environment Management | python-dotenv      |

The core project dependencies are defined in `requirements.txt`.

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Raaid-Yousuf/AQI-Predictor.git

cd AQI-Predictor
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The project also provides specialized dependency files for different environments:

```text
requirements-colab.txt
requirements-kaggle.txt
requirements-pipeline.txt
requirements-training.txt
```

---

# 🔐 Configuration

Create a `.env` file in the project root.

Example:

```env
# City
CITY_NAME=Lahore
CITY_LAT=31.5497
CITY_LON=74.3436

# Database
DATABASE_URL=your_postgresql_connection_string

# AQICN
AQICN_TOKEN=your_aqicn_token

# MLflow
MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# Alerts
HAZARD_AQI_THRESHOLD=150

TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

ALERT_EMAIL_FROM=
ALERT_EMAIL_TO=
ALERT_EMAIL_APP_PASSWORD=
```

The default configuration targets Lahore, while city coordinates can be changed through environment variables.

> **Never commit your `.env` file or API credentials to GitHub.**

---

# 🗄️ Database Setup

Create a PostgreSQL/Supabase database and obtain its connection string.

Then execute:

```bash
psql "$DATABASE_URL" -f db/schema.sql
```

This creates the required:

```text
raw_readings
features
predictions
```

tables and their indexes.

---

# 📥 Populate Historical Data

To retrieve historical data:

```bash
python backfill/backfill_historical.py --days 90
```

You can increase the historical window if more training data is required.

---

# ⚙️ Run the Feature Pipeline

Run:

```bash
python feature_pipeline/run_feature_pipeline.py
```

The pipeline:

```text
Fetch Data
    ↓
Validate / Normalize
    ↓
Feature Engineering
    ↓
Store in PostgreSQL
```

---

# 🧠 Train the Models

Run:

```bash
python training_pipeline/train.py
```

The training pipeline will:

1. Load features from the feature store.
2. Prepare the forecasting datasets.
3. Create train/test splits.
4. Train Ridge Regression.
5. Train Random Forest.
6. Train LSTM.
7. Evaluate all models.
8. Compare absolute and delta forecasting.
9. Track experiments in MLflow.
10. Select the best model for each horizon.
11. Save the resulting model bundles.

---

# 📊 Run the Dashboard

Start Streamlit:

```bash
streamlit run dashboard/app.py
```

The dashboard can then be accessed through the local Streamlit URL displayed in the terminal.

---

# 🔬 Model Training Strategy

The system trains independent models for three forecasting horizons:

```text
                ┌─────────────┐
                │ Feature Set │
                └──────┬──────┘
                       │
          ┌────────────┼────────────┐
          ↓            ↓            ↓
       24 Hours     48 Hours     72 Hours
          │            │            │
          ↓            ↓            ↓
      Ridge/RF/LSTM Ridge/RF/LSTM Ridge/RF/LSTM
          │            │            │
          ↓            ↓            ↓
      Best Model    Best Model    Best Model
```

This allows the system to learn different relationships at different forecast horizons.

---

# 📐 Data Flow

A simplified data flow is:

```text
Open-Meteo ─────┐
                ├──► Raw Readings
AQICN ──────────┘
                      │
                      ▼
               Feature Engineering
                      │
                      ├── Time Features
                      ├── AQI Lags
                      ├── Rolling Statistics
                      ├── AQI Change Rate
                      ├── Weather Features
                      └── Future Weather
                      │
                      ▼
                Feature Store
                      │
                      ▼
                Model Training
                      │
                      ▼
              Model Evaluation
                      │
                      ▼
               Best Model
                      │
                      ▼
               Live Inference
                      │
                      ▼
              AQI Prediction
                      │
              ┌───────┴────────┐
              ▼                ▼
         Dashboard           Alerts
```

---

# 🌐 Live Inference

During live prediction, the system obtains the latest stored AQI and environmental features.

For future weather variables, the dashboard retrieves a real Open-Meteo weather forecast.

For example:

```text
Current timestamp
      ↓
+24 hours ──► Future weather
      ↓
+48 hours ──► Future weather
      ↓
+72 hours ──► Future weather
```

These values are inserted into the same feature structure expected by the trained models.

This ensures that live inference uses **forecasted weather rather than unavailable future observations**.

---

# 🧪 Data Science Considerations

## Time-Series Splitting

The training pipeline uses an ordered train/test split rather than randomly shuffling observations.

```text
Historical Data
│
├─────────────── Training ───────────────┤
│                                         │
│                              Test       │
└─────────────────────────────────────────┘
```

This better reflects the temporal nature of AQI forecasting.

---

## Forecast Horizon Difficulty

Predicting AQI further into the future is inherently more difficult.

Therefore:

```text
24h → Short-term forecast
48h → Medium-term forecast
72h → Longer-term forecast
```

The system explicitly evaluates each horizon separately rather than assuming that one model performs equally well at every prediction distance.

---

# ⚠️ Experimental Forecasts

The model bundle stores an `experimental` flag when the resulting R² is below a predefined threshold.

This allows the dashboard to communicate that certain forecasts may have lower reliability instead of presenting every prediction as equally trustworthy.

This is particularly important for longer forecasting horizons.

---

# 📌 Configuration Defaults

The main configuration is centralized in:

```text
config.py
```

Default values include:

```text
City: Lahore
Latitude: 31.5497
Longitude: 74.3436

Forecast Horizon: 3 days

AQI Hazard Threshold: 150

MLflow Experiment:
pearls-aqi-forecaster
```

All sensitive credentials are loaded through environment variables.

---

# 📊 Why This Project?

AQI forecasting is a practical machine-learning problem involving:

* Time-series modelling
* Environmental data
* Weather dynamics
* Feature engineering
* Regression
* Deep learning
* Model evaluation
* Explainable AI
* Automated ML pipelines
* Production-oriented deployment

The goal of Pearls AQI Predictor is therefore not simply to train a model that predicts AQI, but to demonstrate how an ML model can be incorporated into a **complete, automated, explainable forecasting system**.

---

# 👨‍💻 Author

**Raaid Yousuf**

Software Engineering
UET Taxila

GitHub:
https://github.com/Raaid-Yousuf

Project Repository:
https://github.com/Raaid-Yousuf/AQI-Predictor

---

# 📄 License

This project is intended for educational, research, and experimental purposes.

If you plan to use the project commercially or redistribute it, add an appropriate open-source license such as MIT before doing so.

---

# ⭐ Acknowledgements

This project makes use of:

* Open-Meteo for weather and air-quality data
* AQICN / World Air Quality Index for current AQI observations
* Supabase/PostgreSQL for data storage
* Scikit-learn for classical machine learning
* TensorFlow/Keras for LSTM modelling
* MLflow for experiment tracking
* SHAP for model explainability
* Streamlit for the interactive dashboard
* GitHub Actions for pipeline automation

---

## 🔗 Repository

**Pearls AQI Predictor**

https://github.com/Raaid-Yousuf/AQI-Predictor
