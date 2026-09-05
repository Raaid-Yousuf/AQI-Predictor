"""
Diagnostic script — run this BEFORE adding more data or retraining.
Answers two questions honestly:

1. Baseline check: does a dumb "AQI(t+h) = AQI(t) right now" persistence model
   beat our trained models? If yes, our models aren't learning real signal yet.
2. Cross-validated check: using TimeSeriesSplit (multiple rolling folds) instead
   of one single train/test split, are Ridge/RandomForest actually learning
   something, or was our earlier single-split R2 just an unlucky slice?

This is intentionally lightweight — no TensorFlow/LSTM here, so it runs fine
locally on your machine. No need for Colab/Kaggle for this step.

Usage:
    python training_pipeline/diagnose.py
"""
import sys
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CITY_NAME
from db.db_utils import get_features
from training_pipeline.evaluate import compute_metrics
from training_pipeline.train import get_feature_cols, HORIZONS


def persistence_baseline(data: pd.DataFrame, target_col: str) -> dict:
    """Predicted AQI(t+h) = AQI(t) right now, i.e. just use aqi_lag_1h as-is."""
    y_true = data[target_col].values
    y_pred = data["aqi_lag_1h"].values  # "no change" prediction
    return compute_metrics(y_true, y_pred)


def cross_validated_scores(data: pd.DataFrame, feature_cols: list, target_col: str,
                            n_splits: int = 5, predict_delta: bool = False) -> dict:
    X = data[feature_cols].values
    y = data[target_col].values
    current_aqi = data["aqi_lag_1h"].values  # anchor for delta reconstruction

    tscv = TimeSeriesSplit(n_splits=n_splits)
    ridge_scores, rf_scores = [], []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        anchor_train, anchor_test = current_aqi[train_idx], current_aqi[test_idx]

        if predict_delta:
            y_train_fit = y_train - anchor_train
        else:
            y_train_fit = y_train

        scaler = StandardScaler().fit(X_train)
        X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

        ridge = Ridge(alpha=1.0).fit(X_train_s, y_train_fit)
        ridge_raw_pred = ridge.predict(X_test_s)
        ridge_pred = ridge_raw_pred + anchor_test if predict_delta else ridge_raw_pred
        ridge_metrics = compute_metrics(y_test, ridge_pred)
        ridge_scores.append(ridge_metrics)

        rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train_fit)
        rf_raw_pred = rf.predict(X_test)
        rf_pred = rf_raw_pred + anchor_test if predict_delta else rf_raw_pred
        rf_metrics = compute_metrics(y_test, rf_pred)
        rf_scores.append(rf_metrics)

        print(f"    Fold {fold+1}: train={len(train_idx)} test={len(test_idx)} "
              f"| Ridge R2={ridge_metrics['r2']:.3f} | RF R2={rf_metrics['r2']:.3f}")

    def avg(scores, key):
        return float(np.mean([s[key] for s in scores]))

    return {
        "ridge": {k: avg(ridge_scores, k) for k in ["rmse", "mae", "r2"]},
        "random_forest": {k: avg(rf_scores, k) for k in ["rmse", "mae", "r2"]},
    }


def main():
    df = get_features(CITY_NAME)
    df["is_weekend"] = df["is_weekend"].astype(int)
    print(f"Loaded {len(df)} feature rows for {CITY_NAME}.\n")

    for horizon_name, target_col in HORIZONS.items():
        feature_cols = get_feature_cols(horizon_name)
        data = df.dropna(subset=feature_cols + [target_col]).copy()
        print(f"=== Horizon: {horizon_name} ({len(data)} usable rows) ===")

        if len(data) < 100:
            print("  Not enough data for a meaningful diagnostic (need 100+ rows). Skipping.\n")
            continue

        baseline = persistence_baseline(data, target_col)
        print(f"  Persistence baseline (predict = current AQI): {baseline}")

        print("  Running 5-fold TimeSeriesSplit CV — predicting ABSOLUTE AQI level...")
        cv_absolute = cross_validated_scores(data, feature_cols, target_col, predict_delta=False)
        print(f"  CV avg (absolute) -> Ridge: {cv_absolute['ridge']}")
        print(f"  CV avg (absolute) -> RandomForest: {cv_absolute['random_forest']}")

        print("  Running 5-fold TimeSeriesSplit CV — predicting DELTA from current AQI...")
        cv_delta = cross_validated_scores(data, feature_cols, target_col, predict_delta=True)
        print(f"  CV avg (delta) -> Ridge: {cv_delta['ridge']}")
        print(f"  CV avg (delta) -> RandomForest: {cv_delta['random_forest']}")

        best_absolute_rmse = min(cv_absolute["ridge"]["rmse"], cv_absolute["random_forest"]["rmse"])
        best_delta_rmse = min(cv_delta["ridge"]["rmse"], cv_delta["random_forest"]["rmse"])
        best_overall_rmse = min(baseline["rmse"], best_absolute_rmse, best_delta_rmse)

        if best_overall_rmse == baseline["rmse"]:
            print("  ⚠️  VERDICT: Persistence baseline still wins even with delta modeling.")
        elif best_overall_rmse == best_delta_rmse:
            print("  ✅ VERDICT: DELTA modeling beats both the baseline and absolute-value modeling."
                  " This confirms the fix — switch training_pipeline/train.py to predict deltas.")
        else:
            print("  ℹ️  VERDICT: Absolute-value modeling already beats baseline — delta modeling not needed.")
        print()


if __name__ == "__main__":
    main()