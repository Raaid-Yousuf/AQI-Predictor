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
from training_pipeline.train import FEATURE_COLS, HORIZONS


def persistence_baseline(data: pd.DataFrame, target_col: str) -> dict:
    """Predicted AQI(t+h) = AQI(t) right now, i.e. just use aqi_lag_1h as-is."""
    y_true = data[target_col].values
    y_pred = data["aqi_lag_1h"].values  # "no change" prediction
    return compute_metrics(y_true, y_pred)


def cross_validated_scores(data: pd.DataFrame, target_col: str, n_splits: int = 5) -> dict:
    X = data[FEATURE_COLS].values
    y = data[target_col].values

    tscv = TimeSeriesSplit(n_splits=n_splits)
    ridge_scores, rf_scores = [], []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler().fit(X_train)
        X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

        ridge = Ridge(alpha=1.0).fit(X_train_s, y_train)
        ridge_metrics = compute_metrics(y_test, ridge.predict(X_test_s))
        ridge_scores.append(ridge_metrics)

        rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_metrics = compute_metrics(y_test, rf.predict(X_test))
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
        data = df.dropna(subset=FEATURE_COLS + [target_col]).copy()
        print(f"=== Horizon: {horizon_name} ({len(data)} usable rows) ===")

        if len(data) < 100:
            print("  Not enough data for a meaningful diagnostic (need 100+ rows). Skipping.\n")
            continue

        baseline = persistence_baseline(data, target_col)
        print(f"  Persistence baseline (predict = current AQI): {baseline}")

        print("  Running 5-fold TimeSeriesSplit cross-validation...")
        cv_results = cross_validated_scores(data, target_col)
        print(f"  CV avg -> Ridge: {cv_results['ridge']}")
        print(f"  CV avg -> RandomForest: {cv_results['random_forest']}")

        best_model_rmse = min(baseline["rmse"], cv_results["ridge"]["rmse"], cv_results["random_forest"]["rmse"])
        if best_model_rmse == baseline["rmse"]:
            print("  ⚠️  VERDICT: Persistence baseline WINS. Trained models are not yet beating"
                  " a naive guess — points to a data volume/regime issue, not necessarily a code bug.")
        else:
            print("  ✅ VERDICT: A trained model beats the naive baseline — the pipeline"
                  " is learning real signal; earlier negative R2 was likely just an unlucky single split.")
        print()


if __name__ == "__main__":
    main()
