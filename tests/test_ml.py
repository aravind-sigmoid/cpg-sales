"""Unit tests for ML feature engineering and prediction logic."""
from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from ml.features import build_features, get_feature_columns
from ml.predict import predict_revenue


# ── Feature engineering ───────────────────────────────────────────────────────

def _sample_df(n=50):
    return pd.DataFrame({
        "transaction_date": pd.date_range("2024-01-01", periods=n, freq="W"),
        "category": np.random.choice(["Beverages", "Snacks", "Dairy"], n),
        "region": np.random.choice(["Northeast", "West", "Midwest"], n),
        "revenue": np.random.uniform(10, 500, n),
    })


def test_build_features_adds_time_columns():
    df = _sample_df()
    result = build_features(df)
    for col in ["month", "quarter", "day_of_week", "week_of_year"]:
        assert col in result.columns


def test_build_features_ohe_categoricals():
    df = _sample_df()
    result = build_features(df)
    # One-hot encoded columns should be present
    ohe_cols = [c for c in result.columns if c.startswith("category_") or c.startswith("region_")]
    assert len(ohe_cols) > 0


def test_get_feature_columns_returns_list():
    df = _sample_df()
    feat_df = build_features(df)
    cols = get_feature_columns(feat_df)
    assert isinstance(cols, list)
    assert len(cols) > 0
    assert "month" in cols


def test_no_missing_values_in_features():
    df = _sample_df()
    feat_df = build_features(df)
    feature_cols = get_feature_columns(feat_df)
    assert feat_df[feature_cols].isna().sum().sum() == 0


# ── Prediction ────────────────────────────────────────────────────────────────

def _make_temp_model(feature_cols):
    """Create and save a minimal model to a temp directory."""
    X = pd.DataFrame(np.zeros((10, len(feature_cols))), columns=feature_cols)
    y = np.random.uniform(100, 500, 10)
    model = RandomForestRegressor(n_estimators=2, random_state=42)
    model.fit(X, y)
    return model, {"feature_cols": feature_cols, "mae": 50.0, "r2": 0.85}


def test_predict_revenue_returns_expected_keys():
    df = _sample_df(100)
    feat_df = build_features(df)
    feature_cols = get_feature_columns(feat_df)
    model, meta = _make_temp_model(feature_cols)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.joblib")
        meta_path = os.path.join(tmpdir, "model_meta.joblib")
        joblib.dump(model, model_path)
        joblib.dump(meta, meta_path)

        with patch("ml.predict.MODEL_PATH", model_path), \
             patch("ml.predict.META_PATH", meta_path), \
             patch("ml.predict._load_model", return_value=(model, meta)):
            result = predict_revenue(
                category="Beverages",
                region="Northeast",
                month=7,
            )

    assert "predicted_revenue" in result
    assert "model_mae" in result
    assert "model_r2" in result
    assert result["predicted_revenue"] > 0


def test_predict_revenue_unknown_category_defaults():
    """Unknown category should still return a prediction (just with no OHE match)."""
    df = _sample_df(100)
    feat_df = build_features(df)
    feature_cols = get_feature_columns(feat_df)
    model, meta = _make_temp_model(feature_cols)

    with patch("ml.predict.MODEL_PATH", "/tmp/m.joblib"), \
         patch("ml.predict.META_PATH", "/tmp/mm.joblib"), \
         patch("ml.predict._load_model", return_value=(model, meta)):
        result = predict_revenue(
            category="NonExistentCategory",
            region="Northeast",
            month=3,
        )
    assert isinstance(result["predicted_revenue"], float)
