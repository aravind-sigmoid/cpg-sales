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

from ml.features import aggregate_monthly, build_features, get_feature_columns
from ml.predict import predict_revenue


# ── Sample data helpers ───────────────────────────────────────────────────────

def _raw_txn_df(n=200):
    """Simulate raw transaction rows as returned from Postgres."""
    dates = pd.date_range("2022-01-01", periods=n, freq="3D")
    return pd.DataFrame({
        "transaction_date": dates,
        "category": np.random.choice(["Beverages", "Snacks", "Dairy"], n),
        "region": np.random.choice(["Northeast", "West", "Midwest"], n),
        "revenue": np.random.uniform(10, 500, n),
    })


# ── Aggregation ───────────────────────────────────────────────────────────────

def test_aggregate_monthly_reduces_rows():
    raw = _raw_txn_df(200)
    agg = aggregate_monthly(raw)
    assert len(agg) < len(raw)
    assert "total_revenue" in agg.columns
    assert "transaction_count" in agg.columns


def test_aggregate_monthly_adds_time_columns():
    raw = _raw_txn_df(200)
    agg = aggregate_monthly(raw)
    for col in ["year", "month", "quarter"]:
        assert col in agg.columns


def test_aggregate_monthly_revenue_sums_correctly():
    raw = pd.DataFrame({
        "transaction_date": ["2024-01-10", "2024-01-20", "2024-02-05"],
        "category": ["Beverages", "Beverages", "Beverages"],
        "region": ["Northeast", "Northeast", "Northeast"],
        "revenue": [100.0, 200.0, 150.0],
    })
    agg = aggregate_monthly(raw)
    jan = agg[(agg["month"] == 1) & (agg["year"] == 2024)]
    assert abs(jan["total_revenue"].values[0] - 300.0) < 0.01


# ── Feature engineering ───────────────────────────────────────────────────────

def test_build_features_ohe_categoricals():
    raw = _raw_txn_df(200)
    agg = aggregate_monthly(raw)
    feat_df = build_features(agg)
    ohe_cols = [c for c in feat_df.columns if c.startswith("category_") or c.startswith("region_")]
    assert len(ohe_cols) > 0


def test_get_feature_columns_includes_numeric():
    raw = _raw_txn_df(200)
    agg = aggregate_monthly(raw)
    feat_df = build_features(agg)
    cols = get_feature_columns(feat_df)
    assert "month" in cols
    assert "quarter" in cols
    assert "year" in cols


def test_no_missing_values_in_features():
    raw = _raw_txn_df(200)
    agg = aggregate_monthly(raw)
    feat_df = build_features(agg)
    feature_cols = get_feature_columns(feat_df)
    assert feat_df[feature_cols].isna().sum().sum() == 0


# ── Prediction ────────────────────────────────────────────────────────────────

def _make_temp_model(feature_cols):
    X = pd.DataFrame(np.zeros((20, len(feature_cols))), columns=feature_cols)
    y = np.random.uniform(5000, 50000, 20)
    model = RandomForestRegressor(n_estimators=2, random_state=42)
    model.fit(X, y)
    return model, {"feature_cols": feature_cols, "mae": 1000.0, "r2": 0.85, "training_rows": 20}


def test_predict_revenue_returns_expected_keys():
    raw = _raw_txn_df(200)
    agg = aggregate_monthly(raw)
    feat_df = build_features(agg)
    feature_cols = get_feature_columns(feat_df)
    model, meta = _make_temp_model(feature_cols)

    with patch("ml.predict._load_model", return_value=(model, meta)):
        result = predict_revenue(category="Beverages", region="Northeast", month=7, year=2025)

    assert "predicted_revenue" in result
    assert "model_mae" in result
    assert "model_r2" in result
    assert result["predicted_revenue"] >= 0


def test_predict_revenue_unknown_category_returns_float():
    """Unknown category should still return a non-negative prediction."""
    raw = _raw_txn_df(200)
    agg = aggregate_monthly(raw)
    feat_df = build_features(agg)
    feature_cols = get_feature_columns(feat_df)
    model, meta = _make_temp_model(feature_cols)

    with patch("ml.predict._load_model", return_value=(model, meta)):
        result = predict_revenue(category="NonExistent", region="Northeast", month=3, year=2025)

    assert isinstance(result["predicted_revenue"], float)
    assert result["predicted_revenue"] >= 0
