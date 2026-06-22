"""
Inference module — predicts total monthly revenue for a
category / region / year+month combination.
"""
from __future__ import annotations

import os
from functools import lru_cache

import joblib
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
META_PATH = os.path.join(os.path.dirname(__file__), "model_meta.joblib")


@lru_cache(maxsize=1)
def _load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model not found. Run `python ml/train.py` first."
        )
    return joblib.load(MODEL_PATH), joblib.load(META_PATH)


def predict_revenue(
    category: str,
    region: str,
    month: int,
    year: int = 2025,
) -> dict:
    """
    Predict total monthly revenue for a category / region / month.

    Returns:
        predicted_revenue  - forecasted total revenue for that month
        model_mae          - training MAE (same units: $)
        model_r2           - training R²
    """
    model, meta = _load_model()
    feature_cols: list[str] = meta["feature_cols"]

    quarter = (month - 1) // 3 + 1

    row = {col: 0 for col in feature_cols}
    row["month"] = month
    row["quarter"] = quarter
    row["year"] = year

    cat_col = f"category_{category}"
    region_col = f"region_{region}"
    if cat_col in row:
        row[cat_col] = 1
    if region_col in row:
        row[region_col] = 1

    X = pd.DataFrame([row])
    predicted = float(model.predict(X)[0])

    return {
        "predicted_revenue": round(max(predicted, 0.0), 2),
        "model_mae": round(meta["mae"], 2),
        "model_r2": round(meta["r2"], 4),
    }


def model_is_ready() -> bool:
    return os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)
