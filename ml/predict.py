"""
Inference module — loads the trained model and returns revenue predictions.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

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
    model = joblib.load(MODEL_PATH)
    meta = joblib.load(META_PATH)
    return model, meta


def predict_revenue(
    category: str,
    region: str,
    month: int,
    quarter: Optional[int] = None,
    day_of_week: int = 0,
    week_of_year: int = 1,
) -> dict:
    """
    Predict revenue for a given category / region / time combination.

    Returns a dict with:
        predicted_revenue  - model output
        model_mae          - training MAE for confidence context
        model_r2           - training R²
    """
    model, meta = _load_model()
    feature_cols: list[str] = meta["feature_cols"]

    if quarter is None:
        quarter = (month - 1) // 3 + 1

    # Build a row with the same structure used during training
    row = {col: 0 for col in feature_cols}
    row["month"] = month
    row["quarter"] = quarter
    row["day_of_week"] = day_of_week
    row["week_of_year"] = week_of_year

    cat_col = f"category_{category}"
    region_col = f"region_{region}"
    if cat_col in row:
        row[cat_col] = 1
    if region_col in row:
        row[region_col] = 1

    X = pd.DataFrame([row])
    predicted = float(model.predict(X)[0])

    return {
        "predicted_revenue": round(predicted, 2),
        "model_mae": round(meta["mae"], 2),
        "model_r2": round(meta["r2"], 4),
    }


def model_is_ready() -> bool:
    return os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)
