"""
Feature engineering for the revenue forecasting model.

Takes a DataFrame of sales transactions and returns
X (feature matrix) and y (revenue target).
"""
from __future__ import annotations

import pandas as pd
import numpy as np


CATEGORICAL_FEATURES = ["category", "region"]
NUMERIC_FEATURES = ["month", "quarter", "day_of_week", "week_of_year"]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a transactions DataFrame (must contain transaction_date, category,
    region, revenue), return an enriched feature DataFrame.
    """
    df = df.copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])

    df["month"] = df["transaction_date"].dt.month
    df["quarter"] = df["transaction_date"].dt.quarter
    df["day_of_week"] = df["transaction_date"].dt.dayofweek
    df["week_of_year"] = df["transaction_date"].dt.isocalendar().week.astype(int)

    # One-hot encode categoricals
    df = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, drop_first=False)

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the feature column names after get_dummies encoding."""
    ohe_cols = [c for c in df.columns if any(c.startswith(f"{cat}_") for cat in CATEGORICAL_FEATURES)]
    return NUMERIC_FEATURES + ohe_cols
