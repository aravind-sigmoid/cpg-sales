"""
Feature engineering for the monthly revenue forecasting model.

Aggregates transaction-level data to (category, region, year, month)
grain before building the feature matrix the model trains on.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

CATEGORICAL_FEATURES = ["category", "region"]
NUMERIC_FEATURES = ["month", "quarter", "year"]


def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse raw transactions to monthly category-region totals.

    Input columns required: transaction_date, category, region, revenue.
    Returns one row per (category, region, year, month) with:
        total_revenue, transaction_count, avg_revenue_per_txn
    """
    df = df.copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df["year"] = df["transaction_date"].dt.year
    df["month"] = df["transaction_date"].dt.month

    agg = (
        df.groupby(["category", "region", "year", "month"], as_index=False)
        .agg(
            total_revenue=("revenue", "sum"),
            transaction_count=("revenue", "count"),
            avg_revenue_per_txn=("revenue", "mean"),
        )
    )
    agg["quarter"] = ((agg["month"] - 1) // 3 + 1).astype(int)
    return agg


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode categoricals on the aggregated DataFrame.
    Input must already be at monthly grain (output of aggregate_monthly).
    """
    df = df.copy()
    df = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, drop_first=False)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return feature column names after OHE."""
    ohe_cols = [
        c for c in df.columns
        if any(c.startswith(f"{cat}_") for cat in CATEGORICAL_FEATURES)
    ]
    return NUMERIC_FEATURES + ohe_cols
