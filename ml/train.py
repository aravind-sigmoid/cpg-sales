"""
Train the monthly revenue forecasting model.

Usage:
    python ml/train.py

Reads transactions from Postgres, aggregates to monthly category-region
totals, trains a RandomForestRegressor, and saves the artefact to
ml/model.joblib + ml/model_meta.joblib.
"""
from __future__ import annotations

import logging
import os
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from db.database import SessionLocal
from ml.features import aggregate_monthly, build_features, get_feature_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
META_PATH = os.path.join(os.path.dirname(__file__), "model_meta.joblib")


def load_training_data(session) -> pd.DataFrame:
    """Pull transactions with category from Postgres."""
    query = """
        SELECT
            t.transaction_date,
            t.revenue,
            t.region,
            p.category
        FROM sales_transactions t
        LEFT JOIN product_catalog p ON t.sku = p.sku
        WHERE p.category IS NOT NULL
    """
    df = pd.read_sql(query, session.bind)
    logger.info("Loaded %d raw transactions for training.", len(df))
    return df


def train(df: pd.DataFrame) -> None:
    # Aggregate to monthly grain: (category, region, year, month) → total_revenue
    monthly = aggregate_monthly(df)
    logger.info("Aggregated to %d monthly category-region rows.", len(monthly))

    if len(monthly) < 30:
        raise ValueError(
            "Not enough aggregated rows to train. "
            "Run ingestion with more data first."
        )

    feat_df = build_features(monthly)
    feature_cols = get_feature_columns(feat_df)

    X = feat_df[feature_cols].fillna(0)
    y = feat_df["total_revenue"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logger.info("Test MAE: $%.2f | R²: %.3f", mae, r2)

    joblib.dump(model, MODEL_PATH)
    meta = {
        "feature_cols": feature_cols,
        "mae": mae,
        "r2": r2,
        "training_rows": len(monthly),
    }
    joblib.dump(meta, META_PATH)
    logger.info("Model saved to %s", MODEL_PATH)


if __name__ == "__main__":
    session = SessionLocal()
    try:
        df = load_training_data(session)
        train(df)
    finally:
        session.close()
