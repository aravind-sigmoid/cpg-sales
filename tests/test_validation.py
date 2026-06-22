"""Unit tests for the ingestion validation layer."""
import pandas as pd
import pytest

from ingestion.validate import validate_transactions


def _base_df():
    return pd.DataFrame({
        "transaction_id": [f"TXN-{i:04d}" for i in range(1, 6)],
        "transaction_date": ["2024-01-15 10:00:00"] * 5,
        "sku": ["SKU-0001"] * 5,
        "quantity": [3, 5, 2, 4, 1],
        "unit_price": [10.0, 20.0, 15.0, 8.0, 25.0],
        "store_id": ["STORE-001"] * 5,
        "source_system": ["POS_NORTH"] * 5,
    })


# ── Basic happy path ──────────────────────────────────────────────────────────

def test_valid_rows_pass():
    df = _base_df()
    valid, rejected, stats = validate_transactions(df)
    assert len(valid) == 5
    assert len(rejected) == 0
    assert stats.rows_valid == 5
    assert stats.rows_rejected == 0


def test_revenue_computed():
    df = _base_df()
    valid, _, _ = validate_transactions(df)
    for _, row in valid.iterrows():
        assert abs(row["revenue"] - row["quantity"] * row["unit_price"]) < 0.01


# ── Null handling ─────────────────────────────────────────────────────────────

def test_null_quantity_rejected():
    df = _base_df()
    df.loc[0, "quantity"] = None
    valid, rejected, stats = validate_transactions(df)
    assert len(valid) == 4
    assert stats.rows_rejected == 1
    assert "null_quantity" in stats.rejection_reasons


def test_null_price_rejected():
    df = _base_df()
    df.loc[2, "unit_price"] = None
    valid, rejected, stats = validate_transactions(df)
    assert len(valid) == 4
    assert "null_unit_price" in stats.rejection_reasons


# ── Negative quantity ─────────────────────────────────────────────────────────

def test_negative_quantity_rejected():
    df = _base_df()
    df.loc[1, "quantity"] = -5
    valid, rejected, stats = validate_transactions(df)
    assert "non_positive_quantity" in stats.rejection_reasons
    assert all(valid["quantity"] > 0)


# ── Date format normalisation ─────────────────────────────────────────────────

def test_alternate_date_format_accepted():
    df = _base_df()
    df.loc[0, "transaction_date"] = "15/01/2024"   # DD/MM/YYYY
    valid, rejected, stats = validate_transactions(df)
    assert len(valid) == 5
    assert pd.api.types.is_datetime64_any_dtype(valid["transaction_date"])


def test_unparseable_date_rejected():
    df = _base_df()
    df.loc[0, "transaction_date"] = "not-a-date"
    valid, rejected, stats = validate_transactions(df)
    assert len(valid) == 4
    assert "unparseable_date" in stats.rejection_reasons


# ── Store ID normalisation ────────────────────────────────────────────────────

def test_store_id_casing_normalised():
    df = _base_df()
    df.loc[0, "store_id"] = "store-001"   # lowercase drift
    valid, _, _ = validate_transactions(df)
    assert all(valid["store_id"] == valid["store_id"].str.upper())


# ── Deduplication ─────────────────────────────────────────────────────────────

def test_duplicates_removed():
    df = _base_df()
    dup = df.iloc[:2].copy()
    combined = pd.concat([df, dup], ignore_index=True)
    valid, _, stats = validate_transactions(combined)
    assert stats.rows_duplicate == 2
    assert len(valid) == 5


# ── Missing required column raises ───────────────────────────────────────────

def test_missing_column_raises():
    df = _base_df().drop(columns=["sku"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_transactions(df)
