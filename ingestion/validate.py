"""
Data validation rules for raw transaction CSVs.

Returns a tuple (valid_df, rejected_df, stats_dict) so the caller
can persist both outcomes and the audit log.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Acceptable date formats from source systems
DATE_FORMATS = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"]

REQUIRED_COLUMNS = {"transaction_id", "transaction_date", "sku", "quantity", "unit_price", "store_id"}


@dataclass
class ValidationStats:
    rows_received: int = 0
    rows_valid: int = 0
    rows_rejected: int = 0
    rows_duplicate: int = 0
    rejection_reasons: dict = field(default_factory=dict)

    def add_rejection(self, reason: str, count: int) -> None:
        count = int(count)  # guard against numpy.int64 from pd.Series.sum()
        self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + count
        self.rows_rejected += count


def _parse_dates(series: pd.Series) -> pd.Series:
    """Try multiple date formats; return NaT for unparseable values."""
    parsed = pd.Series(pd.NaT, index=series.index)
    remaining_mask = pd.Series(True, index=series.index)

    for fmt in DATE_FORMATS:
        if not remaining_mask.any():
            break
        try:
            attempt = pd.to_datetime(
                series[remaining_mask], format=fmt, errors="coerce"
            )
            success = attempt.notna()
            parsed[remaining_mask & success.reindex(series.index, fill_value=False)] = \
                attempt[success].values
            remaining_mask[remaining_mask] = ~success.values
        except Exception:
            continue

    # Final pass with pandas inference for anything still remaining
    if remaining_mask.any():
        parsed[remaining_mask] = pd.to_datetime(
            series[remaining_mask], errors="coerce"
        )

    return parsed


def validate_transactions(
    df: pd.DataFrame,
    valid_skus: set[str] | None = None,
    valid_store_ids: set[str] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, ValidationStats]:
    """
    Clean and validate a raw transactions DataFrame.

    Returns:
        valid_df      - rows that passed all checks
        rejected_df   - rows that failed, with a `rejection_reason` column
        stats         - ValidationStats summary
    """
    stats = ValidationStats(rows_received=len(df))
    rejection_frames = []

    df = df.copy()

    # ── 1. Check required columns exist ──────────────────────────────────────
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Input missing required columns: {missing_cols}")

    # ── 2. Normalise store_id casing (schema drift fix) ───────────────────────
    df["store_id"] = df["store_id"].str.upper().str.strip()

    # ── 3. Parse & validate transaction_date ─────────────────────────────────
    df["transaction_date"] = _parse_dates(df["transaction_date"].astype(str))
    bad_dates = df["transaction_date"].isna()
    if bad_dates.any():
        bad = df[bad_dates].copy()
        bad["rejection_reason"] = "unparseable_date"
        rejection_frames.append(bad)
        stats.add_rejection("unparseable_date", bad_dates.sum())
        df = df[~bad_dates]

    # ── 4. Drop rows with null quantity or unit_price ─────────────────────────
    null_qty = df["quantity"].isna()
    if null_qty.any():
        bad = df[null_qty].copy()
        bad["rejection_reason"] = "null_quantity"
        rejection_frames.append(bad)
        stats.add_rejection("null_quantity", null_qty.sum())
        df = df[~null_qty]

    null_price = df["unit_price"].isna()
    if null_price.any():
        bad = df[null_price].copy()
        bad["rejection_reason"] = "null_unit_price"
        rejection_frames.append(bad)
        stats.add_rejection("null_unit_price", null_price.sum())
        df = df[~null_price]

    # ── 5. Cast types ─────────────────────────────────────────────────────────
    df["quantity"] = df["quantity"].astype(float).round().astype(int)
    df["unit_price"] = df["unit_price"].astype(float).round(2)

    # ── 6. Business rules: quantity and price must be positive ────────────────
    bad_qty = df["quantity"] <= 0
    if bad_qty.any():
        bad = df[bad_qty].copy()
        bad["rejection_reason"] = "non_positive_quantity"
        rejection_frames.append(bad)
        stats.add_rejection("non_positive_quantity", bad_qty.sum())
        df = df[~bad_qty]

    bad_price = df["unit_price"] <= 0
    if bad_price.any():
        bad = df[bad_price].copy()
        bad["rejection_reason"] = "non_positive_price"
        rejection_frames.append(bad)
        stats.add_rejection("non_positive_price", bad_price.sum())
        df = df[~bad_price]

    # ── 7. SKU referential integrity (optional) ───────────────────────────────
    if valid_skus:
        bad_sku = ~df["sku"].isin(valid_skus)
        if bad_sku.any():
            bad = df[bad_sku].copy()
            bad["rejection_reason"] = "unknown_sku"
            rejection_frames.append(bad)
            stats.add_rejection("unknown_sku", bad_sku.sum())
            df = df[~bad_sku]

    # ── 8. Store referential integrity (optional) ─────────────────────────────
    if valid_store_ids:
        bad_store = ~df["store_id"].isin(valid_store_ids)
        if bad_store.any():
            bad = df[bad_store].copy()
            bad["rejection_reason"] = "unknown_store"
            rejection_frames.append(bad)
            stats.add_rejection("unknown_store", bad_store.sum())
            df = df[~bad_store]

    # ── 9. Deduplication (transaction_id + source_system) ─────────────────────
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["transaction_id", "source_system"], keep="first")
    stats.rows_duplicate = int(before_dedup - len(df))

    # ── 10. Derive revenue ────────────────────────────────────────────────────
    df["revenue"] = (df["quantity"] * df["unit_price"]).round(2)

    stats.rows_valid = len(df)

    rejected_df = (
        pd.concat(rejection_frames, ignore_index=True)
        if rejection_frames
        else pd.DataFrame(columns=list(df.columns) + ["rejection_reason"])
    )

    logger.info(
        "Validation complete: %d received, %d valid, %d rejected, %d duplicates",
        stats.rows_received, stats.rows_valid, stats.rows_rejected, stats.rows_duplicate,
    )
    return df, rejected_df, stats
