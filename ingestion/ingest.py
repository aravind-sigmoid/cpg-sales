"""
Main ingestion pipeline.

Usage:
    python ingestion/ingest.py                    # ingest all data/ CSVs
    python ingestion/ingest.py --file path/to.csv # ingest a single file
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime

import pandas as pd
from sqlalchemy.dialects.postgresql import insert

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from db.database import SessionLocal, engine
from db.models import (
    Base, IngestionLog, ProductCatalog, SalesTransaction, StoreRegion,
)
from db.init_db import init_db
from ingestion.validate import validate_transactions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# ── Reference loaders ─────────────────────────────────────────────────────────

def load_product_catalog(session) -> set[str]:
    """Upsert product_catalog.csv and return set of known SKUs."""
    path = os.path.join(DATA_DIR, "product_catalog.csv")
    if not os.path.exists(path):
        logger.warning("product_catalog.csv not found — skipping.")
        return set()

    df = pd.read_csv(path)
    df["launch_date"] = pd.to_datetime(df["launch_date"], errors="coerce").dt.date
    df["is_active"] = df["is_active"].astype(bool)
    df["updated_at"] = datetime.utcnow()

    records = df.to_dict(orient="records")
    stmt = insert(ProductCatalog).values(records)
    stmt = stmt.on_conflict_do_update(
        index_elements=["sku"],
        set_={c.key: c for c in stmt.excluded if c.key != "sku"},
    )
    session.execute(stmt)
    session.commit()
    logger.info("Loaded %d products.", len(records))
    return set(df["sku"].tolist())


def load_store_regions(session) -> set[str]:
    """Upsert store_regions.csv and return set of known store IDs."""
    path = os.path.join(DATA_DIR, "store_regions.csv")
    if not os.path.exists(path):
        logger.warning("store_regions.csv not found — skipping.")
        return set()

    df = pd.read_csv(path)
    df["store_id"] = df["store_id"].str.upper().str.strip()
    df["updated_at"] = datetime.utcnow()

    records = df.to_dict(orient="records")
    stmt = insert(StoreRegion).values(records)
    stmt = stmt.on_conflict_do_update(
        index_elements=["store_id"],
        set_={c.key: c for c in stmt.excluded if c.key != "store_id"},
    )
    session.execute(stmt)
    session.commit()
    logger.info("Loaded %d stores.", len(records))
    return set(df["store_id"].tolist())


# ── Transaction loader ────────────────────────────────────────────────────────

def ingest_transactions(
    file_path: str,
    session,
    valid_skus: set[str],
    valid_store_ids: set[str],
) -> None:
    logger.info("Reading %s ...", file_path)
    raw = pd.read_csv(file_path)

    valid_df, rejected_df, stats = validate_transactions(
        raw,
        valid_skus=valid_skus or None,
        valid_store_ids=valid_store_ids or None,
    )

    if not rejected_df.empty:
        rej_path = file_path.replace(".csv", "_rejected.csv")
        rejected_df.to_csv(rej_path, index=False)
        logger.warning(
            "Wrote %d rejected rows to %s", len(rejected_df), rej_path
        )

    # Bulk-insert valid rows; skip existing transaction_id+source_system pairs
    if not valid_df.empty:
        source_system = (
            valid_df["source_system"].iloc[0]
            if "source_system" in valid_df.columns
            else "UNKNOWN"
        )
        records = [
            {
                "transaction_id": row.transaction_id,
                "transaction_date": row.transaction_date,
                "sku": row.sku,
                "quantity": int(row.quantity),
                "unit_price": float(row.unit_price),
                "revenue": float(row.revenue),
                "region": _lookup_region(row.store_id, session),
                "store_id": row.store_id,
                "source_system": row.source_system if hasattr(row, "source_system") else "UNKNOWN",
                "ingested_at": datetime.utcnow(),
            }
            for row in valid_df.itertuples()
        ]
        stmt = insert(SalesTransaction).values(records)
        stmt = stmt.on_conflict_do_nothing(
            constraint="uq_txn_source"
        )
        session.execute(stmt)
        session.commit()
        logger.info("Inserted up to %d transaction rows.", len(records))

    # Audit log
    log_entry = IngestionLog(
        source_file=os.path.basename(file_path),
        rows_received=stats.rows_received,
        rows_valid=stats.rows_valid,
        rows_rejected=stats.rows_rejected,
        rows_duplicate=stats.rows_duplicate,
        notes=str(stats.rejection_reasons),
    )
    session.add(log_entry)
    session.commit()


def _lookup_region(store_id: str, session) -> str:
    store = session.get(StoreRegion, store_id)
    return store.region if store else "Unknown"


# ── Entry point ───────────────────────────────────────────────────────────────

def run(transaction_file: str | None = None) -> None:
    init_db()
    session = SessionLocal()
    try:
        valid_skus = load_product_catalog(session)
        valid_store_ids = load_store_regions(session)

        if transaction_file:
            files = [transaction_file]
        else:
            files = [
                os.path.join(DATA_DIR, f)
                for f in os.listdir(DATA_DIR)
                if f.endswith(".csv") and "transaction" in f.lower() and "rejected" not in f.lower()
            ]

        if not files:
            logger.warning("No transaction files found in data/. Run data/generate_data.py first.")
            return

        for f in files:
            ingest_transactions(f, session, valid_skus, valid_store_ids)

    finally:
        session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPG Sales Ingestion Pipeline")
    parser.add_argument("--file", default=None, help="Path to a specific transactions CSV")
    args = parser.parse_args()
    run(transaction_file=args.file)
