"""SQLAlchemy ORM models for the CPG sales data warehouse."""
from datetime import datetime

from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Date,
    Boolean, Text, UniqueConstraint, Index,
)

from db.database import Base


class SalesTransaction(Base):
    """Cleaned, validated sales transaction events."""
    __tablename__ = "sales_transactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(64), nullable=False)
    transaction_date = Column(DateTime, nullable=False)
    sku = Column(String(32), nullable=False)
    quantity = Column(Integer, nullable=False)
    unit_price = Column(Float, nullable=False)
    revenue = Column(Float, nullable=False)          # quantity * unit_price
    region = Column(String(64), nullable=False)
    store_id = Column(String(32), nullable=False)
    source_system = Column(String(32), nullable=False)  # POS_NORTH, ONLINE, etc.
    ingested_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("transaction_id", "source_system", name="uq_txn_source"),
        Index("ix_txn_date", "transaction_date"),
        Index("ix_txn_sku", "sku"),
        Index("ix_txn_region", "region"),
    )


class ProductCatalog(Base):
    """Product master / slow-changing dimension."""
    __tablename__ = "product_catalog"

    sku = Column(String(32), primary_key=True)
    product_name = Column(String(128), nullable=False)
    category = Column(String(64), nullable=False)
    brand = Column(String(64), nullable=False)
    package_size = Column(String(32))
    list_price = Column(Float, nullable=False)
    launch_date = Column(Date)
    is_active = Column(Boolean, default=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("ix_product_category", "category"),
        Index("ix_product_brand", "brand"),
    )


class StoreRegion(Base):
    """Store-to-region reference table."""
    __tablename__ = "store_regions"

    store_id = Column(String(32), primary_key=True)
    store_name = Column(String(128), nullable=False)
    region = Column(String(64), nullable=False)
    state = Column(String(32), nullable=False)
    city = Column(String(64))
    demographic_segment = Column(String(32))   # urban, suburban, rural
    updated_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_store_region", "region"),
    )


class IngestionLog(Base):
    """Audit trail for every ingestion run."""
    __tablename__ = "ingestion_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_at = Column(DateTime, default=datetime.utcnow)
    source_file = Column(String(256))
    rows_received = Column(Integer, default=0)
    rows_valid = Column(Integer, default=0)
    rows_rejected = Column(Integer, default=0)
    rows_duplicate = Column(Integer, default=0)
    notes = Column(Text)
