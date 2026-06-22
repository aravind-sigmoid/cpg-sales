"""
/data endpoints — query transactions and business metrics.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import SalesTransaction

router = APIRouter(prefix="/data", tags=["data"])


@router.get("/metrics")
def get_metrics(db: Session = Depends(get_db)):
    """High-level KPIs: total revenue, transactions, top category, top region."""
    query = """
        SELECT
            COUNT(*)                    AS total_transactions,
            ROUND(SUM(revenue)::numeric, 2) AS total_revenue,
            ROUND(AVG(revenue)::numeric, 2) AS avg_revenue_per_txn
        FROM sales_transactions
    """
    row = db.execute(__import__("sqlalchemy").text(query)).fetchone()

    cat_query = """
        SELECT p.category, ROUND(SUM(t.revenue)::numeric, 2) AS revenue
        FROM sales_transactions t
        JOIN product_catalog p ON t.sku = p.sku
        GROUP BY p.category
        ORDER BY revenue DESC
        LIMIT 5
    """
    top_cats = db.execute(__import__("sqlalchemy").text(cat_query)).fetchall()

    region_query = """
        SELECT region, ROUND(SUM(revenue)::numeric, 2) AS revenue
        FROM sales_transactions
        GROUP BY region
        ORDER BY revenue DESC
        LIMIT 5
    """
    top_regions = db.execute(__import__("sqlalchemy").text(region_query)).fetchall()

    return {
        "total_transactions": row.total_transactions,
        "total_revenue": float(row.total_revenue or 0),
        "avg_revenue_per_txn": float(row.avg_revenue_per_txn or 0),
        "top_categories": [{"category": r.category, "revenue": float(r.revenue)} for r in top_cats],
        "top_regions": [{"region": r.region, "revenue": float(r.revenue)} for r in top_regions],
    }


@router.get("/transactions")
def get_transactions(
    region: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    limit: int = Query(100, le=1000),
    db: Session = Depends(get_db),
):
    """Paginated transaction query with optional filters."""
    import sqlalchemy as sa

    query = """
        SELECT
            t.transaction_id,
            t.transaction_date,
            t.sku,
            p.category,
            t.quantity,
            t.unit_price,
            t.revenue,
            t.region,
            t.store_id
        FROM sales_transactions t
        LEFT JOIN product_catalog p ON t.sku = p.sku
        WHERE 1=1
    """
    params: dict = {}

    if region:
        query += " AND t.region = :region"
        params["region"] = region
    if category:
        query += " AND p.category = :category"
        params["category"] = category
    if start_date:
        query += " AND t.transaction_date >= :start_date"
        params["start_date"] = start_date
    if end_date:
        query += " AND t.transaction_date <= :end_date"
        params["end_date"] = end_date

    query += " ORDER BY t.transaction_date DESC LIMIT :limit"
    params["limit"] = limit

    rows = db.execute(sa.text(query), params).fetchall()
    return [dict(r._mapping) for r in rows]


@router.get("/monthly-trend")
def monthly_trend(
    category: Optional[str] = Query(None),
    region: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """Monthly revenue trend, optionally filtered by category and region."""
    import sqlalchemy as sa

    query = """
        SELECT
            DATE_TRUNC('month', t.transaction_date) AS month,
            ROUND(SUM(t.revenue)::numeric, 2)        AS revenue,
            COUNT(*)                                  AS transactions
        FROM sales_transactions t
        LEFT JOIN product_catalog p ON t.sku = p.sku
        WHERE 1=1
    """
    params: dict = {}
    if category:
        query += " AND p.category = :category"
        params["category"] = category
    if region:
        query += " AND t.region = :region"
        params["region"] = region

    query += " GROUP BY 1 ORDER BY 1"
    rows = db.execute(sa.text(query), params).fetchall()
    return [
        {
            "month": str(r.month)[:10],
            "revenue": float(r.revenue or 0),
            "transactions": r.transactions,
        }
        for r in rows
    ]
