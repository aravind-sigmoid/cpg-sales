"""
/insights endpoints — LLM-powered natural language insights over sales data.
"""
from __future__ import annotations

import logging

import sqlalchemy as sa
from fastapi import APIRouter, Depends, HTTPException
from openai import OpenAI
from pydantic import BaseModel
from sqlalchemy.orm import Session

from config import settings
from db.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/insights", tags=["insights"])

_openai_client: OpenAI | None = None


def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=settings.openai_api_key)
    return _openai_client


def _call_llm(system_prompt: str, user_message: str) -> str:
    """Thin wrapper so we can swap models or providers in one place."""
    client = get_openai_client()
    try:
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=500,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logger.error("OpenAI call failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=f"LLM call failed: {exc}. Check your OPENAI_API_KEY.",
        )


# ── Data context helpers ───────────────────────────────────────────────────────

def _get_summary_context(db: Session) -> str:
    """Build a concise text summary of DB metrics for the LLM context."""
    metrics_query = sa.text("""
        SELECT
            COUNT(*)                        AS total_txns,
            ROUND(SUM(revenue)::numeric, 2) AS total_revenue,
            MIN(transaction_date)           AS earliest,
            MAX(transaction_date)           AS latest
        FROM sales_transactions
    """)
    m = db.execute(metrics_query).fetchone()

    cat_query = sa.text("""
        SELECT p.category, ROUND(SUM(t.revenue)::numeric, 2) AS rev
        FROM sales_transactions t
        JOIN product_catalog p ON t.sku = p.sku
        GROUP BY p.category ORDER BY rev DESC
    """)
    cats = db.execute(cat_query).fetchall()

    region_query = sa.text("""
        SELECT region, ROUND(SUM(revenue)::numeric, 2) AS rev
        FROM sales_transactions GROUP BY region ORDER BY rev DESC
    """)
    regions = db.execute(region_query).fetchall()

    monthly_query = sa.text("""
        SELECT
            TO_CHAR(DATE_TRUNC('month', transaction_date), 'YYYY-MM') AS month,
            ROUND(SUM(revenue)::numeric, 2) AS rev
        FROM sales_transactions
        GROUP BY 1 ORDER BY 1 DESC LIMIT 6
    """)
    monthly = db.execute(monthly_query).fetchall()

    ctx = (
        f"Dataset: {m.total_txns} transactions, total revenue ${m.total_revenue:,}, "
        f"period {m.earliest} to {m.latest}.\n"
        f"Revenue by category: {', '.join(f'{r.category} ${r.rev:,}' for r in cats)}.\n"
        f"Revenue by region: {', '.join(f'{r.region} ${r.rev:,}' for r in regions)}.\n"
        f"Last 6 months revenue: {', '.join(f'{r.month} ${r.rev:,}' for r in monthly)}."
    )
    return ctx


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/summary")
def get_summary(db: Session = Depends(get_db)):
    """Generate a natural-language paragraph summarising current sales trends."""
    context = _get_summary_context(db)

    system_prompt = (
        "You are a senior data analyst for a CPG company. "
        "Write concise, executive-ready insights. No bullet points — flowing prose only. "
        "Highlight what's working, what's declining, and one actionable recommendation."
    )
    user_message = (
        f"Here is a summary of our sales data:\n\n{context}\n\n"
        "Write a 2–3 paragraph executive summary of our sales performance."
    )

    narrative = _call_llm(system_prompt, user_message)
    return {"summary": narrative, "data_context": context}


class QueryRequest(BaseModel):
    question: str


@router.post("/query")
def query_insights(req: QueryRequest, db: Session = Depends(get_db)):
    """Answer a natural-language question about sales data."""
    context = _get_summary_context(db)

    system_prompt = (
        "You are a data analyst assistant for a CPG company. "
        "Answer the user's question based only on the data provided. "
        "Be concise and factual. If the data doesn't support an answer, say so."
    )
    user_message = (
        f"Sales data context:\n{context}\n\n"
        f"Question: {req.question}"
    )

    answer = _call_llm(system_prompt, user_message)
    return {"question": req.question, "answer": answer}
