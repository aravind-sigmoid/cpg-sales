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

# Shared formatting rule injected into every system prompt so the LLM
# never emits markdown symbols that Streamlit would misrender.
_NO_MARKDOWN = (
    "Important: write plain text only. "
    "Do not use markdown — no asterisks, no underscores, no hashes, no bullet dashes. "
    "Use plain paragraph breaks only."
)


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
            max_tokens=600,
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
    """Build a rich text summary of DB metrics for the LLM, including growth signals."""

    # Overall totals
    m = db.execute(sa.text("""
        SELECT
            COUNT(*)                        AS total_txns,
            ROUND(SUM(revenue)::numeric, 2) AS total_revenue,
            MIN(transaction_date)           AS earliest,
            MAX(transaction_date)           AS latest
        FROM sales_transactions
    """)).fetchone()

    # Revenue by category
    cats = db.execute(sa.text("""
        SELECT p.category, ROUND(SUM(t.revenue)::numeric, 2) AS rev
        FROM sales_transactions t
        JOIN product_catalog p ON t.sku = p.sku
        GROUP BY p.category ORDER BY rev DESC
    """)).fetchall()

    # Revenue by region
    regions = db.execute(sa.text("""
        SELECT region, ROUND(SUM(revenue)::numeric, 2) AS rev
        FROM sales_transactions GROUP BY region ORDER BY rev DESC
    """)).fetchall()

    # Year-over-year revenue by region (last two full years)
    yoy_region = db.execute(sa.text("""
        SELECT
            region,
            EXTRACT(YEAR FROM transaction_date)       AS yr,
            ROUND(SUM(revenue)::numeric, 2)           AS rev
        FROM sales_transactions
        WHERE EXTRACT(YEAR FROM transaction_date) >= EXTRACT(YEAR FROM NOW()) - 2
        GROUP BY region, yr
        ORDER BY region, yr
    """)).fetchall()

    # Year-over-year revenue by category
    yoy_cat = db.execute(sa.text("""
        SELECT
            p.category,
            EXTRACT(YEAR FROM t.transaction_date)     AS yr,
            ROUND(SUM(t.revenue)::numeric, 2)         AS rev
        FROM sales_transactions t
        JOIN product_catalog p ON t.sku = p.sku
        WHERE EXTRACT(YEAR FROM t.transaction_date) >= EXTRACT(YEAR FROM NOW()) - 2
        GROUP BY p.category, yr
        ORDER BY p.category, yr
    """)).fetchall()

    # Last 6 months trend
    monthly = db.execute(sa.text("""
        SELECT
            TO_CHAR(DATE_TRUNC('month', transaction_date), 'YYYY-MM') AS month,
            ROUND(SUM(revenue)::numeric, 2) AS rev
        FROM sales_transactions
        GROUP BY 1 ORDER BY 1 DESC LIMIT 6
    """)).fetchall()

    # Build YoY growth strings
    def _yoy_lines(rows, key_attr):
        """Group rows by key and compute % change between consecutive years."""
        from collections import defaultdict
        by_key = defaultdict(dict)
        for r in rows:
            by_key[getattr(r, key_attr)][int(r.yr)] = float(r.rev)
        lines = []
        for name, yr_map in sorted(by_key.items()):
            years = sorted(yr_map)
            if len(years) >= 2:
                prev, curr = yr_map[years[-2]], yr_map[years[-1]]
                pct = ((curr - prev) / prev * 100) if prev else 0
                direction = "up" if pct >= 0 else "down"
                lines.append(
                    f"{name}: {int(years[-2])} ${prev:,.0f} -> {int(years[-1])} ${curr:,.0f} "
                    f"({direction} {abs(pct):.1f}%)"
                )
            else:
                lines.append(f"{name}: only one year of data (${yr_map[years[0]]:,.0f})")
        return lines

    region_growth = _yoy_lines(yoy_region, "region")
    cat_growth = _yoy_lines(yoy_cat, "category")

    ctx = (
        f"Dataset overview: {m.total_txns} transactions, total revenue ${m.total_revenue:,}, "
        f"period {m.earliest} to {m.latest}.\n\n"
        f"Revenue by category (all time): {', '.join(f'{r.category} ${r.rev:,}' for r in cats)}.\n\n"
        f"Revenue by region (all time): {', '.join(f'{r.region} ${r.rev:,}' for r in regions)}.\n\n"
        f"Year-over-year growth by region:\n" + "\n".join(f"  {l}" for l in region_growth) + "\n\n"
        f"Year-over-year growth by category:\n" + "\n".join(f"  {l}" for l in cat_growth) + "\n\n"
        f"Last 6 months revenue (most recent first): "
        f"{', '.join(f'{r.month} ${r.rev:,}' for r in monthly)}."
    )
    return ctx


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/summary")
def get_summary(db: Session = Depends(get_db)):
    """Generate a natural-language paragraph summarising current sales trends."""
    context = _get_summary_context(db)

    system_prompt = (
        "You are a senior data analyst for a CPG company. "
        "Write concise, executive-ready insights in flowing prose. "
        "No bullet points. Highlight what is growing, what is declining, "
        "and give one actionable recommendation. "
        + _NO_MARKDOWN
    )
    user_message = (
        f"Here is a summary of our sales data:\n\n{context}\n\n"
        "Write a 2 to 3 paragraph executive summary of our sales performance."
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
        "Answer the user's question using only the data provided. "
        "Be specific and quantitative where possible. "
        "If the data does not support a precise answer, say what you can infer and why. "
        + _NO_MARKDOWN
    )
    user_message = (
        f"Sales data context:\n\n{context}\n\n"
        f"Question: {req.question}"
    )

    answer = _call_llm(system_prompt, user_message)
    return {"question": req.question, "answer": answer}
