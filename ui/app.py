"""
Streamlit dashboard — CPG Sales Intelligence.

Tabs:
  1. 📊 Metrics    — KPIs and revenue charts
  2. 🔮 Predictions — revenue forecast form
  3. 🤖 AI Insights — LLM summary + natural language Q&A
"""
from __future__ import annotations

import os
import sys

import plotly.graph_objects as go
import requests
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="CPG Sales Intelligence",
    page_icon="📦",
    layout="wide",
)


# ── Helper ────────────────────────────────────────────────────────────────────

def api_get(path: str, params: dict | None = None):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the API. Is it running? (`uvicorn api.main:app`)")
        return None
    except Exception as exc:
        st.error(f"API error: {exc}")
        return None


def api_post(path: str, payload: dict):
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the API. Is it running?")
        return None
    except Exception as exc:
        st.error(f"API error: {exc}")
        return None


# ── Layout ────────────────────────────────────────────────────────────────────

st.title("📦 CPG Sales Intelligence")
st.caption("Powered by FastAPI · RandomForest · OpenAI")

tab1, tab2, tab3 = st.tabs(["📊 Metrics", "🔮 Predictions", "🤖 AI Insights"])


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Metrics
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Business Metrics")

    metrics = api_get("/data/metrics")
    if metrics:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Revenue", f"${metrics['total_revenue']:,.0f}")
        c2.metric("Total Transactions", f"{metrics['total_transactions']:,}")
        c3.metric("Avg Revenue / Txn", f"${metrics['avg_revenue_per_txn']:,.2f}")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Revenue by Category**")
            cats = metrics.get("top_categories", [])
            if cats:
                fig = go.Figure(go.Bar(
                    x=[c["category"] for c in cats],
                    y=[c["revenue"] for c in cats],
                    marker_color="steelblue",
                ))
                fig.update_layout(margin=dict(t=20, b=20), height=300)
                st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.markdown("**Revenue by Region**")
            regions = metrics.get("top_regions", [])
            if regions:
                fig = go.Figure(go.Bar(
                    x=[r["region"] for r in regions],
                    y=[r["revenue"] for r in regions],
                    marker_color="coral",
                ))
                fig.update_layout(margin=dict(t=20, b=20), height=300)
                st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Monthly Revenue Trend")

    col1, col2 = st.columns(2)
    filter_category = col1.selectbox(
        "Filter by category", ["All", "Beverages", "Snacks", "Dairy", "Personal Care", "Household"]
    )
    filter_region = col2.selectbox(
        "Filter by region", ["All", "Northeast", "Southeast", "Midwest", "West", "Southwest"]
    )

    trend_params = {}
    if filter_category != "All":
        trend_params["category"] = filter_category
    if filter_region != "All":
        trend_params["region"] = filter_region

    trend = api_get("/data/monthly-trend", params=trend_params)
    if trend:
        months = [r["month"] for r in trend]
        revenues = [r["revenue"] for r in trend]
        fig = go.Figure(go.Scatter(x=months, y=revenues, mode="lines+markers", line=dict(color="steelblue")))
        fig.update_layout(
            xaxis_title="Month", yaxis_title="Revenue ($)",
            margin=dict(t=20, b=20), height=350,
        )
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Predictions
# ─────────────────────────────────────────────────────────────────────────────
MONTH_OPTIONS = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}

with tab2:
    st.subheader("Monthly Revenue Forecast")
    st.write("Select a category, region, and month to forecast total revenue for that period.")

    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        category = col1.selectbox("Category", ["Beverages", "Snacks", "Dairy", "Personal Care", "Household"])
        region = col2.selectbox("Region", ["Northeast", "Southeast", "Midwest", "West", "Southwest"])

        col3, col4 = st.columns(2)
        month_name = col3.selectbox("Month", list(MONTH_OPTIONS.keys()), index=6)
        year = col4.selectbox("Year", [2024, 2025, 2026], index=1)

        submitted = st.form_submit_button("Forecast Revenue", use_container_width=True)

    if submitted:
        with st.spinner("Running forecast..."):
            result = api_post("/predict/", {
                "category": category,
                "region": region,
                "month": MONTH_OPTIONS[month_name],
                "year": year,
            })
        if result:
            st.success(
                f"**Forecasted Revenue for {month_name} {year} — "
                f"{category} / {region}: ${result['predicted_revenue']:,.0f}**"
            )
            st.caption(result["confidence_note"])


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — AI Insights
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("AI-Powered Insights")

    col_sum, col_qa = st.columns([1, 1])

    with col_sum:
        st.markdown("**📝 Executive Summary**")
        if st.button("Generate Summary", use_container_width=True):
            with st.spinner("Asking the LLM..."):
                result = api_get("/insights/summary")
            if result:
                st.write(result["summary"])

    with col_qa:
        st.markdown("**💬 Ask a Question**")
        question = st.text_input(
            "Ask anything about the data",
            placeholder="e.g. Which region has the highest growth?",
        )
        if st.button("Ask", use_container_width=True) and question:
            with st.spinner("Thinking..."):
                result = api_post("/insights/query", {"question": question})
            if result:
                st.info(f"**Q:** {result['question']}")
                st.write(f"**A:** {result['answer']}")
