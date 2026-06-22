# ADR-001: System Architecture for CPG Sales Intelligence

**Date:** 2024-06-18  
**Status:** Accepted  
**Deciders:** Aravind Aitipamula

---

## Context

We need a system that ingests raw CPG transaction data, forecasts revenue, exposes LLM-powered insights, and provides a user-facing dashboard — all as a skeleton a project team can inherit and extend.

---

## Decision

### Layered monorepo (single repo, clearly separated modules)

```
data/          ← synthetic data generation
ingestion/     ← ETL: validate → clean → load
db/            ← SQLAlchemy models + Postgres
ml/            ← feature engineering, train, predict
api/           ← FastAPI (REST)
ui/            ← Streamlit dashboard
tests/         ← pytest suite
```

### Technology choices

| Concern | Choice | Rationale |
|---|---|---|
| Language | Python 3.11 | Team default; best ecosystem for ML + data |
| Database | PostgreSQL | Relational, queryable, production-grade; fits star-schema pattern |
| ORM | SQLAlchemy 2 | Type-safe, well-supported; easy migrations with Alembic |
| API | FastAPI | Async-capable, auto docs, Pydantic validation built-in |
| ML model | RandomForestRegressor (sklearn) | Handles mixed features well, no tuning needed for skeleton; easy to swap |
| LLM | OpenAI gpt-4o-mini | Cheap, fast, accessible API; swap to Claude or local model via single config change |
| UI | Streamlit | Zero-JS data dashboard in pure Python; fast to iterate |
| Container | Docker + Compose | Single `docker-compose up` starts everything |
| CI | GitHub Actions | Zero infra, runs tests on every push |

### Data flow

```
generate_data.py → raw CSVs → ingestion/ingest.py → Postgres
                                                         │
                                               ml/train.py (reads Postgres)
                                                         │
                                               ml/model.joblib
                                                         │
                              ┌──────────────────────────┤
                          FastAPI ────────────── OpenAI API
                              │
                          Streamlit UI
```

---

## Alternatives Considered

| Option | Reason Rejected |
|---|---|
| SQLite | No concurrent writes; poor fit for multi-service Docker compose |
| Flask | Less ergonomic than FastAPI for request validation and auto-docs |
| XGBoost | More performant than RF but no meaningful gain at skeleton scale |
| Dash (Plotly) | More code than Streamlit for same result |
| dbt for transforms | Overkill for a 2-week skeleton; adds infra complexity |

---

## Consequences

**Positive:**
- Any engineer with Python + Docker can run the whole system in one command
- Each layer is independently testable and replaceable
- OpenAI key is the only external dependency

**Negative / Trade-offs:**
- Streamlit is not suitable for production multi-user loads (session state is per-connection)
- RandomForest doesn't extrapolate well outside training range — needs retraining as data grows
- No message queue; ingestion is synchronous batch, not streaming
