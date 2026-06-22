# Extension Points for the Inheriting Team

This document describes where and how to extend each layer of the system.

---

## 1. Data Ingestion

**File:** `ingestion/ingest.py`, `ingestion/validate.py`

| Extension | How |
|---|---|
| Add a new source system | Add its format to `DATE_FORMATS` in `validate.py`; create a new ingestion function in `ingest.py` |
| Stream ingestion (Kafka/Kinesis) | Replace the `pd.read_csv` call with a consumer; validation layer is unchanged |
| Add validation rules | Add a check block in `validate_transactions()` — follows the existing reject-and-log pattern |
| Schema evolution | Add/alter columns in `db/models.py` and generate an Alembic migration (`alembic revision --autogenerate`) |

---

## 2. ML Model

**Files:** `ml/features.py`, `ml/train.py`, `ml/predict.py`

| Extension | How |
|---|---|
| Swap to a better model | Replace `RandomForestRegressor` in `train.py`; `predict.py` only calls `model.predict()` so it's model-agnostic |
| Add features | Add columns in `build_features()` in `features.py`; re-train |
| Time-series forecasting | Replace sklearn model with `prophet` or `statsforecast`; update `predict_revenue()` signature |
| Scheduled retraining | Add a cron job or Airflow DAG that calls `python ml/train.py` on a schedule |
| Feature store | Extract `build_features()` into a standalone service; share features between training and serving |

---

## 3. API

**Files:** `api/main.py`, `api/routes/`

| Extension | How |
|---|---|
| Add an endpoint | Create a new file in `api/routes/`, define a router, include it in `api/main.py` |
| Authentication | Add OAuth2/JWT middleware in `api/main.py` using FastAPI's `Security` dependency |
| Rate limiting | Add `slowapi` middleware |
| Versioning | Prefix routers with `/v1/`, `/v2/` |
| Caching | Wrap expensive DB queries with Redis (add `redis` + `fastapi-cache2`) |

---

## 4. LLM / AI Component

**File:** `api/routes/insights.py`

| Extension | How |
|---|---|
| Swap LLM provider | Change `_call_llm()` — the rest of the route is provider-agnostic |
| Use Anthropic Claude | `pip install anthropic`; replace `OpenAI` client with `Anthropic` client in `_call_llm()` |
| Add RAG | Store embeddings of data summaries in pgvector; retrieve relevant chunks before LLM call |
| Structured output | Use OpenAI function calling / structured outputs to return JSON instead of prose |
| Prompt versioning | Move system prompts to a `prompts/` directory; load by name |

---

## 5. UI

**File:** `ui/app.py`

| Extension | How |
|---|---|
| Add a page | Add a new `st.tab` or use `st.sidebar` navigation |
| Replace Streamlit | The UI only calls the FastAPI REST API — swap it for React, Next.js, etc. without touching the backend |
| Multi-user auth | Add `streamlit-authenticator` for basic auth |

---

## 6. Infrastructure

| Extension | How |
|---|---|
| Production deployment | Replace Compose with Kubernetes manifests; use managed Postgres (RDS/Cloud SQL) |
| Observability | Add `opentelemetry-sdk` + export to Grafana/Datadog |
| Database migrations | `alembic init alembic`; use `alembic upgrade head` in the API container startup |
| Secret management | Replace `.env` file with AWS Secrets Manager / Vault; update `config.py` |
