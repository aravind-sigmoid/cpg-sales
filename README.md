# CPG Sales Intelligence

A working skeleton for a mid-size CPG company's sales performance platform.  
Built as part of the AIA Engineer evaluation — demonstrates end-to-end data pipeline, ML forecasting, LLM insights, and a user-facing dashboard.

---

## Architecture

```
data/          Synthetic CPG data generator (with quality issues)
ingestion/     ETL: validate → clean → load into Postgres
db/            SQLAlchemy models + DB init
ml/            Feature engineering, RandomForest training, inference
api/           FastAPI — /predict, /insights, /data
ui/            Streamlit dashboard (metrics, predictions, AI Q&A)
tests/         pytest suite
docs/          ADR-001, extension points
```

Full architecture rationale: [docs/ADR-001-architecture.md](docs/ADR-001-architecture.md)  
How to extend: [docs/extension-points.md](docs/extension-points.md)

---

## Quick Start (Docker Compose — recommended)

### Prerequisites
- Docker + Docker Compose
- An OpenAI API key

### 1. Configure environment

```bash
cp .env.example .env
# Edit .env — replace OPENAI_API_KEY with your real key
```

### 2. Start everything

```bash
docker-compose up --build
```

This starts:
- **Postgres** on port `5432`
- **API** on port `8000` → http://localhost:8000/docs
- **UI** on port `8501` → http://localhost:8501

### 3. Seed the database (first run only)

In a separate terminal:

```bash
# Generate synthetic data
docker-compose exec api python data/generate_data.py

# Run ingestion pipeline
docker-compose exec api python ingestion/ingest.py

# Train the ML model
docker-compose exec api python ml/train.py
```

### 4. Open the dashboard

→ http://localhost:8501

---

## Local Development (without Docker)

### Prerequisites
- Python 3.11+
- Postgres running locally

### Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env with your Postgres credentials and OpenAI key
```

### Run the pipeline

```bash
python data/generate_data.py      # generate raw CSVs
python ingestion/ingest.py        # validate + load into Postgres
python ml/train.py                # train the forecasting model
```

### Start the API

```bash
uvicorn api.main:app --reload
# → http://localhost:8000/docs
```

### Start the UI

```bash
streamlit run ui/app.py
# → http://localhost:8501
```

---

## Running Tests

```bash
pytest
```

Tests cover:
- Validation logic (nulls, negatives, date formats, deduplication)
- API endpoints (mocked model + DB)
- ML feature engineering and prediction

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/predict/` | Revenue forecast |
| GET | `/predict/options` | Valid category/region values |
| GET | `/insights/summary` | LLM executive summary |
| POST | `/insights/query` | Natural language Q&A |
| GET | `/data/metrics` | KPIs |
| GET | `/data/transactions` | Filtered transaction query |
| GET | `/data/monthly-trend` | Revenue trend by month |

Interactive docs at http://localhost:8000/docs

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `POSTGRES_HOST` | `localhost` | Postgres host |
| `POSTGRES_DB` | `cpg_sales` | Database name |
| `POSTGRES_USER` | `cpg_user` | DB username |
| `POSTGRES_PASSWORD` | `cpg_password` | DB password |
| `OPENAI_API_KEY` | *(required)* | Replace dummy with real key |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model name |
| `API_BASE_URL` | `http://localhost:8000` | UI → API URL |

---

## What's Next (suggested for the inheriting team)

1. Replace Streamlit with a React frontend for multi-user production use
2. Add Alembic migrations for schema evolution
3. Introduce a job scheduler (Airflow/Prefect) for automated daily ingestion + retraining
4. Add pgvector + RAG for richer LLM grounding
5. Add auth (OAuth2/JWT) to the API
6. See [docs/extension-points.md](docs/extension-points.md) for full details
