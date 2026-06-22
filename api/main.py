"""
FastAPI application entry point.

Start with:
    uvicorn api.main:app --reload
or via Docker Compose.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import data, insights, predictions

app = FastAPI(
    title="CPG Sales Intelligence API",
    description=(
        "Serves revenue predictions, AI-generated insights, "
        "and queryable sales metrics for a mid-size CPG company."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predictions.router)
app.include_router(insights.router)
app.include_router(data.router)


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok"}


@app.get("/", tags=["meta"])
def root():
    return {
        "message": "CPG Sales Intelligence API",
        "docs": "/docs",
        "health": "/health",
    }
