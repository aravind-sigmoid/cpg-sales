"""
/predict endpoint — revenue forecasting via the trained ML model.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ml.predict import model_is_ready, predict_revenue

router = APIRouter(prefix="/predict", tags=["predictions"])

VALID_CATEGORIES = ["Beverages", "Snacks", "Dairy", "Personal Care", "Household"]
VALID_REGIONS = ["Northeast", "Southeast", "Midwest", "West", "Southwest"]


class PredictRequest(BaseModel):
    category: str = Field(..., examples=["Beverages"])
    region: str = Field(..., examples=["Northeast"])
    month: int = Field(..., ge=1, le=12, examples=[7])
    day_of_week: int = Field(0, ge=0, le=6, description="0=Monday, 6=Sunday")
    week_of_year: int = Field(1, ge=1, le=53)


class PredictResponse(BaseModel):
    category: str
    region: str
    month: int
    predicted_revenue: float
    model_mae: float
    model_r2: float
    confidence_note: str


@router.post("/", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Predict revenue for a given category, region, and time period.

    - **category**: one of Beverages, Snacks, Dairy, Personal Care, Household
    - **region**: one of Northeast, Southeast, Midwest, West, Southwest
    - **month**: 1–12
    """
    if not model_is_ready():
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet. Run `python ml/train.py` first.",
        )

    result = predict_revenue(
        category=req.category,
        region=req.region,
        month=req.month,
        day_of_week=req.day_of_week,
        week_of_year=req.week_of_year,
    )

    confidence_note = (
        f"Model R²={result['model_r2']:.3f}. "
        f"Typical prediction error ±${result['model_mae']:.2f}."
    )

    return PredictResponse(
        category=req.category,
        region=req.region,
        month=req.month,
        predicted_revenue=result["predicted_revenue"],
        model_mae=result["model_mae"],
        model_r2=result["model_r2"],
        confidence_note=confidence_note,
    )


@router.get("/options")
def prediction_options():
    """Returns valid values for category and region."""
    return {"categories": VALID_CATEGORIES, "regions": VALID_REGIONS}
