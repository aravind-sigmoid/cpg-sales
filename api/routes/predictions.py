"""
/predict endpoint — monthly revenue forecasting via the trained ML model.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ml.predict import model_is_ready, predict_revenue

router = APIRouter(prefix="/predict", tags=["predictions"])

VALID_CATEGORIES = ["Beverages", "Snacks", "Dairy", "Personal Care", "Household"]
VALID_REGIONS = ["Northeast", "Southeast", "Midwest", "West", "Southwest"]
MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


class PredictRequest(BaseModel):
    category: str = Field(..., examples=["Beverages"])
    region: str = Field(..., examples=["Northeast"])
    month: int = Field(..., ge=1, le=12, examples=[7])
    year: int = Field(2025, ge=2020, le=2030, examples=[2025])


class PredictResponse(BaseModel):
    category: str
    region: str
    month: int
    month_name: str
    year: int
    predicted_revenue: float
    model_mae: float
    model_r2: float
    confidence_note: str


@router.post("/", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Predict total monthly revenue for a given category, region, and month.

    - **category**: one of Beverages, Snacks, Dairy, Personal Care, Household
    - **region**: one of Northeast, Southeast, Midwest, West, Southwest
    - **month**: 1–12
    - **year**: forecast year (used as a feature; default 2025)
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
        year=req.year,
    )

    confidence_note = (
        f"Model R²={result['model_r2']:.3f}. "
        f"Typical prediction error ±${result['model_mae']:,.0f}."
    )

    return PredictResponse(
        category=req.category,
        region=req.region,
        month=req.month,
        month_name=MONTH_NAMES[req.month - 1],
        year=req.year,
        predicted_revenue=result["predicted_revenue"],
        model_mae=result["model_mae"],
        model_r2=result["model_r2"],
        confidence_note=confidence_note,
    )


@router.get("/options")
def prediction_options():
    """Returns valid values for category, region, and month names."""
    return {
        "categories": VALID_CATEGORIES,
        "regions": VALID_REGIONS,
        "months": [{"value": i + 1, "label": name} for i, name in enumerate(MONTH_NAMES)],
    }
