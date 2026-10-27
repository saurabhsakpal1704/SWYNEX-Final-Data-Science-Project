"""FastAPI prediction service."""
from pathlib import Path
import sys
from typing import Any
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.modeling import load_model, _scores

app = FastAPI(title="SWYNEX Purchase Intent API", version="1.0.0", description="Predict purchase intent from online session attributes.")
_model = None

class SessionInput(BaseModel):
    Administrative: float = Field(0, ge=0)
    Administrative_Duration: float = Field(0, ge=0)
    Informational: float = Field(0, ge=0)
    Informational_Duration: float = Field(0, ge=0)
    ProductRelated: float = Field(1, ge=0)
    ProductRelated_Duration: float = Field(30, ge=0)
    BounceRates: float = Field(0.02, ge=0, le=1)
    ExitRates: float = Field(0.03, ge=0, le=1)
    PageValues: float = Field(0, ge=0)
    SpecialDay: float = Field(0, ge=0, le=1)
    Month: str = "Nov"
    OperatingSystems: int = Field(2, ge=1)
    Browser: int = Field(2, ge=1)
    Region: int = Field(1, ge=1)
    TrafficType: int = Field(2, ge=1)
    VisitorType: str = "Returning_Visitor"
    Weekend: bool = False

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": Path("artifacts/model.joblib").exists()}

@app.get("/")
def root():
    return {"message": "SWYNEX Purchase Intent API", "docs": "/docs", "health": "/health"}

@app.post("/predict")
def predict(session: SessionInput) -> dict[str, Any]:
    global _model
    try:
        if _model is None: _model = load_model()
        frame = pd.DataFrame([session.model_dump()])
        probability = float(_scores(_model, frame)[0])
        threshold = 0.5
        try:
            import json
            threshold = float(json.loads(Path("artifacts/metrics.json").read_text())["test_metrics"]["threshold"])
        except Exception: pass
        return {"purchase_probability": round(probability, 4), "predicted_purchase": bool(probability >= threshold), "threshold": threshold,
                "interpretation": "Higher probability indicates a greater estimated likelihood of purchase; this is not a guarantee."}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
