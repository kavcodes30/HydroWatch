"""
FastAPI service for the water-borne disease risk early-warning system.

Serves the trained model, the early-warning alerts and the (clearly labelled)
synthetic symptom stream to the dashboard in app/static/index.html.

Run:
    cd app
    uvicorn main:app --reload --port 8000
    # then open http://127.0.0.1:8000

Every response that carries a risk judgement also carries the disclaimer. The
API never returns an outbreak claim.
"""

from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))

from predict import RiskPredictor  # noqa: E402

MODELS_DIR = BASE / "models"
REPORTS_DIR = BASE / "reports"
STATIC_DIR = Path(__file__).resolve().parent / "static"

DISCLAIMER = (
    "Decision-support indicator of potential water-quality stress consistent with "
    "contamination risk. Not a diagnosis, not a confirmed outbreak, and not "
    "laboratory confirmation of contamination. Verify with repeat sampling and "
    "public-health review."
)

SPATIAL_CAVEAT = (
    "Validated only for states present in training. Leave-one-state-out testing gave "
    "macro F1 0.31 and High-Risk recall 0.06 on unseen states, so do not apply this "
    "model to a region outside its training coverage without retraining."
)

app = FastAPI(
    title="Water-Borne Disease Risk Early-Warning API",
    description="Community-level water-quality risk classification and alerting.",
    version="1.1.0",
)


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
class PredictionRequest(BaseModel):
    station: Optional[str] = None
    district: Optional[str] = None
    state: str = Field(..., description="State name, must match training coverage")
    latitude: float
    longitude: float
    month: int = Field(..., ge=1, le=12)
    year_month: Optional[str] = None
    ph: Optional[float] = Field(None, ge=0, le=14)
    electrical_conductivity: Optional[float] = Field(None, ge=0, le=60000)
    rainfall_monthly_mm: Optional[float] = Field(None, ge=0)
    t2m_c: Optional[float] = None
    lag1_ph: Optional[float] = None
    lag1_ec: Optional[float] = None


class BatchRequest(BaseModel):
    records: list[PredictionRequest]


# --------------------------------------------------------------------------
# Cached resources
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_predictor() -> RiskPredictor:
    return RiskPredictor(str(MODELS_DIR))


@lru_cache(maxsize=1)
def get_alerts() -> pd.DataFrame:
    path = REPORTS_DIR / "alerts_holdout.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def get_symptoms() -> pd.DataFrame:
    path = REPORTS_DIR / "synthetic_symptom_stream.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def get_model_info() -> dict:
    with open(MODELS_DIR / "model_metadata.json") as fh:
        meta = json.load(fh)
    with open(REPORTS_DIR / "evaluation_results.json") as fh:
        ev = json.load(fh)
    spatial = {}
    spatial_path = REPORTS_DIR / "spatial_validation_with_location.json"
    if spatial_path.exists():
        with open(spatial_path) as fh:
            s = json.load(fh)
        spatial = {k: s[k] for k in
                   ("mean_accuracy", "mean_macro_f1", "mean_high_recall")}
    deployment = {}
    dpath = REPORTS_DIR / "deployment_model_summary.json"
    if dpath.exists():
        with open(dpath) as fh:
            deployment = json.load(fh)
    return {
        "selected_model": meta["selected_model"],
        "serving": get_predictor().model_filename,
        "features": len(meta["features"]),
        "holdout": {
            "accuracy": round(ev["test_accuracy"], 4),
            "macro_f1": round(ev["macro_f1"], 4),
            "high_risk_recall": round(ev["high_risk_recall"], 4),
            "rows": ev["holdout_rows"],
        },
        "baseline": {
            "accuracy": round(meta["majority_baseline_accuracy"], 4),
            "macro_f1": round(meta["majority_baseline_macro_f1"], 4),
        },
        "spatial_holdout": spatial,
        "deployment": deployment.get("shipped"),
        "target_definition": meta["target_definition"],
        "disclaimer": DISCLAIMER,
        "spatial_caveat": SPATIAL_CAVEAT,
    }


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def dashboard():
    index = STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(500, "dashboard asset missing")
    return FileResponse(index)


@app.get("/health")
def health():
    return {"status": "ok", "model": get_predictor().model_filename}


@app.get("/model-info")
def model_info():
    return get_model_info()


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        return get_predictor().predict_one(request.model_dump())
    except Exception as exc:  # surface a clean error rather than a stack trace
        raise HTTPException(500, f"prediction failed: {exc}") from exc


@app.post("/predict/batch")
def predict_batch(request: BatchRequest):
    if not request.records:
        raise HTTPException(400, "no records supplied")
    if len(request.records) > 500:
        raise HTTPException(400, "batch limit is 500 records")
    return {"predictions": get_predictor().predict(
        [r.model_dump() for r in request.records])}


@app.get("/alerts")
def alerts(
    state: Optional[str] = None,
    severity: Optional[str] = Query(None, pattern="^(alert|watch)$"),
    limit: int = Query(50, ge=1, le=500),
):
    df = get_alerts()
    if df.empty:
        return {"alerts": [], "note": "no alert file; run src/early_warning.py"}
    if state:
        df = df[df["state"].str.lower() == state.lower()]
    if severity:
        df = df[df["severity"] == severity]
    df = df.head(limit)
    return {
        "count": int(len(df)),
        "alerts": df.drop(columns=["disclaimer"]).to_dict("records"),
        "disclaimer": DISCLAIMER,
    }


@app.get("/symptoms")
def symptoms(state: Optional[str] = None):
    df = get_symptoms()
    if df.empty:
        return {"symptoms": [], "note": "run src/early_warning.py"}
    if state:
        df = df[df["state"].str.lower() == state.lower()]
    monthly = (df.groupby("month")["symptom_reports"].sum()
                 .reindex(range(1, 13), fill_value=0).reset_index())
    return {
        "monthly_totals": monthly.to_dict("records"),
        "data_source": "synthetic",
        "warning": ("SIMULATED data for dashboard demonstration only. No real health "
                    "data was supplied and this stream is never used for training or "
                    "evaluation."),
    }


@app.get("/disclaimer")
def disclaimer():
    return {"disclaimer": DISCLAIMER, "spatial_caveat": SPATIAL_CAVEAT}
