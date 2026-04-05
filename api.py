from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .ml_system import FabricMLPredictor


MODEL_DIR = Path(__file__).resolve().parents[1] / "artifacts"
app = FastAPI(title="Fabric Property Prediction API", version="1.0.0")
predictor: Optional[FabricMLPredictor] = None


class TrainRequest(BaseModel):
    n_samples: int = Field(default=600, ge=50, le=20000)


class TrainFromCSVRequest(BaseModel):
    csv_path: str


class PredictRequest(BaseModel):
    smiles: str


@app.on_event("startup")
def _startup() -> None:
    global predictor
    if (MODEL_DIR / "models.joblib").exists():
        predictor = FabricMLPredictor.load(MODEL_DIR)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": predictor is not None}


@app.post("/train/demo")
def train_demo(request: TrainRequest):
    global predictor
    predictor = FabricMLPredictor()
    report = predictor.fit_demo(n_samples=request.n_samples)
    predictor.save(MODEL_DIR)
    return report.__dict__


@app.post("/train/csv")
def train_csv(request: TrainFromCSVRequest):
    global predictor
    from pandas import read_csv

    csv_path = Path(request.csv_path)
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail=f"CSV not found: {csv_path}")
    frame = read_csv(csv_path)
    predictor = FabricMLPredictor()
    report = predictor.fit(frame)
    predictor.save(MODEL_DIR)
    return report.__dict__


@app.post("/predict")
def predict(request: PredictRequest):
    if predictor is None:
        raise HTTPException(status_code=400, detail="Model is not loaded. Train first.")
    try:
        return predictor.predict_with_uncertainty(request.smiles)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
