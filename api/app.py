from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

FEATURES = [
    "PS1",
    "PS2",
    "PS3",
    "PS4",
    "PS5",
    "PS6",
    "EPS1",
    "FS1",
    "FS2",
    "TS1",
    "TS2",
    "TS3",
    "TS4",
    "VS1",
    "CE",
    "CP",
    "SE",
]

TARGETS = [
    "cooler_condition",
    "valve_condition",
    "pump_leakage",
    "accumulator_pressure",
]

MODEL_PATH = Path("models/model.pkl")

app = FastAPI(title="Hydraulic Condition Prediction API")

# Expose /metrics endpoint for Prometheus scraping
Instrumentator().instrument(app).expose(app)


class PredictionInput(BaseModel):
    PS1: float
    PS2: float
    PS3: float
    PS4: float
    PS5: float
    PS6: float
    EPS1: float
    FS1: float
    FS2: float
    TS1: float
    TS2: float
    TS3: float
    TS4: float
    VS1: float
    CE: float
    CP: float
    SE: float


model = joblib.load(MODEL_PATH)


@app.get("/")
def home():
    return {
        "message": "Hydraulic multi-output condition prediction API",
        "targets": TARGETS,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictionInput):
    X = pd.DataFrame([[getattr(payload, col) for col in FEATURES]], columns=FEATURES)

    pred = model.predict(X)[0]

    result = {target: int(pred[i]) for i, target in enumerate(TARGETS)}

    return {
        "predictions": result,
    }
