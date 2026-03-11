import logging
import os
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

logger = logging.getLogger(__name__)

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

MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "HydraulicConditionModel")
MODEL_FALLBACK_PATH = Path(os.getenv("MODEL_PATH", "models/model.pkl"))

app = FastAPI(title="Hydraulic Condition Prediction API")

# Expose /metrics endpoint for Prometheus scraping
Instrumentator().instrument(app).expose(app)


def load_model():
    """Load model from MLflow Production registry; fall back to local pkl."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        try:
            model_uri = f"models:/{MODEL_NAME}/Production"
            loaded = mlflow.sklearn.load_model(model_uri)
            logger.info("Model loaded from MLflow registry: %s", model_uri)
            return loaded
        except Exception as exc:
            logger.warning(
                "Could not load model from MLflow (%s). Falling back to local pkl.", exc
            )

    if MODEL_FALLBACK_PATH.exists():
        logger.info("Loading model from local path: %s", MODEL_FALLBACK_PATH)
        return joblib.load(MODEL_FALLBACK_PATH)

    raise RuntimeError(
        f"No model available: MLflow registry unavailable and {MODEL_FALLBACK_PATH} not found."
    )


model = load_model()


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
