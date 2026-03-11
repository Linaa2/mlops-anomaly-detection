import concurrent.futures
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

logger = logging.getLogger(__name__)

FEATURES = [
    "PS1", "PS2", "PS3", "PS4", "PS5", "PS6",
    "EPS1", "FS1", "FS2",
    "TS1", "TS2", "TS3", "TS4",
    "VS1", "CE", "CP", "SE",
]

TARGETS = [
    "cooler_condition",
    "valve_condition",
    "pump_leakage",
    "accumulator_pressure",
]

MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "HydraulicConditionModel")
MODEL_FALLBACK_PATH = Path(os.getenv("MODEL_PATH", "models/model.pkl"))
MLFLOW_LOAD_TIMEOUT = int(os.getenv("MLFLOW_LOAD_TIMEOUT", "10"))

model = None

# ── Custom Prometheus metrics ──────────────────────────────────────────────────
predictions_total = Counter(
    "model_predictions_total",
    "Total number of predictions served",
)

prediction_class = Counter(
    "model_prediction_class_total",
    "Prediction count per target and class",
    ["target", "predicted_class"],
)

prediction_latency = Histogram(
    "model_prediction_latency_seconds",
    "Time spent running model.predict()",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
# ──────────────────────────────────────────────────────────────────────────────


def _load_from_mlflow() -> object | None:
    """Try to load Production model from MLflow registry (timeout: MLFLOW_LOAD_TIMEOUT s)."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        return None

    def _fetch():
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_fetch)
    try:
        loaded = future.result(timeout=MLFLOW_LOAD_TIMEOUT)
        executor.shutdown(wait=False)
        logger.info("Model loaded from MLflow registry: models:/%s/Production", MODEL_NAME)
        return loaded
    except concurrent.futures.TimeoutError:
        executor.shutdown(wait=False)
        logger.warning(
            "MLflow registry timed out after %ss. Falling back to pkl.", MLFLOW_LOAD_TIMEOUT
        )
        return None
    except Exception as exc:
        executor.shutdown(wait=False)
        logger.warning("MLflow registry unavailable (%s). Falling back to pkl.", exc)
        return None


def _load_from_pkl() -> object:
    """Load model from local pkl file."""
    if not MODEL_FALLBACK_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_FALLBACK_PATH}")
    logger.info("Loading model from local path: %s", MODEL_FALLBACK_PATH)
    return joblib.load(MODEL_FALLBACK_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        mlflow_model = _load_from_mlflow()
        model = mlflow_model if mlflow_model is not None else _load_from_pkl()
        logger.info("Model ready: %s", type(model).__name__)
    except Exception as exc:
        logger.error("Failed to load model: %s", exc, exc_info=True)
        model = None
    yield


app = FastAPI(title="Hydraulic Condition Prediction API", lifespan=lifespan)
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


@app.get("/")
def home():
    return {"message": "Hydraulic multi-output condition prediction API", "targets": TARGETS}


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictionInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    X = pd.DataFrame([[getattr(payload, col) for col in FEATURES]], columns=FEATURES)

    import time
    t0 = time.perf_counter()
    pred = model.predict(X)[0]
    prediction_latency.observe(time.perf_counter() - t0)

    result = {target: int(pred[i]) for i, target in enumerate(TARGETS)}

    predictions_total.inc()
    for target, cls in result.items():
        prediction_class.labels(target=target, predicted_class=str(cls)).inc()

    return {"predictions": result}
