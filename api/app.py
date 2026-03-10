import joblib
import pandas as pd
from fastapi import FastAPI

FEATURES = [
    "temperature",
    "humidity",
    "wind_speed",
    "pressure",
    "precipitation",
]

app = FastAPI()

model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")


@app.get("/")
def home():
    return {"message": "Weather anomaly detection API"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(
    temperature: float,
    humidity: float,
    wind_speed: float,
    pressure: float,
    precipitation: float,
):
    X = pd.DataFrame(
        [[temperature, humidity, wind_speed, pressure, precipitation]],
        columns=FEATURES,
    )

    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)
    score = model.decision_function(X_scaled)

    return {
        "prediction": int(pred[0]),
        "anomaly_score": float(score[0]),
    }
