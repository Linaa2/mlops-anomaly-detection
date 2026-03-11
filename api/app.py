import joblib
import pandas as pd
from fastapi import FastAPI

FEATURES = [
    "PS1",
    "PS2",
    "PS3",
    "TS1",
    "TS2",
    "TS3",
    "TS4",
    "VS1",
    "CE",
    "CP",
]

app = FastAPI()

model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")


@app.get("/")
def home():
    return {"message": "Hydraulic anomaly detection API"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(
    PS1: float,
    PS2: float,
    PS3: float,
    TS1: float,
    TS2: float,
    TS3: float,
    TS4: float,
    VS1: float,
    CE: float,
    CP: float,
):
    X = pd.DataFrame(
        [[PS1, PS2, PS3, TS1, TS2, TS3, TS4, VS1, CE, CP]],
        columns=FEATURES,
    )

    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)
    score = model.decision_function(X_scaled)

    return {
        "prediction": int(pred[0]),
        "anomaly_score": float(score[0]),
    }
