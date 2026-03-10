import joblib
import pandas as pd
from fastapi import FastAPI

app = FastAPI()

model = joblib.load("models/model.pkl")


@app.get("/")
def home():
    return {"message": "Weather anomaly detection API"}


@app.post("/predict")
def predict(temp: float, humidity: float):
    X = pd.DataFrame([{"temperature": temp, "humidity": humidity}])

    pred = model.predict(X)
    score = model.decision_function(X)

    return {
        "prediction": int(pred[0]),
        "anomaly_score": float(score[0]),
    }
