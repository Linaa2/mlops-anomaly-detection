from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("models/model.pkl")

@app.get("/")
def home():
    return {"message": "MLOps anomaly detection API"}

@app.post("/predict")
def predict(data: list):
    data = np.array(data).reshape(1,-1)
    pred = model.predict(data)

    return {"prediction": int(pred[0])}