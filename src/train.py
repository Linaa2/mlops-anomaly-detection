from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

INPUT_PATH = "data/processed/weather_clean.csv"
MODEL_PATH = "models/model.pkl"
OUTPUT_PATH = "data/processed/weather_with_predictions.csv"


def train():
    df = pd.read_csv(INPUT_PATH)

    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(df)

    preds = model.predict(df)
    scores = model.decision_function(df)

    results = df.copy()
    results["prediction"] = preds
    results["anomaly_score"] = scores

    anomaly_count = (preds == -1).sum()
    total_count = len(preds)
    anomaly_ratio = anomaly_count / total_count

    Path("models").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    results.to_csv(OUTPUT_PATH, index=False)

    print("Model saved")
    print(f"Predictions saved to: {OUTPUT_PATH}")
    print(f"Total observations: {total_count}")
    print(f"Detected anomalies: {anomaly_count}")
    print(f"Anomaly ratio: {anomaly_ratio:.2%}")


if __name__ == "__main__":
    train()
