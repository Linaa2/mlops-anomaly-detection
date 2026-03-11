from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

INPUT_PATH = "data/processed/hydraulic_clean.csv"
MODEL_PATH = "models/model.pkl"
SCALER_PATH = "models/scaler.pkl"
OUTPUT_PATH = "data/processed/hydraulic_with_predictions.csv"

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


def train() -> None:
    df = pd.read_csv(INPUT_PATH)

    X = df[FEATURES]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        contamination=0.05,
        random_state=42,
    )
    model.fit(X_scaled)

    preds = model.predict(X_scaled)
    scores = model.decision_function(X_scaled)

    results = df.copy()
    results["prediction"] = preds
    results["anomaly_score"] = scores

    anomaly_count = int((preds == -1).sum())
    total_count = len(preds)
    anomaly_ratio = anomaly_count / total_count

    Path("models").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    results.to_csv(OUTPUT_PATH, index=False)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Scaler saved to: {SCALER_PATH}")
    print(f"Predictions saved to: {OUTPUT_PATH}")
    print(f"Total observations: {total_count}")
    print(f"Detected anomalies: {anomaly_count}")
    print(f"Anomaly ratio: {anomaly_ratio:.2%}")


if __name__ == "__main__":
    train()
