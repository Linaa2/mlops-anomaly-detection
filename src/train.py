import os
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier

INPUT_PATH = "data/processed/hydraulic_clean.csv"
MODEL_PATH = "models/model.pkl"

FEATURES = [
    "PS1", "PS2", "PS3", "PS4", "PS5", "PS6",
    "EPS1",
    "FS1", "FS2",
    "TS1", "TS2", "TS3", "TS4",
    "VS1",
    "CE", "CP", "SE",
]

TARGETS = ["cooler_condition", "valve_condition", "pump_leakage", "accumulator_pressure"]

N_ESTIMATORS = 100
TEST_SIZE = 0.2
RANDOM_STATE = 42


def train() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("hydraulic-condition-monitoring")

    df = pd.read_csv(INPUT_PATH)
    X = df[FEATURES].values
    y = df[TARGETS].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y[:, 0]
    )

    model = MultiOutputClassifier(
        RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    )

    with mlflow.start_run():
        mlflow.log_params({
            "n_estimators": N_ESTIMATORS,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "n_features": len(FEATURES),
            "n_targets": len(TARGETS),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        })

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        for i, target in enumerate(TARGETS):
            f1 = f1_score(y_test[:, i], y_pred[:, i], average="macro")
            mlflow.log_metric(f"f1_macro_{target}", f1)

        Path("models").mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        mlflow.log_artifact(MODEL_PATH, artifact_path="model")

    print(f"Tracking URI: {tracking_uri}")
    print(f"Model saved to: {MODEL_PATH}\n")
    for i, target in enumerate(TARGETS):
        print(f"── {target} ──")
        print(classification_report(y_test[:, i], y_pred[:, i], digits=3))
        print(f"Confusion matrix:\n{confusion_matrix(y_test[:, i], y_pred[:, i])}\n")


if __name__ == "__main__":
    train()
