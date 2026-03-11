from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier

INPUT_PATH = "data/processed/hydraulic_clean.csv"
MODEL_PATH = "models/model.pkl"

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

TARGETS = ["cooler_condition", "valve_condition", "pump_leakage", "accumulator_pressure"]


def train() -> None:
    df = pd.read_csv(INPUT_PATH)

    X = df[FEATURES].values
    y = df[TARGETS].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y[:, 0]
    )

    model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    Path("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"Model saved to: {MODEL_PATH}\n")
    for i, target in enumerate(TARGETS):
        print(f"── {target} ──")
        print(classification_report(y_test[:, i], y_pred[:, i], digits=3))
        print(f"Confusion matrix:\n{confusion_matrix(y_test[:, i], y_pred[:, i])}\n")


if __name__ == "__main__":
    train()
