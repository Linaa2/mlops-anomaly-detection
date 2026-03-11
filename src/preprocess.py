from pathlib import Path

import pandas as pd

INPUT_PATH = "data/raw/hydraulic_data.csv"
OUTPUT_PATH = "data/processed/hydraulic_clean.csv"

SENSORS = [
    "PS1", "PS2", "PS3", "PS4", "PS5", "PS6",
    "EPS1",
    "FS1", "FS2",
    "TS1", "TS2", "TS3", "TS4",
    "VS1",
    "CE", "CP", "SE",
]

TARGETS = ["cooler_condition", "valve_condition", "pump_leakage", "accumulator_pressure"]


def preprocess() -> None:
    df = pd.read_csv(INPUT_PATH)

    # Remove unstable cycles
    df = df[df["stable_flag"] == 0].copy()

    df = df[SENSORS + TARGETS].dropna()

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Rows: {len(df)} | Features: {len(SENSORS)} | Targets: {TARGETS}")
    for col in TARGETS:
        print(f"  {col}: {df[col].value_counts().to_dict()}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    preprocess()
