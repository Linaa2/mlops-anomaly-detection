from pathlib import Path

import pandas as pd

INPUT_PATH = "data/raw/weather.csv"
OUTPUT_PATH = "data/processed/weather_clean.csv"

FEATURES = [
    "temperature",
    "humidity",
    "wind_speed",
    "pressure",
    "precipitation",
]


def preprocess() -> None:
    df = pd.read_csv(INPUT_PATH)

    df = df[FEATURES].dropna().copy()

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Preprocessed data saved to: {OUTPUT_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    preprocess()
