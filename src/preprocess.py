from pathlib import Path

import pandas as pd

INPUT_PATH = "data/raw/hydraulic_data.csv"
OUTPUT_PATH = "data/processed/hydraulic_clean.csv"

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
