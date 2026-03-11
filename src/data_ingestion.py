import zipfile
from pathlib import Path

import pandas as pd
import requests

DATA_URL = (
    "https://archive.ics.uci.edu/static/public/447/condition+monitoring+of+hydraulic+systems.zip"
)

RAW_DIR = Path("data/raw")
ZIP_PATH = RAW_DIR / "hydraulic.zip"
EXTRACT_DIR = RAW_DIR / "hydraulic"
OUTPUT_CSV = RAW_DIR / "hydraulic_data.csv"

SENSORS = [
    "PS1", "PS2", "PS3", "PS4", "PS5", "PS6",
    "EPS1",
    "FS1", "FS2",
    "TS1", "TS2", "TS3", "TS4",
    "VS1",
    "CE", "CP", "SE",
]

PROFILE_COLS = [
    "cooler_condition", "valve_condition", "pump_leakage", "accumulator_pressure", "stable_flag"
]


def download_dataset() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if ZIP_PATH.exists():
        print(f"Zip already exists: {ZIP_PATH}")
        return

    print("Downloading dataset...")
    response = requests.get(DATA_URL, timeout=30)
    response.raise_for_status()

    with open(ZIP_PATH, "wb") as f:
        f.write(response.content)

    print("Download complete")


def unzip_dataset() -> None:
    if EXTRACT_DIR.exists() and any(EXTRACT_DIR.iterdir()):
        print(f"Dataset already extracted in: {EXTRACT_DIR}")
        return

    print("Unzipping dataset...")
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

    print("Extraction complete")


def find_sensor_file(sensor: str) -> Path:
    matches = list(EXTRACT_DIR.rglob(f"{sensor}.txt"))
    if not matches:
        raise FileNotFoundError(f"Could not find file for sensor {sensor}")
    return matches[0]


def load_profile() -> pd.DataFrame:
    profile_path = next(EXTRACT_DIR.rglob("profile.txt"))
    return pd.read_csv(profile_path, sep=r"\s+", header=None, names=PROFILE_COLS, engine="python")


def merge_sensors() -> None:
    print("Merging sensor files...")

    sensor_series = []

    for sensor in SENSORS:
        file_path = find_sensor_file(sensor)
        df = pd.read_csv(file_path, sep=r"\s+", header=None, engine="python")
        series = df.mean(axis=1)
        series.name = sensor
        sensor_series.append(series)
        print(f"{sensor}: raw shape={df.shape} -> merged shape={series.shape}")

    df_sensors = pd.concat(sensor_series, axis=1)
    df_profile = load_profile()

    df_final = pd.concat([df_sensors, df_profile], axis=1)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUTPUT_CSV, index=False)

    print(f"CSV created: {OUTPUT_CSV}")
    print(df_final.shape)


def main() -> None:
    download_dataset()
    unzip_dataset()
    merge_sensors()


if __name__ == "__main__":
    main()
