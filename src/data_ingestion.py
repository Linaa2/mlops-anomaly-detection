from pathlib import Path

import pandas as pd
import requests

URL = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude=48.85&longitude=2.35"
    "&hourly=temperature_2m,relativehumidity_2m,windspeed_10m,pressure_msl,precipitation"
)


def fetch_data():
    response = requests.get(URL)
    data = response.json()["hourly"]

    df = pd.DataFrame(
        {
            "temperature": data["temperature_2m"],
            "humidity": data["relativehumidity_2m"],
            "wind_speed": data["windspeed_10m"],
            "pressure": data["pressure_msl"],
            "precipitation": data["precipitation"],
        }
    )

    Path("data/raw").mkdir(parents=True, exist_ok=True)

    df.to_csv("data/raw/weather.csv", index=False)

    print("Weather data saved")


if __name__ == "__main__":
    fetch_data()
