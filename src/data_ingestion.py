import pandas as pd
import requests

URL = "https://api.open-meteo.com/v1/forecast?latitude=48.85&longitude=2.35&hourly=temperature_2m,relativehumidity_2m"


def fetch_data():
    response = requests.get(URL)

    data = response.json()

    df = pd.DataFrame(
        {
            "temperature": data["hourly"]["temperature_2m"],
            "humidity": data["hourly"]["relativehumidity_2m"],
        }
    )

    df.to_csv("data/raw/weather.csv", index=False)


if __name__ == "__main__":
    fetch_data()
