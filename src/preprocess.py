import pandas as pd


def preprocess():
    df = pd.read_csv("data/raw/weather.csv")

    df = df.dropna()

    df.to_csv("data/processed/weather_clean.csv", index=False)


if __name__ == "__main__":
    preprocess()
