import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

# dataset simple
df = pd.read_csv("data/data.csv")

X = df.drop(columns=["label"], errors="ignore")

model = IsolationForest(contamination=0.01)

model.fit(X)

joblib.dump(model, "models/model.pkl")

print("Model trained and saved.")