import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_BASE = os.getenv("API_URL", "http://api:8000")
API_URL = f"{API_BASE}/predict"
METRICS_PATH = Path("reports/model_metrics.json")

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

st.set_page_config(page_title="Hydraulic Condition Monitor", layout="wide")

st.title("Hydraulic Condition Prediction")
st.write("Enter sensor values and get predicted component conditions.")

tab1, tab2 = st.tabs(["Prediction", "Model evaluation"])

with tab1:
    inputs = {}

    col1, col2 = st.columns(2)

    for i, feature in enumerate(FEATURES):
        with col1 if i % 2 == 0 else col2:
            inputs[feature] = st.number_input(feature, value=0.0, format="%.4f")

    if st.button("Predict"):
        try:
            response = requests.post(API_URL, json=inputs, timeout=10)

            if response.status_code == 200:
                result = response.json()
                preds = result.get("predictions", {})

                st.success("Prediction successful")

                if preds:
                    pred_df = pd.DataFrame(
                        {
                            "Target": list(preds.keys()),
                            "Predicted class": list(preds.values()),
                        }
                    )
                    st.subheader("Predicted conditions")
                    st.dataframe(pred_df, use_container_width=True)
                else:
                    st.warning("No predictions returned by API.")

            else:
                st.error(f"API error: {response.status_code}")
                st.text(response.text)

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to FastAPI. Make sure the API is running on port 8000.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

with tab2:
    st.subheader("Model evaluation metrics")

    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        for target, values in metrics.items():
            st.markdown(f"### {target}")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{values['accuracy']:.3f}")
            c2.metric("Precision", f"{values['precision_weighted']:.3f}")
            c3.metric("Recall", f"{values['recall_weighted']:.3f}")
            c4.metric("F1", f"{values['f1_weighted']:.3f}")

            st.write("Confusion matrix")
            cm_df = pd.DataFrame(values["confusion_matrix"])
            st.dataframe(cm_df, use_container_width=False)

            with st.expander("Detailed classification report"):
                report_df = pd.DataFrame(values["classification_report"]).transpose()
                st.dataframe(report_df, use_container_width=True)

            st.divider()
    else:
        st.warning(
            "Metrics file not found. Run training first to generate reports/model_metrics.json."
        )
