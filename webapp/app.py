import requests
import streamlit as st

API_URL = "http://api:8000/predict"

st.set_page_config(page_title="Weather Anomaly Detection", page_icon="🌦️")

st.title("🌦️ Weather Anomaly Detection")
st.write("Enter weather measurements to detect whether the conditions are anomalous.")

temperature = st.number_input("Temperature", value=10.0)
humidity = st.number_input("Humidity", value=70.0)
wind_speed = st.number_input("Wind Speed", value=10.0)
pressure = st.number_input("Pressure", value=1015.0)
precipitation = st.number_input("Precipitation", value=0.0)

if st.button("Predict"):
    params = {
        "temperature": temperature,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "pressure": pressure,
        "precipitation": precipitation,
    }

    try:
        response = requests.post(API_URL, params=params, timeout=10)
        response.raise_for_status()
        result = response.json()

        prediction = result["prediction"]
        anomaly_score = result["anomaly_score"]

        st.subheader("Prediction Result")
        st.write(f"**Prediction:** {prediction}")
        st.write(f"**Anomaly score:** {anomaly_score:.4f}")

        if prediction == -1:
            st.error("Anomalous weather conditions detected.")
        else:
            st.success("Normal weather conditions detected.")

    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to API: {e}")
