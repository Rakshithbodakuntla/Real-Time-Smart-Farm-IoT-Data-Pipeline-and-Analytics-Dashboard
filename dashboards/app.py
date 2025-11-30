# dashboards/app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "sensor_data.csv"
MODEL_PATH = BASE_DIR / "models" / "model.pkl"

# ---------- Page Setup ----------
st.set_page_config(
    page_title="Smart Farming IoT Dashboard",
    layout="wide",
)

st.title("🌾 Smart Farming IoT Dashboard")
st.caption("IoT-inspired synthetic sensor data + ML-based irrigation prediction")

# ---------- Load Data ----------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    # Ensure timestamp is parsed for sorting/plotting
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

df = load_data()

# ---------- Layout ----------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📋 Latest Sensor Readings")
    st.dataframe(df.sort_values("timestamp", ascending=False).head(20), use_container_width=True)

    st.subheader("📈 Soil Moisture Trend")
    moisture_df = df.sort_values("timestamp")
    st.line_chart(moisture_df["soil_moisture"])

with col2:
    st.subheader("📊 Summary Statistics")
    st.write(df.describe())

# ---------- Load Model ----------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ---------- Irrigation Prediction UI ----------
st.subheader("💧 Irrigation Need Prediction")

c1, c2, c3 = st.columns(3)

with c1:
    soil_moisture = st.slider("Soil Moisture (%)", 0, 100, 35)
    humidity = st.slider("Humidity (%)", 0, 100, 60)
    soil_ph = st.slider("Soil pH", 4.0, 9.0, 6.5, step=0.1)

with c2:
    soil_temp = st.slider("Soil Temperature (°C)", 0, 50, 28)
    light = st.slider("Light Intensity", 0, 3000, 1200)

with c3:
    rain = st.slider("Rainfall (mm)", 0, 50, 5)

if st.button("🔍 Predict Irrigation Need"):
    # NOTE: Order of features MUST match training: 
    # soil_moisture, soil_temp, humidity, light, soil_ph, rain
    X = np.array([[soil_moisture, soil_temp, humidity, light, soil_ph, rain]])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]  # probability of class 1

    if pred == 1:
        st.success(f"💧 Irrigation Needed (probability: {proba:.2f})")
    else:
        st.info(f"✔ No Irrigation Required (probability of need: {proba:.2f})")

    # Debug / explanation panel
    with st.expander("See input values used for prediction"):
        st.write({
            "soil_moisture": soil_moisture,
            "soil_temp": soil_temp,
            "humidity": humidity,
            "light": light,
            "soil_ph": soil_ph,
            "rain": rain
        })
