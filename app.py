import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Load Saved Artifacts
# -----------------------------
model = joblib.load("traffic_volume_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# -----------------------------
# App Configuration
# -----------------------------
st.set_page_config(
    page_title="Traffic Volume Prediction",
    layout="centered"
)

st.title("🚦 Traffic Volume Prediction App")
st.write("Predict metro interstate traffic volume using machine learning.")

# -----------------------------
# User Input Section
# -----------------------------
st.header("Enter Traffic & Weather Details")

temp = st.number_input("Temperature (Kelvin)", min_value=200.0, max_value=350.0, value=290.0)
rain_1h = st.number_input("Rain in last 1 hour (mm)", min_value=0.0, value=0.0)
snow_1h = st.number_input("Snow in last 1 hour (mm)", min_value=0.0, value=0.0)
clouds_all = st.slider("Cloud Coverage (%)", 0, 100, 50)

hour = st.slider("Hour of Day", 0, 23, 12)
day = st.slider("Day of Month", 1, 31, 15)
month = st.slider("Month", 1, 12, 6)
dayofweek = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)
is_weekend = st.selectbox("Is Weekend?", [0, 1])

holiday = st.selectbox(
    "Holiday",
    ["None", "Christmas Day", "Thanksgiving Day", "New Years Day"]
)

weather_main = st.selectbox(
    "Weather Condition",
    ["Clear", "Clouds", "Rain", "Snow", "Mist", "Fog"]
)

# -----------------------------
# Build Input DataFrame
# -----------------------------
input_dict = {
    "temp": temp,
    "rain_1h": rain_1h,
    "snow_1h": snow_1h,
    "clouds_all": clouds_all,
    "hour": hour,
    "day": day,
    "month": month,
    "dayofweek": dayofweek,
    "is_weekend": is_weekend
}

input_df = pd.DataFrame([input_dict])

# -----------------------------
# Handle One-Hot Encoding
# -----------------------------
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Holiday Encoding
if holiday != "None":
    col_name = f"holiday_{holiday}"
    if col_name in input_df.columns:
        input_df[col_name] = 1

# Weather Encoding
weather_col = f"weather_main_{weather_main}"
if weather_col in input_df.columns:
    input_df[weather_col] = 1

# Ensure correct column order
input_df = input_df[feature_columns]

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Traffic Volume"):
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    st.success(f"🚗 Predicted Traffic Volume: {int(prediction[0])} vehicles/hour")
