import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ---------------------------------
# Load Saved Artifacts
# ---------------------------------
model = joblib.load("traffic_volume_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="Traffic Volume Prediction",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 Traffic Volume Prediction System")
st.markdown("AI-based system to predict metro interstate traffic volume.")

# ---------------------------------
# Sidebar Presets
# ---------------------------------
st.sidebar.header("⚙️ Traffic Presets")

preset = st.sidebar.selectbox(
    "Select Scenario",
    ["Custom", "Morning Rush Hour", "Evening Rush Hour", "Night Time", "Weekend"]
)

def preset_values(preset):
    if preset == "Morning Rush Hour":
        return 7, 1
    if preset == "Evening Rush Hour":
        return 17, 0
    if preset == "Night Time":
        return 2, 0
    if preset == "Weekend":
        return 12, 1
    return 12, 0

hour_default, weekend_default = preset_values(preset)

# ---------------------------------
# User Inputs
# ---------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("🕒 Time Information")
    hour = st.slider("Hour of Day", 0, 23, hour_default)
    day = st.slider("Day of Month", 1, 31, 15)
    month = st.slider("Month", 1, 12, 6)
    dayofweek = st.slider("Day of Week (0=Mon)", 0, 6, 2)
    is_weekend = st.selectbox("Is Weekend?", [0, 1], index=weekend_default)

with col2:
    st.subheader("🌦️ Weather Information")
    temp = st.number_input("Temperature (Kelvin)", 200.0, 350.0, 290.0)
    rain_1h = st.number_input("Rain (last 1 hour)", 0.0, 50.0, 0.0)
    snow_1h = st.number_input("Snow (last 1 hour)", 0.0, 50.0, 0.0)
    clouds_all = st.slider("Cloud Coverage (%)", 0, 100, 50)

holiday = st.selectbox(
    "Holiday",
    ["None", "Christmas Day", "Thanksgiving Day", "New Years Day"]
)

weather_main = st.selectbox(
    "Weather Condition",
    ["Clear", "Clouds", "Rain", "Snow", "Mist", "Fog"]
)

# ---------------------------------
# Build Input Data
# ---------------------------------
input_data = {
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

input_df = pd.DataFrame([input_data])

# Add missing columns
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# One-hot encoding
if holiday != "None":
    holiday_col = f"holiday_{holiday}"
    if holiday_col in input_df.columns:
        input_df[holiday_col] = 1

weather_col = f"weather_main_{weather_main}"
if weather_col in input_df.columns:
    input_df[weather_col] = 1

input_df = input_df[feature_columns]

# ---------------------------------
# Prediction
# ---------------------------------
if st.button("🚗 Predict Traffic Volume"):

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    # Confidence range (approx.)
    confidence_low = prediction * 0.9
    confidence_high = prediction * 1.1

    st.success(f"### Predicted Traffic Volume: **{int(prediction)} vehicles/hour**")
    st.info(f"Confidence Range: {int(confidence_low)} – {int(confidence_high)} vehicles/hour")

    # ---------------------------------
    # Visualization
    # ---------------------------------
    st.subheader("📊 Feature Contribution (Top Inputs)")

    fig, ax = plt.subplots()
    feature_values = input_df.iloc[0][["hour", "is_weekend", "temp", "clouds_all"]]
    feature_values.plot(kind="bar", ax=ax)
    ax.set_ylabel("Value")
    ax.set_title("Key Input Features")
    st.pyplot(fig)

# ---------------------------------
# Footer
# ---------------------------------
st.markdown("---")
st.markdown("**Built with Machine Learning & Streamlit** 🚀")
