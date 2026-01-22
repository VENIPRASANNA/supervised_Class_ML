import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("traffic_classification_model.pkl")

# Page config
st.set_page_config(
    page_title="Traffic Condition Prediction",
    layout="centered"
)

# -----------------------------
# HEADER
# -----------------------------
st.markdown(
    """
    <h1 style='text-align: center;'>🚦 Traffic Condition Prediction</h1>
    <p style='text-align: center; color: gray;'>
    Predict real-time traffic conditions using key traffic indicators
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# -----------------------------
# INPUT SECTION
# -----------------------------
st.subheader("🔧 Traffic Inputs")

vehicle_count = st.slider("🚗 Vehicle Count", 0, 500, 120)
traffic_speed = st.slider("🚘 Average Speed (km/h)", 0, 120, 40)
road_occupancy = st.slider("🛣️ Road Occupancy (%)", 0, 100, 60)

col1, col2 = st.columns(2)
with col1:
    traffic_light = st.selectbox("🚦 Traffic Light State", ["Red", "Yellow", "Green"])
with col2:
    weather = st.selectbox("🌦️ Weather Condition", ["Clear", "Rainy", "Foggy"])

accident = st.selectbox("🚑 Accident Report", ["No", "Yes"])
hour = st.slider("⏰ Time of Day (Hour)", 0, 23, 18)

st.divider()

# -----------------------------
# INTERNAL DEFAULTS (HIDDEN)
# -----------------------------
latitude = 12.97
longitude = 77.59
sentiment = 0.0
ride_demand = 50
parking = 30
emission = 120.0
energy = 8.0
day = 15
month = 6
weekday = 2

# -----------------------------
# ENCODING
# -----------------------------
traffic_light_map = {"Red": 0, "Yellow": 1, "Green": 2}
weather_map = {"Clear": 0, "Rainy": 1, "Foggy": 2}
accident_map = {"No": 0, "Yes": 1}

# -----------------------------
# INPUT DATAFRAME (MODEL ORDER)
# -----------------------------
input_df = pd.DataFrame([[
    latitude,
    longitude,
    vehicle_count,
    traffic_speed,
    road_occupancy,
    traffic_light_map[traffic_light],
    weather_map[weather],
    accident_map[accident],
    sentiment,
    ride_demand,
    parking,
    emission,
    energy,
    hour,
    day,
    month,
    weekday
]])

# -----------------------------
# LABEL MAPPING
# -----------------------------
traffic_condition_map = {
    0: "Low Traffic",
    1: "Moderate Traffic",
    2: "High Traffic",
    3: "Severe Congestion"
}

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🔍 Predict Traffic Condition", use_container_width=True):
    pred_class = model.predict(input_df)[0]
    prediction = traffic_condition_map.get(pred_class, "Unknown")

    st.subheader("📊 Prediction Result")

    if prediction == "Low Traffic":
        st.success("🟢 Low Traffic — Smooth flow of vehicles")
    elif prediction == "Moderate Traffic":
        st.info("🟡 Moderate Traffic — Slight delays expected")
    elif prediction == "High Traffic":
        st.warning("🟠 High Traffic — Expect congestion")
    else:
        st.error("🔴 Severe Congestion — Avoid this route if possible")

    st.caption("Prediction generated using a supervised machine learning classification model.")
