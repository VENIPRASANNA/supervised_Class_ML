import streamlit as st
import pandas as pd
import joblib

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Traffic Condition Prediction",
    page_icon="🚦",
    layout="centered"
)

st.title("🚦 Smart Traffic Condition Prediction System")
st.caption("End-to-end supervised ML application with multi-model comparison")

# =========================
# LOAD MODEL
# =========================
model = joblib.load("traffic_model.pkl")

# =========================
# USER INPUTS
# =========================
st.header("📥 Enter Traffic Details")

col1, col2 = st.columns(2)

with col1:
    traffic_light = st.selectbox(
        "Traffic Light State",
        ["Red", "Yellow", "Green"]
    )

    weather = st.selectbox(
        "Weather Condition",
        ["Clear", "Rain", "Fog", "Snow"]
    )

    accident = st.selectbox(
        "Accident Report",
        ["No", "Yes"]
    )

    hour = st.slider(
        "Time of Day (Hour)",
        min_value=0,
        max_value=23,
        value=12
    )

with col2:
    vehicle_count = st.number_input(
        "Vehicle Count",
        min_value=0,
        max_value=1000,
        value=120
    )

    speed = st.number_input(
        "Traffic Speed (km/h)",
        min_value=0.0,
        max_value=150.0,
        value=40.0
    )

    road_occ = st.slider(
        "Road Occupancy (%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0
    )

# =========================
# DEFAULT VALUES (HIDDEN)
# =========================
defaults = {
    "Latitude": 13.0827,
    "Longitude": 80.2707,
    "Sentiment_Score": 0.1,
    "Day": 15,
    "Month": 6,
    "Weekday": 2
}

# =========================
# PREDICTION
# =========================
if st.button("🔍 Predict Traffic Condition"):
    input_df = pd.DataFrame([{
        'Traffic_Light_State': traffic_light,
        'Weather_Condition': weather,
        'Accident_Report': accident,

        'Latitude': 13.0827,
        'Longitude': 80.2707,
        'Vehicle_Count': vehicle_count,
        'Traffic_Speed_kmh': speed,
        'Road_Occupancy_%': road_occ,
        'Sentiment_Score': 0.1,

        'Hour': hour,
        'Day': 15,
        'Month': 6,
        'Weekday': 2
    }])

    # 🔒 CRITICAL LINE — DO NOT REMOVE
    input_df = input_df[model.feature_names_in_]

    prediction = model.predict(input_df)[0]

    st.success(f"🚦 Predicted Traffic Condition: **{prediction}**")

    # Confidence score (if supported)
    if hasattr(model.named_steps['model'], "predict_proba"):
        proba = model.predict_proba(input_df)[0]
        confidence = max(proba) * 100
        st.metric("Prediction Confidence", f"{confidence:.2f}%")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("Built with Scikit-learn Pipelines & Streamlit | Supervised ML Project")
