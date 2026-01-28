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
st.caption("End-to-end supervised ML application using Scikit-learn Pipeline")

# =========================
# LOAD TRAINED PIPELINE
# =========================
@st.cache_resource
def load_model():
    return joblib.load("traffic_model.pkl")

model = load_model()

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
# DEFAULT / STATIC VALUES
# =========================
LATITUDE = 13.0827
LONGITUDE = 80.2707
SENTIMENT = 0.1
DAY = 15
MONTH = 6
WEEKDAY = 2

# =========================
# PREDICTION
# =========================
if st.button("🔍 Predict Traffic Condition"):
    input_df = pd.DataFrame([{
        "Traffic_Light_State": traffic_light,
        "Weather_Condition": weather,
        "Accident_Report": accident,

        "Latitude": float(LATITUDE),
        "Longitude": float(LONGITUDE),
        "Vehicle_Count": int(vehicle_count),
        "Traffic_Speed_kmh": float(speed),
        "Road_Occupancy_%": float(road_occ),
        "Sentiment_Score": float(SENTIMENT),

        "Hour": int(hour),
        "Day": int(DAY),
        "Month": int(MONTH),
        "Weekday": int(WEEKDAY)
    }])

    try:
        prediction = model.predict(input_df)[0]
        st.success(f"🚦 Predicted Traffic Condition: **{prediction}**")

        # Confidence (if classifier supports probability)
        final_model = model.named_steps.get("model", None)
        if final_model and hasattr(final_model, "predict_proba"):
            proba = model.predict_proba(input_df)[0]
            confidence = max(proba) * 100
            st.metric("Prediction Confidence", f"{confidence:.2f}%")

    except Exception as e:
        st.error("⚠️ Prediction failed. Check model & input consistency.")
        st.code(str(e))

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("Built with Streamlit & Scikit-learn | Supervised ML Academic Project")
