import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Predictive Maintenance AI", page_icon="🔧", layout="wide")

MODEL = "predictive_maintenance_model.pkl"

st.title("🔧 Predictive Maintenance AI")
st.write("Predict whether a machine is at risk of failure using sensor and operating data.")

try:
    model = joblib.load(MODEL)
except FileNotFoundError:
    st.error("Model not found. Run: python train_model.py")
    st.stop()

st.sidebar.header("Machine Sensor Inputs")

temperature = st.sidebar.slider("Temperature (°C)", 45.0, 110.0, 72.0)
vibration = st.sidebar.slider("Vibration (mm/s)", 0.5, 12.0, 4.5)
pressure = st.sidebar.slider("Pressure (kPa)", 70.0, 130.0, 101.0)
rpm = st.sidebar.slider("Rotational Speed (RPM)", 800, 2200, 1500)
operating_hours = st.sidebar.number_input("Operating Hours", 100, 10000, 5000)
load_percentage = st.sidebar.slider("Load (%)", 10.0, 100.0, 68.0)
maintenance_count = st.sidebar.number_input("Maintenance Count", 0, 20, 3)
machine_age_years = st.sidebar.slider("Machine Age (years)", 1, 15, 7)

input_df = pd.DataFrame([{
    "temperature": temperature,
    "vibration": vibration,
    "pressure": pressure,
    "rpm": rpm,
    "operating_hours": operating_hours,
    "load_percentage": load_percentage,
    "maintenance_count": maintenance_count,
    "machine_age_years": machine_age_years
}])

if st.button("Predict Failure Risk", type="primary"):
    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])

    st.subheader("Prediction")
    if prediction == 1:
        st.error(f"⚠️ High risk of machine failure. Estimated risk: {probability:.1%}")
        st.warning("Recommended action: inspect the machine and schedule preventive maintenance.")
    else:
        st.success(f"✅ Low risk of machine failure. Estimated risk: {probability:.1%}")
        st.info("Continue routine monitoring and scheduled maintenance.")

st.subheader("Current Machine Data")
st.dataframe(input_df, use_container_width=True)

st.caption("Educational demonstration only. Do not use this model for real industrial safety or maintenance decisions.")
