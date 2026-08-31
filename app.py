import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")

# ---------------- TITLE ----------------
st.title("❤️ Heart Disease Prediction")
st.markdown(
    "Enter the patient's health information below to estimate the risk of heart disease."
)

st.divider()

# ---------------- INPUTS ----------------
st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 40)
    sex = st.selectbox("Sex", ["M", "F"])
    resting_bp = st.number_input(
        "Resting Blood Pressure (mm Hg)", 80, 200, 120
    )
    cholesterol = st.number_input(
        "Cholesterol (mg/dL)", 100, 600, 200
    )
    fasting_bs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dL", [0, 1]
    )

with col2:
    chest_pain = st.selectbox(
        "Chest Pain Type", ["ATA", "NAP", "TA", "ASY"]
    )
    resting_ecg = st.selectbox(
        "Resting ECG", ["Normal", "ST", "LVH"]
    )
    max_hr = st.slider("Maximum Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox(
        "Exercise-Induced Angina", ["Y", "N"]
    )
    oldpeak = st.slider(
        "Oldpeak (ST Depression)", 0.0, 6.0, 1.0
    )

st_slope = st.selectbox(
    "ST Slope", ["Up", "Flat", "Down"]
)

st.divider()

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Heart Disease Risk", use_container_width=True):

    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_" + sex: 1,
        "ChestPainType_" + chest_pain: 1,
        "RestingECG_" + resting_ecg: 1,
        "ExerciseAngina_" + exercise_angina: 1,
        "ST_Slope_" + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    # Add missing columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Arrange columns exactly as expected by the model
    input_df = input_df[expected_columns]

    # Scale input
    scaled_input = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(scaled_input)[0]

    # ---------------- RESULT ----------------
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(
            "⚠️ High Risk of Heart Disease"
        )
    else:
        st.success(
            "✅ Low Risk of Heart Disease"
        )

    st.caption(
        "⚠️ This prediction is for educational purposes only "
        "and should not be considered medical advice."
    )