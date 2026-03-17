"""Interactive risk prediction page."""

from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

from model.train_model import MODEL_PATH
from utils.feature_engineering import add_engineered_features

st.title("🎯 Risk Prediction")

if not Path(MODEL_PATH).exists():
    st.warning("Model file not found. Please train model first on the Model Training page.")
    st.stop()

model = joblib.load(MODEL_PATH)

col1, col2 = st.columns(2)
with col1:
    attendance = st.slider("Attendance (%)", 0, 100, 75)
    assignments_submitted = st.slider("Assignments Submitted (0-10)", 0, 10, 6)
with col2:
    login_frequency = st.slider("Login Frequency (times/month)", 0, 30, 12)
    avg_grade = st.slider("Average Grade (%)", 0, 100, 70)

if st.button("Predict Risk"):
    input_df = pd.DataFrame(
        [
            {
                "attendance": attendance,
                "assignments_submitted": assignments_submitted,
                "login_frequency": login_frequency,
                "avg_grade": avg_grade,
            }
        ]
    )
    input_df = add_engineered_features(input_df)

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    classes = model.classes_

    st.subheader(f"Predicted Risk Level: {prediction}")

    proba_df = pd.DataFrame({"Risk Level": classes, "Probability": probabilities})
    proba_df["Probability"] = proba_df["Probability"].map(lambda x: f"{x:.2%}")
    st.write("Prediction Probabilities")
    st.table(proba_df)
