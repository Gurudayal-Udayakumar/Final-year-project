"""Interactive risk prediction page."""

from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

from model.train_model import MODEL_PATH
from utils.preprocessing import FEATURE_COLUMNS
from utils.feature_engineering import add_engineered_features

st.title("🎯 Risk Prediction")

if not Path(MODEL_PATH).exists():
    st.warning("Model file not found. Please train model first on the Model Training page.")
    st.stop()

model = joblib.load(MODEL_PATH)

# Guardrail for old model files that were trained with different columns.
if hasattr(model, "feature_names_in_"):
    model_features = list(model.feature_names_in_)
    if model_features != FEATURE_COLUMNS:
        st.error(
            "Saved model was trained with a different feature set. "
            "Please retrain from the Model Training page."
        )
        st.write("Expected features:", FEATURE_COLUMNS)
        st.write("Model features:", model_features)
        st.stop()

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

    # Apply same feature engineering used in training.
    input_df = add_engineered_features(input_df)

    # Keep identical feature order as model training.
    input_features = input_df[FEATURE_COLUMNS]

    prediction = model.predict(input_features)[0]
    probabilities = model.predict_proba(input_features)[0]
    classes = model.classes_

    st.subheader(f"Predicted Risk Level: {prediction}")

    proba_df = pd.DataFrame({"Risk Level": classes, "Probability": probabilities})
    proba_df["Probability"] = proba_df["Probability"].map(lambda x: f"{x:.2%}")
    st.write("Prediction Probabilities")
    st.table(proba_df)
