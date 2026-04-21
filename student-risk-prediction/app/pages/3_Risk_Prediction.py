"""Interactive risk prediction page."""

from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

from model.train_model import MODEL_PATH
from utils.preprocessing import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, load_dataset

st.title("🎯 Risk Prediction")

if not Path(MODEL_PATH).exists():
    st.warning("Model file not found. Please train model first on the Model Training page.")
    st.stop()

model = joblib.load(MODEL_PATH)
dataset = load_dataset()

col1, col2, col3 = st.columns(3)
with col1:
    num_of_prev_attempts = st.number_input("Previous Attempts", min_value=0, max_value=10, value=0)
    studied_credits = st.number_input("Studied Credits", min_value=0, max_value=600, value=60, step=15)
    assessments_submitted = st.number_input("Assessments Submitted", min_value=0, max_value=20, value=4)
with col2:
    total_clicks = st.number_input("Total VLE Clicks", min_value=0, max_value=50000, value=1000, step=100)
    active_days = st.number_input("Active VLE Days", min_value=0, max_value=300, value=40)
    avg_clicks_per_day = st.number_input("Average Clicks per Active Day", min_value=0.0, max_value=1000.0, value=25.0, step=1.0)
with col3:
    avg_score = st.slider("Average Assessment Score", 0, 100, 70)
    late_submissions = st.number_input("Late Submissions", min_value=0, max_value=20, value=0)
    procrastination_index = st.slider("Procrastination Index", 0.0, 1.0, 0.25, step=0.05)
    days_since_last_drift = st.number_input("Days Since Last Engagement Drift", min_value=0, max_value=100, value=100)

cat_values = {}
for feature in CATEGORICAL_FEATURES:
    options = sorted(dataset[feature].dropna().astype(str).unique().tolist())
    default_index = 0
    if feature == "gender" and "M" in options:
        default_index = options.index("M")
    elif feature == "disability" and "N" in options:
        default_index = options.index("N")
    cat_values[feature] = st.selectbox(feature.replace("_", " ").title(), options, index=default_index)

if st.button("Predict Risk"):
    input_values = {
        "num_of_prev_attempts": num_of_prev_attempts,
        "studied_credits": studied_credits,
        "total_clicks": total_clicks,
        "active_days": active_days,
        "avg_clicks_per_day": avg_clicks_per_day,
        "assessments_submitted": assessments_submitted,
        "avg_score": avg_score,
        "late_submissions": late_submissions,
        "procrastination_index": procrastination_index,
        "days_since_last_drift": days_since_last_drift,
        **cat_values,
    }
    input_df = pd.DataFrame([input_values])
    input_df = input_df[FEATURE_COLUMNS]

    for feature in NUMERIC_FEATURES:
        input_df[feature] = pd.to_numeric(input_df[feature], errors="coerce")
    for feature in CATEGORICAL_FEATURES:
        input_df[feature] = input_df[feature].astype(str)

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    classes = model.classes_

    st.subheader(f"Predicted Risk Level: {prediction}")

    proba_df = pd.DataFrame({"Risk Level": classes, "Probability": probabilities})
    proba_df["Probability"] = proba_df["Probability"].map(lambda x: f"{x:.2%}")
    st.write("Prediction Probabilities")
    st.table(proba_df)
