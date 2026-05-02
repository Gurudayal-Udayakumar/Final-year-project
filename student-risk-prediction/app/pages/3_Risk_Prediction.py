"""Interactive risk prediction page."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from model.train_model import MODEL_PATH
from utils.preprocessing import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, load_dataset


@st.cache_resource
def load_model():
    """Load the trained pipeline once per Streamlit session."""
    return joblib.load(MODEL_PATH)


st.title("Risk Prediction")

if not Path(MODEL_PATH).exists():
    st.warning("Model file not found. Please train the model first on the Model Training page.")
    st.stop()

model = load_model()
dataset = load_dataset()

st.caption("Provide a synthetic student profile to estimate High vs Not High academic risk.")

col1, col2, col3 = st.columns(3)

with col1:
    num_of_prev_attempts = st.number_input("Previous Attempts", min_value=0, max_value=6, value=0)
    studied_credits = st.number_input("Studied Credits", min_value=30, max_value=655, value=60, step=15)
    total_clicks = st.number_input("Total Clicks", min_value=0.0, max_value=25000.0, value=1000.0, step=50.0)
    active_days = st.number_input("Active Days", min_value=0, max_value=286, value=40)
    avg_clicks_per_day = st.number_input("Average Clicks per Day", min_value=0.0, max_value=25.0, value=25.0, step=0.1)
    assessments_submitted = st.number_input("Assessments Submitted", min_value=0, max_value=14, value=4)

with col2:
    avg_score = st.slider("Average Score", 0.0, 100.0, 70.0, step=0.5)
    late_submissions = st.number_input("Late Submissions", min_value=0.0, max_value=12.0, value=0.0, step=1.0)
    procrastination_index = st.slider("Procrastination Index", 0.0, 1.0, 0.25, step=0.01)
    days_since_last_drift = st.number_input("Days Since Last Drift", min_value=0, max_value=100, value=80)
    registration_delay_days = st.number_input("Registration Delay Days", min_value=-60, max_value=30, value=-5)
    completion_ratio = st.slider("Completion Ratio", 0.0, 1.0, 0.6, step=0.01)

with col3:
    weighted_avg_score = st.slider("Weighted Average Score", 0.0, 100.0, 68.0, step=0.5)
    recent_7_day_clicks = st.number_input("Recent 7 Day Clicks", min_value=0, max_value=520, value=80)
    days_since_last_activity = st.number_input("Days Since Last Activity", min_value=0, max_value=140, value=5)
    engagement_trend = st.slider("Engagement Trend", -4.0, 3.0, 0.2, step=0.1)
    withdrawal_applicable = st.checkbox("Withdrawal Timing Known", value=False)
    days_registered_before_withdrawal = (
        st.number_input("Days Registered Before Withdrawal", min_value=0, max_value=120, value=30)
        if withdrawal_applicable
        else None
    )

cat_values = {}
for feature in CATEGORICAL_FEATURES:
    options = sorted(dataset[feature].dropna().astype(str).replace("", "Unknown").unique().tolist())
    if "Unknown" not in options:
        options = ["Unknown", *options]
    default_index = 0
    if feature == "gender" and "M" in options:
        default_index = options.index("M")
    elif feature == "disability" and "N" in options:
        default_index = options.index("N")
    elif feature == "code_module" and "BBB" in options:
        default_index = options.index("BBB")
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
        "registration_delay_days": registration_delay_days,
        "days_registered_before_withdrawal": days_registered_before_withdrawal,
        "completion_ratio": completion_ratio,
        "weighted_avg_score": weighted_avg_score,
        "recent_7_day_clicks": recent_7_day_clicks,
        "days_since_last_activity": days_since_last_activity,
        "engagement_trend": engagement_trend,
        **cat_values,
    }
    input_df = pd.DataFrame([input_values])[FEATURE_COLUMNS]

    for feature in NUMERIC_FEATURES:
        input_df[feature] = pd.to_numeric(input_df[feature], errors="coerce")
    for feature in CATEGORICAL_FEATURES:
        input_df[feature] = input_df[feature].fillna("Unknown").replace("", "Unknown").astype(str)

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    classes = model.classes_

    st.subheader(f"Predicted Risk Level: {prediction}")

    proba_df = pd.DataFrame({"Risk Level": classes, "Probability": probabilities})
    proba_df["Probability"] = proba_df["Probability"].map(lambda x: f"{x:.2%}")
    st.write("Prediction Probabilities")
    st.table(proba_df)
