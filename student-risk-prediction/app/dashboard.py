"""Streamlit dashboard for predicting student burnout/dropout risk."""

from pathlib import Path
import joblib
import pandas as pd
import streamlit as st


# Define important paths once.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model" / "student_risk_model.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "students.csv"


@st.cache_resource
def load_model():
    """Load the trained model from disk (cached for faster app performance)."""
    return joblib.load(MODEL_PATH)


def main():
    """Render the Streamlit user interface and run predictions."""
    st.title("Student Burnout and Dropout Risk Prediction System")
    st.write("Enter student activity details below to estimate risk level.")

    # Input widgets for student data.
    attendance = st.slider("Attendance (%)", min_value=0, max_value=100, value=75)
    assignments = st.slider("Assignments Submitted (out of 10)", min_value=0, max_value=10, value=6)
    login_frequency = st.slider("Login Frequency (times/month)", min_value=0, max_value=30, value=12)
    avg_grade = st.slider("Average Grade (%)", min_value=0, max_value=100, value=70)

    # Predict button triggers model inference.
    if st.button("Predict Risk"):
        model = load_model()

        # Arrange inputs in the same feature order used during training.
        input_df = pd.DataFrame(
            [
                {
                    "attendance": attendance,
                    "assignments_submitted": assignments,
                    "login_frequency": login_frequency,
                    "avg_grade": avg_grade,
                }
            ]
        )

        prediction = model.predict(input_df)[0]
        st.subheader(f"Predicted Risk Level: {prediction}")

    # Show a simple dataset risk distribution chart for demo explanation.
    st.markdown("---")
    st.subheader("Risk Distribution in Dataset")
    dataset = pd.read_csv(DATA_PATH)
    risk_counts = dataset["risk"].value_counts().sort_index()
    st.bar_chart(risk_counts)


if __name__ == "__main__":
    main()
