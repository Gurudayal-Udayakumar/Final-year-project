"""Main entry page for the Student Risk Analytics Platform."""

import streamlit as st

st.set_page_config(
    page_title="Student Risk Analytics Platform",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Student Burnout & Dropout Risk Analytics Platform")
st.markdown(
    """
Welcome to the decision-support dashboard for identifying **Low**, **Medium**, and **High** student risk.

Use the left sidebar to explore:
- **Dataset Explorer**
- **Model Training & Comparison**
- **Risk Prediction**
- **Analytics Dashboard**
- **Explainable AI (SHAP)**
"""
)

st.info("Tip: Train the model first from the 'Model Training' page before using prediction and explainability pages.")
