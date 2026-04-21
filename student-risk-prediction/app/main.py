"""Main entry page for the Student Academic Risk Decision Support Platform."""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="Student Risk Analytics Platform",
    page_icon="🎓",
    layout="wide",
)

st.title("Student Academic Risk Decision Support Platform")
st.markdown(
    """
This app uses the OULAD learning analytics dataset, DuckDB feature views,
LightGBM, and individualized SHAP explanations to support academic-risk review.

Use the left sidebar to explore:
- **Dataset Explorer**
- **Model Training**
- **Risk Prediction**
- **Analytics Dashboard**
- **Individual Explainability**
"""
)

st.info("Tip: place OULAD CSV files in data/oulad, then train the model before using prediction and explainability pages.")
