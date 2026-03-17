"""Explainable AI page using SHAP values."""

from pathlib import Path
import joblib
import shap
import streamlit as st
import matplotlib.pyplot as plt

from model.train_model import MODEL_PATH
from utils.preprocessing import load_dataset, prepare_features_and_target

st.title("🔍 Explainable AI")
st.write("This page shows how features influence model decisions using SHAP.")

if not Path(MODEL_PATH).exists():
    st.warning("Model file not found. Please train model first.")
    st.stop()

model = joblib.load(MODEL_PATH)
df = load_dataset()
X, _ = prepare_features_and_target(df)

sample_size = min(50, len(X))
X_sample = X.sample(sample_size, random_state=42)

try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    st.subheader("SHAP Summary Plot")
    fig_summary, ax_summary = plt.subplots(figsize=(10, 5))
    # For multiclass tree models, SHAP returns a list (one array per class).
    values_to_plot = shap_values[0] if isinstance(shap_values, list) else shap_values
    shap.summary_plot(values_to_plot, X_sample, show=False)
    st.pyplot(fig_summary)

    st.subheader("Feature Importance Explanation")
    st.write(
        "Features with larger SHAP magnitude have greater impact on predicted risk. "
        "Positive SHAP values push predictions toward higher-risk classes, and negative "
        "values push predictions toward lower-risk classes."
    )
except Exception as exc:
    st.error(f"Unable to generate SHAP visualizations for this model type: {exc}")
    st.info("Tip: Train a tree-based best model (Random Forest or Gradient Boosting) for SHAP TreeExplainer.")
