"""Explainable AI page using SHAP values."""

from pathlib import Path
import joblib
import shap
import pandas as pd
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

# Align columns if the saved model stores feature names
if hasattr(model, "feature_names_in_"):
    expected = list(model.feature_names_in_)
    X = X[expected]

sample_size = min(50, len(X))
X_sample = X.sample(sample_size, random_state=42)

def _choose_explainer(m, data):
    """Return an appropriate SHAP explainer for the model and data."""
    # Prefer TreeExplainer for tree-based models
    try:
        if hasattr(m, "feature_importances_"):
            return shap.TreeExplainer(m)
        # Linear models (logistic/regression) - use LinearExplainer
        if hasattr(m, "coef_"):
            return shap.LinearExplainer(m, data, feature_perturbation="interventional")
        # Fallback to KernelExplainer (model-agnostic, slower)
        return shap.KernelExplainer(lambda x: m.predict_proba(pd.DataFrame(x, columns=data.columns)), data.sample(min(20, len(data)), random_state=1))
    except Exception:
        # As a last resort, try TreeExplainer and let it raise a descriptive error
        return shap.TreeExplainer(m)

try:
    explainer = _choose_explainer(model, X_sample)
    shap_values = explainer.shap_values(X_sample)

    st.subheader("SHAP Summary Plot")
    fig_summary, ax_summary = plt.subplots(figsize=(10, 5))
    # For multiclass models SHAP may return a list (one array per class)
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
    st.info("Tip: Train a tree-based best model (Random Forest or Gradient Boosting) for the fastest SHAP TreeExplainer experience.")
