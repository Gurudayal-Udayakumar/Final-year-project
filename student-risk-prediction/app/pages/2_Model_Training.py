"""Model training and comparison page."""

from pathlib import Path
import streamlit as st
import plotly.express as px
import pandas as pd

from model.train_model import MODEL_PATH, save_model, train_and_select_best_model

st.title("🧠 Model Training")
st.write("Compare multiple machine learning models and save the best one.")

if st.button("Train Model"):
    progress = st.progress(0)
    status = st.empty()

    status.write("Loading data and training models...")
    progress.progress(30)

    results = train_and_select_best_model()

    progress.progress(70)
    status.write("Saving best model...")
    save_model(results["best_model"])

    progress.progress(100)
    status.success("Training complete!")

    st.subheader("Model Comparison")
    comparison = results["comparison_df"].copy()
    formatted = comparison.copy()
    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 score",
        "ROC-AUC",
        "Optimization ROC-AUC",
    ]
    for col in metric_columns:
        if col in formatted.columns:
            formatted[col] = formatted[col].map(lambda x: f"{x:.3f}")
    st.table(formatted)

    fig_acc = px.bar(
        comparison,
        x="Model",
        y="Accuracy",
        title="Model Accuracy Comparison",
        text="Accuracy",
    )
    st.plotly_chart(fig_acc, use_container_width=True)

    st.subheader(f"Best Model: {results['best_model_name']}")

    st.subheader("Confusion Matrix")
    classes = results["classes"]
    cm = pd.DataFrame(
        results["confusion_matrix"],
        index=[f"True {label}" for label in classes],
        columns=[f"Pred {label}" for label in classes],
    )
    st.dataframe(cm)

    st.subheader("Classification Report")
    st.code(results["classification_report"])

st.caption(f"Saved model path: {MODEL_PATH}")
