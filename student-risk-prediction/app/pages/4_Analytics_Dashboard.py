"""Analytics dashboard with interactive Plotly visualizations."""

from pathlib import Path
import joblib
import streamlit as st
import pandas as pd
import plotly.express as px

from model.train_model import MODEL_PATH
from utils.preprocessing import load_dataset, prepare_model_dataframe

st.title("📈 Analytics Dashboard")

df = load_dataset()
model_df = prepare_model_dataframe(df)

st.subheader("Student Risk Table")


def color_risk(value):
    colors = {
        "Low": "background-color: #d4edda; color: #155724",
        "Medium": "background-color: #fff3cd; color: #856404",
        "High": "background-color: #f8d7da; color: #721c24",
    }
    return colors.get(value, "")

st.dataframe(model_df.style.applymap(color_risk, subset=["risk"]), use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    risk_counts = model_df["risk"].value_counts().reset_index()
    risk_counts.columns = ["risk", "count"]
    fig_pie = px.pie(risk_counts, names="risk", values="count", title="Risk Distribution")
    st.plotly_chart(fig_pie, use_container_width=True)

    fig_box = px.box(model_df, x="risk", y="attendance", color="risk", title="Attendance vs Risk")
    st.plotly_chart(fig_box, use_container_width=True)

with col2:
    fig_scatter = px.scatter(
        model_df,
        x="avg_grade",
        y="attendance",
        color="risk",
        hover_data=["student_id"],
        title="Grades vs Attendance by Risk",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    if Path(MODEL_PATH).exists():
        model = joblib.load(MODEL_PATH)
        if hasattr(model, "feature_importances_"):
            importance_df = pd.DataFrame(
                {
                    "Feature": model.feature_names_in_,
                    "Importance": model.feature_importances_,
                }
            ).sort_values(by="Importance", ascending=False)
            fig_imp = px.bar(importance_df, x="Feature", y="Importance", title="Feature Importance")
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Feature importance chart is available for tree-based models.")
    else:
        st.warning("Train and save a model to view feature importance.")
