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

# Streamlit does not fully support pandas Styler in all cases; show a plain dataframe
try:
    # Create a colored HTML column for display if possible
    styled = model_df.copy()
    styled["risk_colored"] = styled["risk"].map(lambda v: f"{v}")
    st.dataframe(styled.drop(columns=["risk_colored"]), use_container_width=True)
except Exception:
    st.dataframe(model_df, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    risk_counts = model_df["risk"].value_counts().reset_index()
    risk_counts.columns = ["risk", "count"]
    fig_pie = px.pie(risk_counts, names="risk", values="count", title="Risk Distribution")
    st.plotly_chart(fig_pie, use_container_width=True)

    fig_box = px.box(model_df, x="risk", y="total_clicks", color="risk", title="Total VLE Clicks vs Risk")
    st.plotly_chart(fig_box, use_container_width=True)

    education_risk = model_df.groupby(["highest_education", "risk"]).size().reset_index(name="students")
    fig_education = px.bar(
        education_risk,
        x="highest_education",
        y="students",
        color="risk",
        barmode="group",
        title="Risk by Highest Education",
    )
    st.plotly_chart(fig_education, use_container_width=True)

with col2:
    fig_scatter = px.scatter(
        model_df,
        x="avg_score",
        y="total_clicks",
        color="risk",
        hover_data=["id_student", "code_module", "code_presentation"],
        title="Assessment Score vs VLE Clicks by Risk",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    fig_late = px.box(
        model_df,
        x="risk",
        y="late_submissions",
        color="risk",
        title="Late Submissions by Risk",
    )
    st.plotly_chart(fig_late, use_container_width=True)

    fig_pi = px.box(
        model_df,
        x="risk",
        y="procrastination_index",
        color="risk",
        title="Procrastination Index by Risk",
    )
    st.plotly_chart(fig_pi, use_container_width=True)

    fig_drift = px.box(
        model_df,
        x="risk",
        y="days_since_last_drift",
        color="risk",
        title="Days Since Last Engagement Drift by Risk",
    )
    st.plotly_chart(fig_drift, use_container_width=True)

    if Path(MODEL_PATH).exists():
        model = joblib.load(MODEL_PATH)
        final_model = model.named_steps.get("model") if hasattr(model, "named_steps") else model
        preprocessor = model.named_steps.get("preprocess") if hasattr(model, "named_steps") else None
        if hasattr(final_model, "feature_importances_"):
            feature_names = (
                preprocessor.get_feature_names_out()
                if preprocessor is not None
                else model.feature_names_in_
            )
            importance_df = pd.DataFrame(
                {
                    "Feature": feature_names,
                    "Importance": final_model.feature_importances_,
                }
            ).sort_values(by="Importance", ascending=False)
            importance_df = importance_df.head(20)
            fig_imp = px.bar(importance_df, x="Feature", y="Importance", title="Feature Importance")
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Feature importance chart is available for tree-based models.")
    else:
        st.warning("Train and save a model to view feature importance.")
