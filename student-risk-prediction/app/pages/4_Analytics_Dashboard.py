"""Analytics dashboard with interactive Plotly visualizations."""

from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from model.train_model import MODEL_PATH
from utils.preprocessing import load_dataset, prepare_model_dataframe


@st.cache_resource
def load_model():
    """Load the trained pipeline once per Streamlit session."""
    return joblib.load(MODEL_PATH)


st.title("Analytics Dashboard")

df = load_dataset()
model_df = prepare_model_dataframe(df)

st.subheader("Student Risk Table")
st.dataframe(model_df, use_container_width=True)

col1, col2 = st.columns(2)
hover_fields = [field for field in ["student_id", "code_module", "code_presentation", "final_result"] if field in model_df.columns]

with col1:
    risk_counts = model_df["risk"].value_counts().reset_index()
    risk_counts.columns = ["risk", "count"]
    fig_pie = px.pie(risk_counts, names="risk", values="count", title="Risk Distribution")
    st.plotly_chart(fig_pie, use_container_width=True)

    fig_box = px.box(model_df, x="risk", y="recent_7_day_clicks", color="risk", title="Recent 7 Day Clicks vs Risk")
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
        x="weighted_avg_score",
        y="total_clicks",
        color="risk",
        hover_data=hover_fields,
        title="Weighted Score vs Total Clicks by Risk",
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

    fig_activity = px.box(
        model_df,
        x="risk",
        y="days_since_last_activity",
        color="risk",
        title="Days Since Last Activity by Risk",
    )
    st.plotly_chart(fig_activity, use_container_width=True)

    fig_trend = px.box(
        model_df,
        x="risk",
        y="engagement_trend",
        color="risk",
        title="Engagement Trend by Risk",
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    if Path(MODEL_PATH).exists():
        model = load_model()
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
            fig_imp = px.bar(importance_df.head(20), x="Feature", y="Importance", title="Feature Importance")
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Feature importance chart is available for tree-based models.")
    else:
        st.warning("Train and save a model to view feature importance.")
