"""Dataset exploration page."""

import streamlit as st
import plotly.express as px

from utils.preprocessing import load_dataset, prepare_model_dataframe
from utils.visualization import build_correlation_heatmap

st.title("📊 Dataset Explorer")

df = load_dataset()
model_df = prepare_model_dataframe(df)

st.header("Dataset Preview")
st.dataframe(model_df.head(15), use_container_width=True)

st.header("Summary Statistics")
st.dataframe(model_df.describe(include="all"), use_container_width=True)

st.header("Missing Values")
st.dataframe(model_df.isnull().sum().rename("missing_values").to_frame())

st.header("Column Data Types")
st.dataframe(model_df.dtypes.rename("dtype").to_frame())

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Total Clicks Distribution")
    fig_clicks = px.histogram(model_df, x="total_clicks", nbins=30, title="Total VLE Clicks")
    st.plotly_chart(fig_clicks, use_container_width=True)

with col2:
    st.subheader("Assessment Score Distribution")
    fig_score = px.histogram(model_df, x="avg_score", nbins=30, title="Average Assessment Score")
    st.plotly_chart(fig_score, use_container_width=True)

with col3:
    st.subheader("Active Days Distribution")
    fig_days = px.histogram(model_df, x="active_days", nbins=30, title="Active VLE Days")
    st.plotly_chart(fig_days, use_container_width=True)

st.header("Risk by Module")
module_risk = model_df.groupby(["code_module", "risk"]).size().reset_index(name="students")
fig_module = px.bar(
    module_risk,
    x="code_module",
    y="students",
    color="risk",
    barmode="group",
    title="Risk Distribution by Course Module",
)
st.plotly_chart(fig_module, use_container_width=True)

st.header("Procrastination Index by Risk")
fig_pi = px.box(
    model_df,
    x="risk",
    y="procrastination_index",
    color="risk",
    title="Procrastination Index Distribution by Risk",
)
st.plotly_chart(fig_pi, use_container_width=True)

st.header("Engagement Drift Recency by Risk")
fig_drift = px.box(
    model_df,
    x="risk",
    y="days_since_last_drift",
    color="risk",
    title="Days Since Last Negative Engagement Drift by Risk",
)
st.plotly_chart(fig_drift, use_container_width=True)

st.header("Correlation Heatmap")
heatmap_fig = build_correlation_heatmap(model_df)
st.pyplot(heatmap_fig)
