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
    st.subheader("Attendance Distribution")
    fig_att = px.histogram(model_df, x="attendance", nbins=15, title="Attendance Histogram")
    st.plotly_chart(fig_att, use_container_width=True)

with col2:
    st.subheader("Grade Distribution")
    fig_grade = px.histogram(model_df, x="avg_grade", nbins=15, title="Average Grade Histogram")
    st.plotly_chart(fig_grade, use_container_width=True)

with col3:
    st.subheader("Login Frequency Distribution")
    fig_login = px.histogram(model_df, x="login_frequency", nbins=15, title="Login Frequency Histogram")
    st.plotly_chart(fig_login, use_container_width=True)

st.header("Correlation Heatmap")
heatmap_fig = build_correlation_heatmap(model_df)
st.pyplot(heatmap_fig)
