"""Dataset exploration page."""

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.preprocessing import load_dataset, prepare_model_dataframe
from utils.visualization import build_correlation_heatmap


st.title("Dataset Explorer")
st.write("Explore the synthetic student dataset, understand its quality, and spot the patterns that matter before training or interpreting the model.")

df = load_dataset()
model_df = prepare_model_dataframe(df)

st.header("Dataset Preview")
st.caption("Description: A quick look at the raw student records currently loaded into the app.")
st.markdown(
    "**Tutor reading guide:** Use this to sense-check whether the dataset looks realistic and whether key columns such as risk, engagement, and academic progress are populated as expected."
)
st.dataframe(model_df.head(15), use_container_width=True)

st.header("Dataset Quality Snapshot")
quality_col1, quality_col2, quality_col3 = st.columns(3)
with quality_col1:
    st.metric("Rows", len(model_df))
with quality_col2:
    st.metric("Columns", model_df.shape[1])
with quality_col3:
    st.metric("Missing Values", int(model_df.isnull().sum().sum()))

st.caption("Description: A high-level summary of dataset size and completeness.")
st.markdown(
    "**Tutor reading guide:** Missing values should be low in most columns. If they are high in an important field, treat any downstream insight using that field more cautiously."
)

summary_df = model_df.describe(include="all").transpose().reset_index().rename(columns={"index": "Column"})
summary_df = summary_df.fillna("")
for column in summary_df.columns:
    summary_df[column] = summary_df[column].astype(str)

st.header("Field Summary Table")
st.caption("Description: Summary statistics for every column, including counts, unique values, and numeric spread where relevant.")
st.markdown(
    "**Tutor reading guide:** Scan this table to understand typical ranges. Outliers or extreme spreads can explain why some students stand out strongly in the model."
)
st.dataframe(summary_df, use_container_width=True)

missing_df = model_df.isnull().sum().rename("Missing Values").to_frame().reset_index().rename(columns={"index": "Column"})
missing_df["Missing Rate"] = (missing_df["Missing Values"] / len(model_df)).map(lambda value: f"{value:.1%}")

st.header("Missing Data Table")
st.caption("Description: Missing counts and rates for each field.")
st.markdown(
    "**Tutor reading guide:** Pay most attention to fields with both high importance and high missingness, because those can weaken trust in individual cases."
)
st.dataframe(missing_df, use_container_width=True)

dtype_df = pd.DataFrame(
    {
        "Column": model_df.columns,
        "Data Type": model_df.dtypes.astype(str).values,
    }
)

st.header("Column Data Types")
st.caption("Description: The storage type for each field used by the app and model.")
st.markdown(
    "**Tutor reading guide:** Numeric fields drive quantitative comparisons; categorical fields group students into labels such as region, module, and education background."
)
st.dataframe(dtype_df, use_container_width=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Learning Activity Volume")
    st.caption("Description: Distribution of total clicks, showing how much online activity students generated overall.")
    st.markdown(
        "**Tutor reading guide:** A long left tail suggests students with very low engagement. Those are often the first students worth checking for disengagement."
    )
    fig_clicks = px.histogram(model_df, x="total_clicks", nbins=30, title="Total Clicks Across Students")
    st.plotly_chart(fig_clicks, use_container_width=True)

with col2:
    st.subheader("Academic Performance Spread")
    st.caption("Description: Distribution of weighted average scores across students.")
    st.markdown(
        "**Tutor reading guide:** This helps separate broadly strong performers from students whose academic outcomes are already slipping."
    )
    fig_score = px.histogram(model_df, x="weighted_avg_score", nbins=30, title="Weighted Average Score Distribution")
    st.plotly_chart(fig_score, use_container_width=True)

with col3:
    st.subheader("Task Completion Coverage")
    st.caption("Description: Distribution of completion ratio, showing how much assigned work students are finishing.")
    st.markdown(
        "**Tutor reading guide:** Students clustered near zero completion are usually more urgent than students with solid completion but weaker scores."
    )
    fig_completion = px.histogram(model_df, x="completion_ratio", nbins=30, title="Completion Ratio Distribution")
    st.plotly_chart(fig_completion, use_container_width=True)

st.header("Risk Profile by Module")
st.caption("Description: Counts of Low, Medium, and High risk students within each course module.")
st.markdown(
    "**Tutor reading guide:** Use this to spot whether certain modules appear consistently more challenging or produce a heavier concentration of high-risk cases."
)
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

st.header("Procrastination Pattern by Risk")
st.caption("Description: Box plot of procrastination index for each risk group.")
st.markdown(
    "**Tutor reading guide:** If the higher-risk group sits noticeably higher, late or last-minute submission behavior is likely an important warning sign in this dataset."
)
fig_pi = px.box(
    model_df,
    x="risk",
    y="procrastination_index",
    color="risk",
    title="Procrastination Index Distribution by Risk",
)
st.plotly_chart(fig_pi, use_container_width=True)

st.header("Recency of Student Activity")
st.caption("Description: Box plot of days since last activity for each risk group.")
st.markdown(
    "**Tutor reading guide:** Higher values mean students have been inactive for longer. If the High-risk group trends upward here, inactivity is a practical intervention trigger."
)
fig_recent = px.box(
    model_df,
    x="risk",
    y="days_since_last_activity",
    color="risk",
    title="Days Since Last Activity by Risk",
)
st.plotly_chart(fig_recent, use_container_width=True)

st.header("Correlation Heatmap")
st.caption("Description: Pairwise relationships between numeric student features, shown from strongest negative to strongest positive association.")
st.markdown(
    "**Tutor reading guide:** Dark red cells mean two measures rise together; dark blue cells mean one tends to fall when the other rises. "
    "Use this to understand whether several warning signs are telling the same story or different ones."
)
heatmap_fig = build_correlation_heatmap(model_df)
st.pyplot(heatmap_fig)
