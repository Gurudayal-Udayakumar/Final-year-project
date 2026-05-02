"""Individual SHAP explainability dashboard for academic advisors."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import shap
import streamlit as st
import streamlit.components.v1 as components

from model.train_model import MODEL_PATH
from utils.preprocessing import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    load_dataset,
    prepare_features_and_target,
)


RISK_THRESHOLD = 0.75


@st.cache_resource
def load_pipeline():
    """Load the saved sklearn Pipeline once per Streamlit session."""
    return joblib.load(MODEL_PATH)


@st.cache_resource
def build_explainer(_model):
    """Build the SHAP explainer once per fitted model."""
    return shap.TreeExplainer(_model)


@st.cache_data(show_spinner="Loading student features from CSV...")
def load_feature_data():
    """Load model-ready synthetic student features from CSV."""
    df = load_dataset()
    X, _ = prepare_features_and_target(df)
    metadata_cols = ["student_id", "code_module", "code_presentation", "gender", "final_result", "risk"]
    metadata = df[[col for col in metadata_cols if col in df.columns]].copy()
    return X, metadata


def get_high_risk_probability(pipeline, X: pd.DataFrame):
    """Run predict_proba and return the probability assigned to High risk."""
    classes = list(pipeline.classes_)
    if "High" not in classes:
        raise ValueError(f"The saved model does not contain a 'High' class. Classes: {classes}")
    return pipeline.predict_proba(X)[:, classes.index("High")]


def select_high_class_values(explainer, shap_values, classes):
    """Select SHAP expected value and contribution matrix for the High class."""
    classes = list(classes)
    if "High" not in classes:
        raise ValueError(f"The saved model does not contain a 'High' class. Classes: {classes}")

    high_index = classes.index("High")
    expected_value = explainer.expected_value
    values = shap_values

    if isinstance(expected_value, np.ndarray):
        expected_value = expected_value.reshape(-1)[high_index if expected_value.size > 1 else 0]
    elif isinstance(expected_value, (list, tuple)):
        expected_value = expected_value[high_index if len(expected_value) > 1 else 0]

    if isinstance(values, list):
        values = values[high_index if len(values) > 1 else 0]
    elif isinstance(values, np.ndarray) and values.ndim == 3:
        values = values[:, :, high_index if values.shape[2] > 1 else 0]

    return expected_value, values


def render_force_plot(explainer, shap_values, transformed_row: pd.DataFrame, classes):
    """Render SHAP's Javascript force plot inside Streamlit."""
    expected_value, values = select_high_class_values(explainer, shap_values, classes)

    force_plot = shap.force_plot(
        expected_value,
        values[0],
        transformed_row.iloc[0],
        feature_names=transformed_row.columns.tolist(),
        matplotlib=False,
    )
    shap_html = f"""
    <head>{shap.getjs()}</head>
    <body>{force_plot.html()}</body>
    """
    components.html(shap_html, height=340, scrolling=True)


def prettify_feature_name(feature_name: str) -> str:
    """Convert feature names into tutor-friendly labels."""
    return feature_name.replace("_", " ").title()


def resolve_base_feature(transformed_name: str) -> str:
    """Map transformed feature names back to their original source feature."""
    if transformed_name.startswith("num__"):
        return transformed_name.removeprefix("num__")

    if transformed_name.startswith("cat__"):
        encoded_name = transformed_name.removeprefix("cat__")
        for feature in sorted(CATEGORICAL_FEATURES, key=len, reverse=True):
            prefix = f"{feature}_"
            if encoded_name == feature or encoded_name.startswith(prefix):
                return feature

    return transformed_name


def aggregate_local_contributions(
    transformed_row: pd.DataFrame,
    local_values: np.ndarray,
    original_row: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate one-hot and scaled SHAP values back to original feature names."""
    raw_df = pd.DataFrame(
        {
            "Transformed Feature": transformed_row.columns,
            "SHAP Value": local_values,
        }
    )
    raw_df["Feature"] = raw_df["Transformed Feature"].map(resolve_base_feature)

    grouped = (
        raw_df.groupby("Feature", as_index=False)["SHAP Value"]
        .sum()
        .sort_values("SHAP Value", key=lambda series: series.abs(), ascending=False)
    )
    grouped["Absolute Impact"] = grouped["SHAP Value"].abs()
    grouped["Direction"] = grouped["SHAP Value"].map(
        lambda value: "Raises High-Risk Score" if value > 0 else "Lowers High-Risk Score"
    )
    grouped["Feature Value"] = grouped["Feature"].map(lambda name: original_row.iloc[0][name] if name in original_row.columns else "")
    grouped["Display Feature"] = grouped["Feature"].map(prettify_feature_name)
    return grouped


def build_waterfall_figure(driver_df: pd.DataFrame, expected_value: float):
    """Create a readable SHAP waterfall plot from aggregated contributions."""
    plot_df = driver_df.copy().head(12).iloc[::-1]
    explanation = shap.Explanation(
        values=plot_df["SHAP Value"].to_numpy(),
        base_values=expected_value,
        data=plot_df["Feature Value"].astype(str).to_numpy(),
        feature_names=plot_df["Display Feature"].to_list(),
    )

    fig, _ = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(explanation, max_display=len(plot_df), show=False)
    plt.tight_layout()
    return fig


def build_direction_bar_chart(driver_df: pd.DataFrame):
    """Create a tutor-friendly bar chart of local feature effects."""
    plot_df = driver_df.head(12).copy().iloc[::-1]
    plot_df["Bar Label"] = plot_df.apply(
        lambda row: f"{row['Display Feature']} ({row['Feature Value']})",
        axis=1,
    )

    fig = px.bar(
        plot_df,
        x="SHAP Value",
        y="Bar Label",
        color="Direction",
        orientation="h",
        color_discrete_map={
            "Raises High-Risk Score": "#d84c4c",
            "Lowers High-Risk Score": "#3b82f6",
        },
        title="Top Factors Pushing the Prediction Up or Down",
    )
    fig.update_layout(
        xaxis_title="SHAP contribution to High-risk prediction",
        yaxis_title="Feature and current value",
        legend_title_text="Effect on predicted risk",
        height=500,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#999999")
    return fig


st.title("Individual Student Explainability")
st.write("Review high-risk synthetic student profiles and inspect the factors driving each prediction.")

if not Path(MODEL_PATH).exists():
    st.warning("Model file not found. Please train the model first.")
    st.stop()

pipeline = load_pipeline()

if not hasattr(pipeline, "named_steps"):
    st.error("The saved model is not a sklearn Pipeline.")
    st.stop()

if "preprocess" not in pipeline.named_steps or "model" not in pipeline.named_steps:
    st.error("The saved pipeline must contain 'preprocess' and 'model' steps.")
    st.stop()

preprocessor = pipeline.named_steps["preprocess"]
model = pipeline.named_steps["model"]

X, metadata = load_feature_data()

try:
    probabilities = get_high_risk_probability(pipeline, X)
except Exception as exc:
    st.error(f"Unable to calculate risk probabilities: {exc}")
    st.stop()

results = metadata.copy()
results["high_risk_probability"] = probabilities
results["risk_category"] = results["high_risk_probability"].map(
    lambda probability: "High Risk" if probability > RISK_THRESHOLD else "Monitor"
)

high_risk = results[results["risk_category"] == "High Risk"].copy()
high_risk = high_risk.sort_values("high_risk_probability", ascending=False)

st.metric("High Risk Students", len(high_risk))
st.caption(f"High Risk threshold: probability > {RISK_THRESHOLD:.0%}")

if high_risk.empty:
    st.info("No students currently exceed the high-risk threshold.")
    st.stop()

st.subheader("Student Search And Sort")
st.caption("Description: Find a student quickly by ID and sort the explainability list by the tutor fields most useful for review.")
st.markdown(
    "**Tutor reading guide:** Search by student ID when reviewing a specific learner. "
    "Use sorting to group students by outcome, module, gender, or predicted risk strength before opening an explanation."
)

control_col1, control_col2, control_col3 = st.columns([2, 1, 1])
with control_col1:
    student_search = st.text_input("Search Student ID", placeholder="e.g. STU0042")
with control_col2:
    sort_by_label = st.selectbox(
        "Sort By",
        [
            "High Risk Probability",
            "Student ID",
            "Risk",
            "Gender",
            "Final Result",
            "Core Module",
        ],
    )
with control_col3:
    sort_direction = st.selectbox("Sort Order", ["Descending", "Ascending"])

sort_column_map = {
    "High Risk Probability": "high_risk_probability",
    "Student ID": "student_id",
    "Risk": "risk",
    "Gender": "gender",
    "Final Result": "final_result",
    "Core Module": "code_module",
}

filtered_high_risk = high_risk.copy()
if student_search.strip():
    search_value = student_search.strip().lower()
    filtered_high_risk = filtered_high_risk[
        filtered_high_risk["student_id"].astype(str).str.lower().str.contains(search_value, na=False)
    ]

if filtered_high_risk.empty:
    st.warning("No high-risk students matched that search.")
    st.stop()

sort_column = sort_column_map[sort_by_label]
sort_ascending = sort_direction == "Ascending"
filtered_high_risk = filtered_high_risk.sort_values(sort_column, ascending=sort_ascending, kind="stable")

labels = filtered_high_risk.apply(
    lambda row: (
        f"{row.get('student_id', 'student unknown')} | "
        f"{row.get('code_module', 'module unknown')} "
        f"{row.get('code_presentation', '')} | "
        f"{row['high_risk_probability']:.1%}"
    ),
    axis=1,
).tolist()

selected_label = st.selectbox("Select High Risk Student", labels)
selected_index = filtered_high_risk.index[labels.index(selected_label)]

selected_metadata = results.loc[selected_index]
selected_row = X.loc[[selected_index], FEATURE_COLUMNS]

st.subheader("Student Risk Snapshot")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Student ID", str(selected_metadata.get("student_id", "Unknown")))
with col2:
    st.metric("High Risk Probability", f"{selected_metadata['high_risk_probability']:.1%}")
with col3:
    st.metric("Observed Outcome", str(selected_metadata.get("final_result", "Unknown")))

snapshot_df = selected_row.T.rename(columns={selected_index: "value"})
snapshot_df["value"] = snapshot_df["value"].astype(str)
st.dataframe(snapshot_df, use_container_width=True)

try:
    transformed_row = preprocessor.transform(selected_row)
    if not isinstance(transformed_row, pd.DataFrame):
        transformed_row = pd.DataFrame(
            transformed_row,
            columns=preprocessor.get_feature_names_out(),
        )

    explainer = build_explainer(model)
    shap_values = explainer.shap_values(transformed_row)
    expected_value, high_class_values = select_high_class_values(explainer, shap_values, pipeline.classes_)

    local_values = high_class_values[0]
    driver_df = aggregate_local_contributions(transformed_row, local_values, selected_row)

    st.subheader("Risk Contribution Waterfall")
    st.caption(
        "Description: This shows how the student's feature values move the prediction away from the model's baseline "
        "and toward a higher-risk or lower-risk outcome."
    )
    st.markdown(
        "**Tutor reading guide:** Red segments increase High-risk likelihood, blue segments reduce it. "
        "Start with the largest bars to identify the strongest reasons the model is concerned or reassured."
    )
    waterfall_fig = build_waterfall_figure(driver_df, expected_value)
    st.pyplot(waterfall_fig, clear_figure=True)

    st.subheader("Top Factors Pushing the Prediction Up or Down")
    st.caption(
        "Description: This ranks the student's strongest local drivers from most important to least important for this prediction."
    )
    st.markdown(
        "**Tutor reading guide:** Focus first on the longest red bars as intervention priorities. "
        "Long blue bars are protective signals that may offset some risk."
    )
    direction_fig = build_direction_bar_chart(driver_df)
    st.plotly_chart(direction_fig, use_container_width=True)

    st.subheader("Tutor Summary Table")
    st.caption(
        "Description: A compact table of the same local drivers, including the student's observed feature value and the size of each effect."
    )
    st.markdown(
        "**Tutor reading guide:** Use this table when discussing specifics with the student. "
        "Look for features with large absolute impact and ask whether they match the student's real situation."
    )
    st.dataframe(
        driver_df[["Display Feature", "Feature Value", "SHAP Value", "Absolute Impact", "Direction"]]
        .head(15)
        .assign(**{"Feature Value": lambda df: df["Feature Value"].astype(str)})
        .rename(columns={"Display Feature": "Feature"}),
        use_container_width=True,
    )

    with st.expander("Advanced View: Interactive Force Plot"):
        st.caption(
            "Description: A detailed SHAP force plot for technical users who want to inspect the prediction balance at a finer level."
        )
        st.markdown(
            "**Tutor reading guide:** Use this only for deep dives. The waterfall and ranked factors above are the easier starting point."
        )
        try:
            render_force_plot(explainer, shap_values, transformed_row, pipeline.classes_)
        except Exception as plot_exc:
            st.warning(f"Interactive force plot unavailable in this environment: {plot_exc}")

except Exception as exc:
    st.error(f"Unable to generate individualized SHAP explanation: {exc}")
    st.info("Confirm the saved model is a fitted LightGBM pipeline with 'preprocess' and 'model' steps.")
