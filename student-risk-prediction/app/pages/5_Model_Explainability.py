"""Individual SHAP explainability dashboard for academic advisors."""

from pathlib import Path

import joblib
import pandas as pd
import shap
import streamlit as st
import streamlit.components.v1 as components

from model.train_model import MODEL_PATH
from utils.preprocessing import FEATURE_COLUMNS, load_dataset, prepare_features_and_target


RISK_THRESHOLD = 0.75


@st.cache_resource
def load_pipeline():
    """Load the saved sklearn Pipeline once per Streamlit session."""
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner="Loading student features from DuckDB...")
def load_feature_data():
    """Load model-ready OULAD features from DuckDB."""
    df = load_dataset()
    X, _ = prepare_features_and_target(df)
    metadata_cols = ["id_student", "code_module", "code_presentation", "final_result", "risk"]
    metadata = df[[col for col in metadata_cols if col in df.columns]].copy()
    return X, metadata


def get_high_risk_probability(pipeline, X: pd.DataFrame):
    """Run predict_proba and return the probability assigned to High risk."""
    classes = list(pipeline.classes_)
    if "High" not in classes:
        raise ValueError(f"The saved model does not contain a 'High' class. Classes: {classes}")
    return pipeline.predict_proba(X)[:, classes.index("High")]


def select_high_class_values(explainer, shap_values):
    """Select SHAP expected value and contribution matrix for the High class."""
    expected_value = explainer.expected_value
    values = shap_values

    if isinstance(expected_value, (list, tuple)):
        high_index = 1 if len(expected_value) > 1 else 0
        expected_value = expected_value[high_index]
    if isinstance(values, list):
        high_index = 1 if len(values) > 1 else 0
        values = values[high_index]

    return expected_value, values


def render_force_plot(explainer, shap_values, transformed_row: pd.DataFrame):
    """Render SHAP's Javascript force plot inside Streamlit."""
    expected_value, values = select_high_class_values(explainer, shap_values)

    shap.initjs()
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


st.title("Individual Student Explainability")
st.write("Review high-risk students and inspect the model factors behind each prediction.")

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

labels = high_risk.apply(
    lambda row: (
        f"{row['id_student']} | {row.get('code_module', 'module unknown')} "
        f"{row.get('code_presentation', '')} | "
        f"{row['high_risk_probability']:.1%}"
    ),
    axis=1,
).tolist()

selected_label = st.selectbox("Select High Risk Student", labels)
selected_index = high_risk.index[labels.index(selected_label)]

selected_metadata = results.loc[selected_index]
selected_row = X.loc[[selected_index], FEATURE_COLUMNS]

st.subheader("Student Risk Snapshot")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Student ID", str(selected_metadata["id_student"]))
with col2:
    st.metric("High Risk Probability", f"{selected_metadata['high_risk_probability']:.1%}")
with col3:
    st.metric("Observed Outcome", str(selected_metadata.get("final_result", "Unknown")))

st.dataframe(
    selected_row.T.rename(columns={selected_index: "value"}),
    use_container_width=True,
)

try:
    transformed_row = preprocessor.transform(selected_row)
    if not isinstance(transformed_row, pd.DataFrame):
        transformed_row = pd.DataFrame(
            transformed_row,
            columns=preprocessor.get_feature_names_out(),
        )

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(transformed_row)
    _, high_class_values = select_high_class_values(explainer, shap_values)

    st.subheader("Individual SHAP Force Plot")
    render_force_plot(explainer, shap_values, transformed_row)

    st.subheader("Top Local Drivers")
    local_values = high_class_values[0]
    driver_df = pd.DataFrame(
        {
            "Feature": transformed_row.columns,
            "Feature Value": transformed_row.iloc[0].values,
            "SHAP Value": local_values,
            "Absolute Impact": abs(local_values),
        }
    ).sort_values("Absolute Impact", ascending=False)
    st.dataframe(driver_df.head(15), use_container_width=True)

except Exception as exc:
    st.error(f"Unable to generate individualized SHAP explanation: {exc}")
    st.info("Confirm the saved model is a fitted LightGBM pipeline with 'preprocess' and 'model' steps.")
