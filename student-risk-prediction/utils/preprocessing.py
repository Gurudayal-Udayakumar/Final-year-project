"""Utility functions for loading and preparing student risk data."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "students.csv"

IDENTIFIER_COLUMNS = ["student_id"]

NUMERIC_FEATURES = [
    "num_of_prev_attempts",
    "studied_credits",
    "total_clicks",
    "active_days",
    "avg_clicks_per_day",
    "assessments_submitted",
    "avg_score",
    "late_submissions",
    "procrastination_index",
    "days_since_last_drift",
    "registration_delay_days",
    "days_registered_before_withdrawal",
    "completion_ratio",
    "weighted_avg_score",
    "recent_7_day_clicks",
    "days_since_last_activity",
    "engagement_trend",
]

CATEGORICAL_FEATURES = [
    "code_module",
    "code_presentation",
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "disability",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
OPTIONAL_COLUMNS = ["final_result", "risk"]


def load_dataset(csv_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the synthetic student feature dataset from CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)

    missing_columns = [column for column in FEATURE_COLUMNS + ["risk"] if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    for column in NUMERIC_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in CATEGORICAL_FEATURES:
        df[column] = df[column].fillna("Unknown").replace("", "Unknown").astype(str)

    if "risk" in df.columns:
        df["risk"] = df["risk"].fillna("Unknown").astype(str)

    if "final_result" in df.columns:
        df["final_result"] = df["final_result"].fillna("Unknown").astype(str)

    for column in IDENTIFIER_COLUMNS:
        if column in df.columns:
            df[column] = df[column].astype(str)

    return df


def prepare_features_and_target(df: pd.DataFrame):
    """Split the DataFrame into model features (X) and labels (y)."""
    X = df[FEATURE_COLUMNS].copy()
    y = df["risk"].copy()
    return X, y


def prepare_model_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned DataFrame ready for display and plotting in Streamlit pages."""
    cols = [*IDENTIFIER_COLUMNS, *FEATURE_COLUMNS, *OPTIONAL_COLUMNS]
    available = [column for column in cols if column in df.columns]
    model_df = df[available].copy()

    for column in NUMERIC_FEATURES:
        if column in model_df.columns:
            model_df[column] = pd.to_numeric(model_df[column], errors="coerce")
            median = model_df[column].median()
            model_df[column] = model_df[column].fillna(0 if pd.isna(median) else median)

    for column in CATEGORICAL_FEATURES:
        if column in model_df.columns:
            model_df[column] = model_df[column].fillna("Unknown").replace("", "Unknown").astype(str)

    if "risk" in model_df.columns:
        model_df["risk"] = model_df["risk"].fillna("Unknown").astype(str)

    if "final_result" in model_df.columns:
        model_df["final_result"] = model_df["final_result"].fillna("Unknown").astype(str)

    for column in IDENTIFIER_COLUMNS:
        if column in model_df.columns:
            model_df[column] = model_df[column].astype(str)

    return model_df
