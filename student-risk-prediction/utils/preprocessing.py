"""Utility functions for loading and preparing student risk data."""

from pathlib import Path
import pandas as pd

from utils.duckdb_loader import DB_PATH, initialize_oulad_db, load_student_features


# Build project root dynamically so scripts work from any folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "students.csv"

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


def load_dataset(csv_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the OULAD student feature dataset from DuckDB."""
    if not DB_PATH.exists():
        con = initialize_oulad_db()
        con.close()
    return load_student_features()


def prepare_features_and_target(df: pd.DataFrame):
    """Split the DataFrame into model features (X) and labels (y)."""
    # Identifiers and final_result are excluded to avoid leakage.
    X = df[FEATURE_COLUMNS]
    y = df["risk"]
    return X, y


def prepare_model_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned DataFrame ready for display and plotting in Streamlit pages.

    This is a small compatibility wrapper expected by the Streamlit pages. It:
    - selects the key columns used across the app
    - fills simple missing values with sensible defaults
    - ensures numeric columns have proper dtypes
    """
    cols = ["id_student", *FEATURE_COLUMNS, "final_result", "risk"]
    # Keep only available columns to avoid KeyErrors
    available = [c for c in cols if c in df.columns]
    model_df = df[available].copy()

    # Coerce numeric columns and fill simple missing values
    for col in NUMERIC_FEATURES:
        if col in model_df.columns:
            model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
            model_df[col] = model_df[col].fillna(model_df[col].median())

    for col in CATEGORICAL_FEATURES:
        if col in model_df.columns:
            model_df[col] = model_df[col].fillna("Unknown").astype(str)

    # For risk, keep as-is but fill missing with 'unknown'
    if "risk" in model_df.columns:
        model_df["risk"] = model_df["risk"].fillna("Unknown")

    # Ensure identifier columns are string dtype to avoid Arrow conversion issues
    if "id_student" in model_df.columns:
        model_df["id_student"] = model_df["id_student"].astype(str)

    return model_df
