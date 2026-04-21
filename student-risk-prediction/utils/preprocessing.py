"""Utility functions for loading and preparing student risk data."""

from pathlib import Path
import pandas as pd


# Build project root dynamically so scripts work from any folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "students.csv"


def load_dataset(csv_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the student dataset from CSV."""
    return pd.read_csv(csv_path)


def prepare_features_and_target(df: pd.DataFrame):
    """Split the DataFrame into model features (X) and labels (y)."""
    # We do not use student_id for prediction because it is only an identifier.
    feature_columns = [
        "attendance",
        "assignments_submitted",
        "login_frequency",
        "avg_grade",
    ]
    X = df[feature_columns]
    y = df["risk"]
    return X, y


def prepare_model_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned DataFrame ready for display and plotting in Streamlit pages.

    This is a small compatibility wrapper expected by the Streamlit pages. It:
    - selects the key columns used across the app
    - fills simple missing values with sensible defaults
    - ensures numeric columns have proper dtypes
    """
    cols = [
        "student_id",
        "attendance",
        "assignments_submitted",
        "login_frequency",
        "avg_grade",
        "risk",
    ]
    # Keep only available columns to avoid KeyErrors
    available = [c for c in cols if c in df.columns]
    model_df = df[available].copy()

    # Coerce numeric columns and fill simple missing values
    for col in ["attendance", "assignments_submitted", "login_frequency", "avg_grade"]:
        if col in model_df.columns:
            model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
            model_df[col] = model_df[col].fillna(model_df[col].median())

    # For risk, keep as-is but fill missing with 'unknown'
    if "risk" in model_df.columns:
        model_df["risk"] = model_df["risk"].fillna("unknown")

    # Ensure identifier columns are string dtype to avoid Arrow conversion issues
    if "student_id" in model_df.columns:
        model_df["student_id"] = model_df["student_id"].astype(str)

    return model_df
