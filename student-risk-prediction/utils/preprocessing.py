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
