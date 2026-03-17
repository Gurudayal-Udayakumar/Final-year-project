"""Utilities for loading and preparing student risk data."""

from pathlib import Path
import pandas as pd

from utils.feature_engineering import add_engineered_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "students.csv"


BASE_FEATURE_COLUMNS = [
    "attendance",
    "assignments_submitted",
    "login_frequency",
    "avg_grade",
]

ENGINEERED_FEATURE_COLUMNS = ["engagement_score", "performance_index"]


def load_dataset(csv_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load dataset from CSV."""
    return pd.read_csv(csv_path)


def prepare_model_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features and return transformed DataFrame."""
    return add_engineered_features(df)


def prepare_features_and_target(df: pd.DataFrame):
    """Return feature matrix X and label vector y for model training."""
    model_df = prepare_model_dataframe(df)
    feature_columns = BASE_FEATURE_COLUMNS + ENGINEERED_FEATURE_COLUMNS
    X = model_df[feature_columns]
    y = model_df["risk"]
    return X, y
