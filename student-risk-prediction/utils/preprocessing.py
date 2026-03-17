"""Utilities for loading and preparing student risk data."""

from pathlib import Path
import pandas as pd

from utils.feature_engineering import add_engineered_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "students.csv"

# Single source of truth for all model feature columns.
FEATURE_COLUMNS = [
    "attendance",
    "assignments_submitted",
    "login_frequency",
    "avg_grade",
    "engagement_score",
    "performance_index",
]


def load_dataset(csv_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load dataset from CSV file."""
    return pd.read_csv(csv_path)


def prepare_model_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return full model-ready dataframe by applying feature engineering."""
    model_df = add_engineered_features(df)
    return model_df


def prepare_features_and_target(df: pd.DataFrame):
    """Split input dataframe into feature matrix X and target y."""
    model_df = prepare_model_dataframe(df)
    X = model_df[FEATURE_COLUMNS]
    y = model_df["risk"]
    return X, y
