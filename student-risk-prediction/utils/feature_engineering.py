"""Feature engineering helpers for student risk prediction."""

import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features used consistently in training and prediction."""
    engineered_df = df.copy()

    # Engagement score combines attendance, login activity and assignment effort.
    engineered_df["engagement_score"] = (
        engineered_df["attendance"]
        + engineered_df["login_frequency"]
        + engineered_df["assignments_submitted"]
    ) / 3

    # Performance index gives slightly more weight to academic grades.
    engineered_df["performance_index"] = (
        engineered_df["attendance"] * 0.4 + engineered_df["avg_grade"] * 0.6
    )

    return engineered_df
