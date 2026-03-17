"""Feature engineering helpers for student risk prediction."""

import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create beginner-friendly derived features used by ML models."""
    engineered_df = df.copy()

    # Average activity score: higher means better engagement with course activities.
    engineered_df["engagement_score"] = (
        engineered_df["attendance"]
        + engineered_df["login_frequency"]
        + engineered_df["assignments_submitted"]
    ) / 3

    # Weighted academic/attendance score.
    engineered_df["performance_index"] = (
        engineered_df["attendance"] * 0.4 + engineered_df["avg_grade"] * 0.6
    )

    return engineered_df
