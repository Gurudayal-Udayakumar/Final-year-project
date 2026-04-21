"""Feature engineering helpers for student risk prediction."""

import pandas as pd


def calculate_procrastination_index(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate each student's Procrastination Index from assessment submissions.

    For each assessment, submissions on or after the assessment's 75th percentile
    submission day are flagged as late. A student's PI is the mean of those flags
    across their submitted assessments, producing a continuous score from 0 to 1.
    """
    required_cols = {"id_student", "id_assessment", "date_submitted"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    working_df = df[["id_student", "id_assessment", "date_submitted"]].copy()
    working_df["date_submitted"] = pd.to_numeric(
        working_df["date_submitted"],
        errors="coerce",
    )
    working_df = working_df.dropna(
        subset=["id_student", "id_assessment", "date_submitted"]
    )

    working_df["assessment_75th_percentile"] = working_df.groupby("id_assessment")[
        "date_submitted"
    ].transform(lambda values: values.quantile(0.75))
    working_df["late_flag"] = (
        working_df["date_submitted"] >= working_df["assessment_75th_percentile"]
    ).astype(float)

    return (
        working_df.groupby("id_student", as_index=False)["late_flag"]
        .mean()
        .rename(columns={"late_flag": "procrastination_index"})
    )


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
