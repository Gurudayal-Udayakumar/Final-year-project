"""Generate synthetic student-risk training data aligned with the app schema."""

from __future__ import annotations

import csv
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "students.csv"
DEFAULT_ROWS = 1000
DEFAULT_SEED = 42

MODULES = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"]
PRESENTATIONS = ["2013B", "2013J", "2014B", "2014J"]
REGIONS = [
    "East Anglian Region",
    "East Midlands Region",
    "Ireland",
    "London Region",
    "North Region",
    "North Western Region",
    "Scotland",
    "South East Region",
    "South Region",
    "South West Region",
    "Wales",
    "West Midlands Region",
    "Yorkshire Region",
]
EDUCATION_LEVELS = [
    "A Level or Equivalent",
    "HE Qualification",
    "Lower Than A Level",
    "No Formal quals",
    "Post Graduate Qualification",
]
IMD_BANDS = [
    "0-10%",
    "10-20",
    "20-30%",
    "30-40%",
    "40-50%",
    "50-60%",
    "60-70%",
    "70-80%",
    "80-90%",
    "90-100%",
]
OUTCOMES = ["Withdrawn", "Fail", "Pass", "Distinction"]
OUTCOME_WEIGHTS = [10156, 7052, 12361, 3024]

HEADERS = [
    "student_id",
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
    "code_module",
    "code_presentation",
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "disability",
    "registration_delay_days",
    "days_registered_before_withdrawal",
    "completion_ratio",
    "weighted_avg_score",
    "recent_7_day_clicks",
    "days_since_last_activity",
    "engagement_trend",
    "final_result",
    "risk",
]


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _risk_from_result(final_result: str) -> str:
    if final_result in {"Withdrawn", "Fail"}:
        return "High"
    if final_result == "Pass":
        return "Medium"
    return "Low"


def _build_row(index: int, rng: random.Random) -> list[object]:
    final_result = rng.choices(OUTCOMES, weights=OUTCOME_WEIGHTS, k=1)[0]
    risk = _risk_from_result(final_result)

    student_id = f"STU{index:04d}"
    module = rng.choice(MODULES)
    presentation = rng.choice(PRESENTATIONS)
    gender = rng.choice(["F", "M"])
    region = rng.choice(REGIONS)
    education = rng.choices(
        EDUCATION_LEVELS,
        weights=[45, 28, 15, 7, 5],
        k=1,
    )[0]
    imd_band = rng.choice(IMD_BANDS) if rng.random() > 0.034 else ""
    age_band = rng.choices(["0-35", "35-55", "55<="], weights=[70, 25, 5], k=1)[0]
    disability = rng.choices(["N", "Y"], weights=[90, 10], k=1)[0]
    attempts = rng.choices([0, 1, 2, 3, 4, 5, 6], weights=[82, 10, 4, 2, 1, 0.6, 0.4], k=1)[0]
    credits = rng.choices([30, 60, 90, 120, 150, 180], weights=[8, 42, 12, 32, 3, 3], k=1)[0]

    if final_result == "Distinction":
        assessments = rng.randint(10, 14)
        avg_score = round(rng.uniform(82.0, 100.0), 1)
        active_days = rng.randint(110, 240)
        total_clicks = round(rng.uniform(3000.0, 18000.0), 1)
        late_submissions = rng.randint(0, 1)
        procrastination = round(rng.uniform(0.0, 0.18), 2)
        drift = rng.randint(5, 70)
        reg_delay = rng.randint(-60, -8)
        days_withdrawal = ""
        completion = round(rng.uniform(0.9, 1.0), 2)
        recent_clicks = rng.randint(180, 520)
        days_since_activity = rng.randint(0, 2)
        trend = round(rng.uniform(0.7, 2.8), 1)

    elif final_result == "Pass":
        assessments = rng.randint(6, 12)
        avg_score = round(rng.uniform(50.0, 84.0), 1)
        active_days = rng.randint(50, 180)
        total_clicks = round(rng.uniform(700.0, 6500.0), 1)
        late_submissions = rng.randint(0, min(3, assessments))
        procrastination = round(rng.uniform(0.1, 0.5), 2)
        drift = rng.randint(20, 100)
        reg_delay = rng.randint(-45, 5)
        days_withdrawal = ""
        completion = round(rng.uniform(0.6, 0.95), 2)
        recent_clicks = rng.randint(40, 260)
        days_since_activity = rng.randint(1, 12)
        trend = round(rng.uniform(-0.4, 1.6), 1)

    elif final_result == "Fail":
        assessments = rng.randint(1, 8)
        avg_score = round(rng.uniform(18.0, 55.0), 1)
        active_days = rng.randint(8, 85)
        total_clicks = round(rng.uniform(40.0, 1400.0), 1)
        late_submissions = rng.randint(1, min(5, assessments))
        procrastination = round(rng.uniform(0.4, 0.9), 2)
        drift = rng.randint(0, 85)
        reg_delay = rng.randint(-18, 12)
        days_withdrawal = ""
        completion = round(rng.uniform(0.1, 0.6), 2)
        recent_clicks = rng.randint(0, 90)
        days_since_activity = rng.randint(10, 55)
        trend = round(rng.uniform(-2.1, 0.0), 1)

    else:
        assessments = rng.randint(0, 3)
        avg_score = round(rng.uniform(0.0, 40.0), 1) if assessments > 0 else 0.0
        active_days = rng.randint(0, 30)
        total_clicks = round(rng.uniform(0.0, 320.0), 1) if active_days > 0 else 0.0
        late_submissions = rng.randint(0, min(2, assessments)) if assessments > 0 else 0
        procrastination = round(rng.uniform(0.5, 1.0), 2) if assessments > 0 else 0.0
        drift = rng.randint(0, 25)
        reg_delay = rng.randint(0, 30)
        days_withdrawal = rng.randint(5, 120)
        completion = round(rng.uniform(0.0, 0.3), 2) if assessments > 0 else 0.0
        recent_clicks = rng.randint(0, 10)
        days_since_activity = rng.randint(30, 140)
        trend = round(rng.uniform(-4.0, -1.0), 1)

    avg_clicks_per_day = round(total_clicks / active_days, 2) if active_days > 0 else 0.0
    weighted_avg = round(_clamp(avg_score * rng.uniform(0.92, 1.08), 0.0, 100.0), 1) if avg_score > 0 else 0.0

    return [
        student_id,
        attempts,
        credits,
        total_clicks,
        active_days,
        avg_clicks_per_day,
        assessments,
        avg_score,
        late_submissions,
        procrastination,
        drift,
        module,
        presentation,
        gender,
        region,
        education,
        imd_band,
        age_band,
        disability,
        reg_delay,
        days_withdrawal,
        completion,
        weighted_avg,
        recent_clicks,
        days_since_activity,
        trend,
        final_result,
        risk,
    ]


def generate_synthetic_data(
    num_rows: int = DEFAULT_ROWS,
    filename: str | Path = DEFAULT_OUTPUT,
    seed: int = DEFAULT_SEED,
) -> Path:
    """Generate a CSV file of synthetic student-risk data."""
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(HEADERS)
        for index in range(1, num_rows + 1):
            writer.writerow(_build_row(index, rng))

    return output_path


if __name__ == "__main__":
    path = generate_synthetic_data()
    print(f"Synthetic data generation complete: {path}")
