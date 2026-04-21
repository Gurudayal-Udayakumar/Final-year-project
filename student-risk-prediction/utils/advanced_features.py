"""Advanced behavioral feature engineering helpers."""

import pandas as pd


def calculate_days_since_last_negative_drift(
    daily_clicks: pd.Series,
    warmup_window: int = 21,
) -> int:
    """Detect recent chronic engagement decline with a negative CUSUM chart.

    The first ``warmup_window`` days define the student's baseline. A drift alarm
    is recorded when the negative CUSUM exceeds three baseline standard
    deviations. After an alarm, the cumulative sum resets so later drifts can be
    detected independently.
    """
    if not isinstance(daily_clicks, pd.Series):
        daily_clicks = pd.Series(daily_clicks)

    clicks = pd.to_numeric(daily_clicks, errors="coerce").fillna(0)

    if len(clicks) < warmup_window:
        raise ValueError(
            f"Need at least {warmup_window} days to calculate the baseline."
        )

    baseline = clicks.iloc[:warmup_window]
    baseline_mean = baseline.mean()
    baseline_std = baseline.std(ddof=0)

    if baseline_std == 0:
        return len(clicks)

    k = 0.5 * baseline_std
    h = 3.0 * baseline_std
    cusum = 0.0
    last_drift_day = None

    for day_index, current_day_clicks in enumerate(clicks):
        cusum = max(0.0, ((baseline_mean - k) - current_day_clicks) + cusum)

        if cusum > h:
            last_drift_day = day_index
            cusum = 0.0

    if last_drift_day is None:
        return len(clicks)

    final_day_index = len(clicks) - 1
    return int(final_day_index - last_drift_day)
