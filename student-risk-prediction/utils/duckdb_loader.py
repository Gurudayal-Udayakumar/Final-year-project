"""DuckDB helpers for loading OULAD and building model-ready features."""

from pathlib import Path
from typing import Iterable

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OULAD_DIR = DATA_DIR / "oulad"
DB_PATH = DATA_DIR / "student_risk.duckdb"


OULAD_TABLES = {
    "courses": "courses.csv",
    "assessments": "assessments.csv",
    "student_assessment": "studentAssessment.csv",
    "student_info": "studentInfo.csv",
    "student_registration": "studentRegistration.csv",
    "student_vle": "studentVle.csv",
    "vle": "vle.csv",
}


def get_connection(database_path: Path = DB_PATH) -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection for the project database."""
    database_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(database_path))


def initialize_oulad_db(
    oulad_dir: Path = OULAD_DIR,
    database_path: Path = DB_PATH,
) -> duckdb.DuckDBPyConnection:
    """Import OULAD CSV files and create the student_features view."""
    missing_files = _missing_oulad_files(oulad_dir)
    if missing_files:
        missing = ", ".join(path.name for path in missing_files)
        raise FileNotFoundError(f"Missing OULAD CSV file(s): {missing}")

    con = get_connection(database_path)
    _import_oulad_tables(con, oulad_dir)
    create_student_features_view(con)
    con.execute("CHECKPOINT")
    return con


def load_student_features(database_path: Path = DB_PATH):
    """Load the OULAD feature view as a pandas DataFrame."""
    con = get_connection(database_path)
    try:
        _ensure_student_features_view(con)
        return con.execute(
            """
            SELECT *
            FROM student_features
            WHERE risk <> 'Unknown'
            """
        ).df()
    finally:
        con.close()


def create_student_features_view(con: duckdb.DuckDBPyConnection) -> None:
    """Create an ML-ready feature view from the imported OULAD tables."""
    con.execute(
        """
        CREATE OR REPLACE VIEW student_features AS
        WITH RECURSIVE vle_features AS (
            SELECT
                code_module,
                code_presentation,
                id_student,
                SUM(TRY_CAST(sum_click AS INTEGER)) AS total_clicks,
                COUNT(DISTINCT TRY_CAST(date AS INTEGER)) AS active_days,
                AVG(TRY_CAST(sum_click AS DOUBLE)) AS avg_clicks_per_day
            FROM student_vle
            GROUP BY code_module, code_presentation, id_student
        ),
        assessment_features AS (
            SELECT
                a.code_module,
                a.code_presentation,
                sa.id_student,
                COUNT(*) AS assessments_submitted,
                AVG(TRY_CAST(sa.score AS DOUBLE)) AS avg_score,
                SUM(
                    CASE
                        WHEN TRY_CAST(a.date AS INTEGER) IS NOT NULL
                         AND TRY_CAST(sa.date_submitted AS INTEGER) > TRY_CAST(a.date AS INTEGER)
                        THEN 1
                        ELSE 0
                    END
                ) AS late_submissions
            FROM student_assessment AS sa
            JOIN assessments AS a
                ON sa.id_assessment = a.id_assessment
            GROUP BY a.code_module, a.code_presentation, sa.id_student
        ),
        assessment_submission_percentiles AS (
            SELECT
                id_assessment,
                quantile_cont(TRY_CAST(date_submitted AS DOUBLE), 0.75) AS assessment_75th_percentile
            FROM student_assessment
            WHERE TRY_CAST(date_submitted AS DOUBLE) IS NOT NULL
            GROUP BY id_assessment
        ),
        procrastination_features AS (
            SELECT
                sa.id_student,
                AVG(
                    CASE
                        WHEN TRY_CAST(sa.date_submitted AS DOUBLE) >= asp.assessment_75th_percentile
                        THEN 1.0
                        ELSE 0.0
                    END
                ) AS procrastination_index
            FROM student_assessment AS sa
            JOIN assessment_submission_percentiles AS asp
                ON sa.id_assessment = asp.id_assessment
            WHERE TRY_CAST(sa.date_submitted AS DOUBLE) IS NOT NULL
            GROUP BY sa.id_student
        ),
        student_course_days AS (
            SELECT
                si.code_module,
                si.code_presentation,
                si.id_student,
                day_index
            FROM student_info AS si
            CROSS JOIN range(0, 100) AS days(day_index)
        ),
        daily_clicks AS (
            SELECT
                code_module,
                code_presentation,
                id_student,
                TRY_CAST(date AS INTEGER) AS day_index,
                SUM(TRY_CAST(sum_click AS DOUBLE)) AS daily_clicks
            FROM student_vle
            WHERE TRY_CAST(date AS INTEGER) BETWEEN 0 AND 99
            GROUP BY code_module, code_presentation, id_student, TRY_CAST(date AS INTEGER)
        ),
        complete_daily_clicks AS (
            SELECT
                scd.code_module,
                scd.code_presentation,
                scd.id_student,
                scd.day_index,
                COALESCE(dc.daily_clicks, 0) AS daily_clicks
            FROM student_course_days AS scd
            LEFT JOIN daily_clicks AS dc
                ON scd.code_module = dc.code_module
                AND scd.code_presentation = dc.code_presentation
                AND scd.id_student = dc.id_student
                AND scd.day_index = dc.day_index
        ),
        click_baselines AS (
            SELECT
                code_module,
                code_presentation,
                id_student,
                AVG(daily_clicks) AS baseline_mean,
                STDDEV_POP(daily_clicks) AS baseline_std
            FROM complete_daily_clicks
            WHERE day_index < 21
            GROUP BY code_module, code_presentation, id_student
        ),
        cusum_scan AS (
            SELECT
                cdc.code_module,
                cdc.code_presentation,
                cdc.id_student,
                cdc.day_index,
                cdc.daily_clicks,
                cb.baseline_mean,
                cb.baseline_std,
                CASE
                    WHEN cb.baseline_std = 0 THEN 0.0
                    ELSE GREATEST(
                        0.0,
                        (cb.baseline_mean - (0.5 * cb.baseline_std)) - cdc.daily_clicks
                    )
                END AS raw_cusum,
                CASE
                    WHEN cb.baseline_std > 0
                     AND GREATEST(
                        0.0,
                        (cb.baseline_mean - (0.5 * cb.baseline_std)) - cdc.daily_clicks
                     ) > (3.0 * cb.baseline_std)
                    THEN 1
                    ELSE 0
                END AS drift_alarm
            FROM complete_daily_clicks AS cdc
            JOIN click_baselines AS cb
                ON cdc.code_module = cb.code_module
                AND cdc.code_presentation = cb.code_presentation
                AND cdc.id_student = cb.id_student
            WHERE cdc.day_index = 0

            UNION ALL

            SELECT
                cdc.code_module,
                cdc.code_presentation,
                cdc.id_student,
                cdc.day_index,
                cdc.daily_clicks,
                cs.baseline_mean,
                cs.baseline_std,
                CASE
                    WHEN cs.baseline_std = 0 THEN 0.0
                    WHEN GREATEST(
                        0.0,
                        (cs.baseline_mean - (0.5 * cs.baseline_std))
                        - cdc.daily_clicks
                        + CASE
                            WHEN cs.drift_alarm = 1 THEN 0.0
                            ELSE cs.raw_cusum
                          END
                    ) > (3.0 * cs.baseline_std)
                    THEN 0.0
                    ELSE GREATEST(
                        0.0,
                        (cs.baseline_mean - (0.5 * cs.baseline_std))
                        - cdc.daily_clicks
                        + CASE
                            WHEN cs.drift_alarm = 1 THEN 0.0
                            ELSE cs.raw_cusum
                          END
                    )
                END AS raw_cusum,
                CASE
                    WHEN cs.baseline_std > 0
                     AND GREATEST(
                        0.0,
                        (cs.baseline_mean - (0.5 * cs.baseline_std))
                        - cdc.daily_clicks
                        + CASE
                            WHEN cs.drift_alarm = 1 THEN 0.0
                            ELSE cs.raw_cusum
                          END
                     ) > (3.0 * cs.baseline_std)
                    THEN 1
                    ELSE 0
                END AS drift_alarm
            FROM cusum_scan AS cs
            JOIN complete_daily_clicks AS cdc
                ON cs.code_module = cdc.code_module
                AND cs.code_presentation = cdc.code_presentation
                AND cs.id_student = cdc.id_student
                AND cdc.day_index = cs.day_index + 1
        ),
        engagement_drift_features AS (
            SELECT
                code_module,
                code_presentation,
                id_student,
                CASE
                    WHEN MAX(CASE WHEN drift_alarm = 1 THEN day_index ELSE NULL END) IS NULL
                    THEN 100
                    ELSE 99 - MAX(CASE WHEN drift_alarm = 1 THEN day_index ELSE NULL END)
                END AS days_since_last_drift
            FROM cusum_scan
            GROUP BY code_module, code_presentation, id_student
        )
        SELECT
            si.code_module,
            si.code_presentation,
            si.id_student,
            si.gender,
            si.region,
            si.highest_education,
            si.imd_band,
            si.age_band,
            si.disability,
            TRY_CAST(si.num_of_prev_attempts AS INTEGER) AS num_of_prev_attempts,
            TRY_CAST(si.studied_credits AS INTEGER) AS studied_credits,
            COALESCE(vf.total_clicks, 0) AS total_clicks,
            COALESCE(vf.active_days, 0) AS active_days,
            COALESCE(vf.avg_clicks_per_day, 0) AS avg_clicks_per_day,
            COALESCE(af.assessments_submitted, 0) AS assessments_submitted,
            COALESCE(af.avg_score, 0) AS avg_score,
            COALESCE(af.late_submissions, 0) AS late_submissions,
            COALESCE(pf.procrastination_index, 0) AS procrastination_index,
            COALESCE(edf.days_since_last_drift, 100) AS days_since_last_drift,
            si.final_result,
            CASE
                WHEN si.final_result IN ('Withdrawn', 'Fail') THEN 'High'
                WHEN si.final_result = 'Pass' THEN 'Medium'
                WHEN si.final_result = 'Distinction' THEN 'Low'
                ELSE 'Unknown'
            END AS risk
        FROM student_info AS si
        LEFT JOIN vle_features AS vf
            ON si.code_module = vf.code_module
            AND si.code_presentation = vf.code_presentation
            AND si.id_student = vf.id_student
        LEFT JOIN assessment_features AS af
            ON si.code_module = af.code_module
            AND si.code_presentation = af.code_presentation
            AND si.id_student = af.id_student
        LEFT JOIN procrastination_features AS pf
            ON si.id_student = pf.id_student
        LEFT JOIN engagement_drift_features AS edf
            ON si.code_module = edf.code_module
            AND si.code_presentation = edf.code_presentation
            AND si.id_student = edf.id_student
        """
    )


def _import_oulad_tables(con: duckdb.DuckDBPyConnection, oulad_dir: Path) -> None:
    for table_name, file_name in OULAD_TABLES.items():
        csv_path = _sql_path(oulad_dir / file_name)
        con.execute(
            f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT *
            FROM read_csv_auto('{csv_path}', header = true)
            """
        )


def _ensure_student_features_view(con: duckdb.DuckDBPyConnection) -> None:
    expected_column_exists = con.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = 'student_features'
          AND column_name = 'days_since_last_drift'
        """
    ).fetchone()[0]
    if not expected_column_exists:
        create_student_features_view(con)


def _missing_oulad_files(oulad_dir: Path) -> Iterable[Path]:
    return [
        oulad_dir / file_name
        for file_name in OULAD_TABLES.values()
        if not (oulad_dir / file_name).exists()
    ]


def _sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


if __name__ == "__main__":
    connection = initialize_oulad_db()
    connection.close()
    print(f"OULAD DuckDB database initialized at {DB_PATH}")
