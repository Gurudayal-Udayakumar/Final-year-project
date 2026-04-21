# Student Academic Risk Decision Support Platform

Streamlit decision-support tool for identifying students at academic risk using
the Open University Learning Analytics Dataset (OULAD).

The current implementation uses:

- DuckDB for the OULAD data layer
- SQL feature views for engagement, assessments, procrastination, and drift
- scikit-learn pipelines for preprocessing
- LightGBM for binary risk classification
- Mealpy SCSO for LightGBM hyperparameter optimization
- SHAP for individualized advisor explanations

## Quick Start

From `student-risk-prediction/`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Place the OULAD CSV files in:

```text
data/oulad/
```

Build the local DuckDB database:

```powershell
.\.venv\Scripts\python.exe -m utils.duckdb_loader
```

Train the model:

```powershell
.\.venv\Scripts\python.exe -m model.train_model
```

Run the app:

```powershell
.\run_app.ps1
```

or:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/main.py
```

## Required OULAD Files

```text
data/oulad/assessments.csv
data/oulad/courses.csv
data/oulad/studentAssessment.csv
data/oulad/studentInfo.csv
data/oulad/studentRegistration.csv
data/oulad/studentVle.csv
data/oulad/vle.csv
```

The OULAD folder and generated `data/student_risk.duckdb` are ignored by Git
because they are large local data artifacts.

## Project Layout

```text
app/
  main.py                         Streamlit entry point
  pages/                          Multipage app views
data/
  README.md                       Data placement instructions
model/
  train_model.py                  LightGBM + Mealpy SCSO training
utils/
  duckdb_loader.py                OULAD import and DuckDB feature view
  preprocessing.py                Feature lists and model-ready data loading
  feature_engineering.py          Procrastination Index helper
  advanced_features.py            CUSUM engagement drift helper
  visualization.py                Plotting helpers
tools/
  check_shap.py                   SHAP/model diagnostic script
  check_plotly.py                 Environment diagnostic script
  verify_utils_import.py          Import diagnostic script
```

## App Pages

- **Dataset Explorer**: OULAD preview, missing values, distributions, feature correlations
- **Model Training**: trains the SCSO-optimized LightGBM pipeline
- **Risk Prediction**: manual advisor-facing prediction form
- **Analytics Dashboard**: risk distributions, behavior plots, feature importance
- **Model Explainability**: high-risk student selector with individualized SHAP force plot

## Feature Engineering

The DuckDB `student_features` view includes:

- VLE engagement totals and active days
- assessment submission counts and scores
- late submission counts
- Procrastination Index
- CUSUM-based days since last negative engagement drift
- demographic/course categorical variables from OULAD

The model target is binary:

```text
High      = OULAD final_result is Fail or Withdrawn
Not High  = OULAD final_result is Pass or Distinction
```

## Notes

- This is an educational decision-support prototype, not a production advising system.
- Model artifacts are generated locally and ignored by Git.
- Raw OULAD files are not committed because of size limits.
