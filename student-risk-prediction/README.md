# Student Academic Risk Decision Support Platform

Streamlit decision-support tool for identifying students at academic risk using
a synthetic student analytics dataset.

The current implementation uses:

- a synthetic CSV dataset with app-ready features
- scikit-learn pipelines for preprocessing
- LightGBM for multiclass risk classification
- Mealpy SCSO for LightGBM hyperparameter optimization
- SHAP for individualized advisor explanations

## What Is Included

- `data/students.csv` for immediate app usage
- Streamlit pages for exploration, training, prediction, analytics, and explainability
- a PowerShell launcher script in `run_app.ps1`
- a synthetic dataset generator in `data/data_generator.py`

## Quick Start For a New User

From `student-risk-prediction/` in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\run_app.ps1
```

Then open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

## End-to-End Run Order

If you want the smoothest first run, use this order:

1. Create the virtual environment.
2. Install dependencies.
3. Launch the app with `.\run_app.ps1`.
4. Open **Dataset Explorer** to confirm `data/students.csv` loads correctly.
5. Open **Model Training** and click **Train Model** to create the latest model artifact.
6. After training finishes, use **Risk Prediction**, **Analytics Dashboard**, and **Model Explainability**.

## One-Command Launch After Setup

After the virtual environment and dependencies are already in place, you only need:

```powershell
.\run_app.ps1
```

You can also install requirements during launch if the virtual environment already
exists:

```powershell
.\run_app.ps1 -InstallRequirements
```

## Data

The app reads this file by default:

```text
data/students.csv
```

That file is already included in the repository, so no external dataset download
is required to open the app.

If you want to regenerate the synthetic dataset:

```powershell
.\.venv\Scripts\python.exe data\data_generator.py
```

This overwrites or recreates:

```text
data/students.csv
```

See [data/README.md](./data/README.md) for more detail.

## Model Training

The app saves the trained model here:

```text
model/student_risk_model.joblib
```

You can train either:

- from the **Model Training** page inside the app, or
- from the command line:

```powershell
.\.venv\Scripts\python.exe -m model.train_model
```

Training the model is recommended after cloning the repository so prediction and
explainability pages use a fresh local artifact.

## Project Layout

```text
app/
  main.py                         Streamlit entry point
  pages/                          Multipage app views
data/
  README.md                       Synthetic data instructions
  data_generator.py               Synthetic dataset generator
  students.csv                    Default app dataset
model/
  train_model.py                  LightGBM + Mealpy SCSO training
utils/
  preprocessing.py                Feature lists and CSV loading helpers
  feature_engineering.py          Procrastination Index helper
  advanced_features.py            CUSUM engagement drift helper
  visualization.py                Plotting helpers
tools/
  check_shap.py                   SHAP/model diagnostic script
  check_plotly.py                 Environment diagnostic script
  verify_utils_import.py          Import diagnostic script
```

## App Pages

- **Dataset Explorer**: synthetic dataset preview, missing values, distributions, feature correlations
- **Model Training**: trains and saves the SCSO-optimized LightGBM pipeline
- **Risk Prediction**: manual advisor-facing prediction form
- **Analytics Dashboard**: risk distributions, behavior plots, feature importance
- **Model Explainability**: student-level SHAP explanations for model output

## Feature Engineering

The dataset includes:

- VLE engagement totals and active days
- assessment submission counts, scores, and late submissions
- Procrastination Index
- CUSUM-style days since last negative engagement drift
- demographic and course categorical variables
- synthetic final result and risk labels

The model target is multiclass:

```text
High    = final_result is Fail or Withdrawn
Medium  = final_result is Pass
Low     = final_result is Distinction
```

## Troubleshooting

- If `.\run_app.ps1` says the virtualenv Python was not found, create the venv first with
  `python -m venv .venv`.
- If dependencies are missing, rerun:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

- If prediction or explainability pages complain about a missing model, train one from
  the **Model Training** page or run:

```powershell
.\.venv\Scripts\python.exe -m model.train_model
```

- If `data/students.csv` is missing or corrupted, regenerate it with:

```powershell
.\.venv\Scripts\python.exe data\data_generator.py
```

## Notes

- This is an educational decision-support prototype, not a production advising system.
- Model artifacts are generated locally.
- The included dataset is synthetic and intended for demonstration and testing.
