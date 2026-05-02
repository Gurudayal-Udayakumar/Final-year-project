# Data Setup

This project uses a synthetic CSV dataset that is already included in the
repository.

## Default Dataset

```text
data/students.csv
```

The Streamlit app reads this file by default, so a new user does not need to
download any external dataset before opening the app.

## Regenerate the Dataset

If you want to recreate the synthetic data file, run this from
`student-risk-prediction/`:

```powershell
.\.venv\Scripts\python.exe data\data_generator.py
```

This script creates or overwrites:

```text
data/students.csv
```

## What the File Contains

The generated dataset includes:

- student identifiers
- engagement and activity metrics
- assessment performance fields
- demographic and course-related categorical fields
- synthetic `final_result` and `risk` labels

## Recommended Workflow

1. Use the included `data/students.csv` or regenerate it.
2. Launch the app with `.\run_app.ps1`.
3. Train the model from the **Model Training** page or from the command line.

## Optional: Train From the Command Line

After the dataset is ready, you can train the model with:

```powershell
.\.venv\Scripts\python.exe -m model.train_model
```

This creates:

```text
model/student_risk_model.joblib
```
