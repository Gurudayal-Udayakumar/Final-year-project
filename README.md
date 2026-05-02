# Final Year Project

This repository contains the **Student Academic Risk Decision Support Platform**,
a Streamlit app for exploring a synthetic student dataset, training a multiclass
LightGBM model, predicting academic risk, and reviewing model explainability.

The runnable application lives in `student-risk-prediction/`.

## Quick Start

1. Clone or download this repository.
2. Open PowerShell in:

```text
student-risk-prediction/
```

3. Create a virtual environment:

```powershell
python -m venv .venv
```

4. Install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

5. Start the Streamlit app:

```powershell
.\run_app.ps1
```

6. Open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## First Run Notes

- The repository already includes `data/students.csv`, so the app can be opened
  immediately after dependencies are installed.
- To use the **Risk Prediction** and **Model Explainability** pages reliably,
  train a model from the **Model Training** page after launching the app.
- The app defaults to a light theme.

## More Detail

For full setup, data, and training instructions, see:

- [student-risk-prediction/README.md](./student-risk-prediction/README.md)
- [student-risk-prediction/data/README.md](./student-risk-prediction/data/README.md)
