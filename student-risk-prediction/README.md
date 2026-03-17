# Student Burnout and Dropout Risk Prediction Platform

A beginner-friendly **Machine Learning analytics platform** that predicts student burnout/dropout risk as:
- Low
- Medium
- High

This upgraded version includes:
- data exploration
- feature engineering
- model training + model comparison
- interactive risk prediction
- analytics dashboard
- explainable AI (SHAP)

---

## Project Structure

```text
student-risk-prediction/
│
├── app/
│   ├── main.py
│   ├── dashboard.py
│   └── pages/
│       ├── 1_Dataset_Explorer.py
│       ├── 2_Model_Training.py
│       ├── 3_Risk_Prediction.py
│       ├── 4_Analytics_Dashboard.py
│       └── 5_Model_Explainability.py
│
├── data/
│   └── students.csv
│
├── model/
│   ├── model_comparison.py
│   ├── train_model.py
│   └── student_risk_model.joblib   # generated after training
│
├── utils/
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   └── visualization.py
│
└── requirements.txt
```

---

## System Architecture (Simple View)

1. **Dataset Layer**
   - Loads synthetic student activity data from `data/students.csv`.

2. **Feature Engineering Layer**
   - Creates:
     - `engagement_score = (attendance + login_frequency + assignments_submitted) / 3`
     - `performance_index = attendance * 0.4 + avg_grade * 0.6`

3. **ML Layer**
   - Trains and compares 3 models:
     - Logistic Regression
     - Random Forest
     - Gradient Boosting
   - Selects best model by accuracy and saves it with Joblib.

4. **Presentation Layer (Streamlit)**
   - Multi-page UI with dedicated pages for exploration, training, prediction, analytics, and explainability.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Train the Model

```bash
python -m model.train_model
```

This command will:
- load data
- engineer features
- train and compare multiple models
- choose the best model
- print evaluation metrics
- save model to `model/student_risk_model.joblib`

---

## Run the Streamlit App

```bash
python -m streamlit run app/main.py
```

Then open the local URL shown in terminal.

---

## App Pages

- **Home**: intro + navigation help
- **Dataset Explorer**: preview, stats, missing values, dtypes, histograms, correlation heatmap
- **Model Training**: one-click training, progress bar, model comparison, confusion matrix, report
- **Risk Prediction**: slider inputs, predicted risk, probabilities
- **Analytics Dashboard**: pie chart, boxplot, scatter plot, feature importance, colored risk table
- **Explainable AI**: SHAP summary plot + explanation text

---

## Notes

- This project is designed for educational demonstrations, not production use.
- Code is modular and heavily commented for easy explanation in viva/demo.
