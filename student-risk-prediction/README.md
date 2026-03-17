# Student Burnout and Dropout Risk Prediction System

This is a **simple ML demo prototype** for predicting student burnout/dropout risk as:
- Low
- Medium
- High

It uses a synthetic dataset and a `RandomForestClassifier` model.

## Project Structure

```text
student-risk-prediction/
├── data/
│   └── students.csv
├── model/
│   ├── train_model.py
│   └── student_risk_model.joblib   # created after training
├── app/
│   └── dashboard.py
├── utils/
│   └── preprocessing.py
├── requirements.txt
└── README.md
```

## 1) Install Dependencies

```bash
pip install -r requirements.txt
```

## 2) Train the Model

```bash
python model/train_model.py
```

This command will:
- Load the dataset from `data/students.csv`
- Split data into train/test sets
- Train a `RandomForestClassifier`
- Print accuracy and classification report
- Save the trained model to `model/student_risk_model.joblib`

## 3) Run the Streamlit Dashboard

```bash
streamlit run app/dashboard.py
```

The app will open in your browser and let you:
- Enter attendance
- Enter assignments submitted
- Enter login frequency
- Enter average grade
- Click **Predict Risk** to get predicted risk level
- View a bar chart of risk distribution in dataset

## Notes for Demo Presentation

- Dataset is synthetic and small (~100 rows), made only for prototype demonstration.
- The model and dashboard are intentionally simple and beginner-friendly.
- You can explain each file quickly:
  - `utils/preprocessing.py`: loads and prepares data
  - `model/train_model.py`: trains/evaluates/saves the model
  - `app/dashboard.py`: UI and predictions
