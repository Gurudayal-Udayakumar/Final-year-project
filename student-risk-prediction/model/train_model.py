"""Train a simple RandomForest model for student burnout/dropout risk."""

from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from utils.preprocessing import load_dataset, prepare_features_and_target


# Path where the trained model file will be saved.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model" / "student_risk_model.joblib"


def main():
    """Main training workflow: load data, train model, evaluate, save model."""
    results = train_and_select_best_model()
    print("Best model:", results["best_model_name"]) 
    # Save the best model
    save_model(results["best_model"])
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()


def save_model(model) -> Path:
    """Persist the trained model to disk and return the path."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return MODEL_PATH


def train_and_select_best_model(random_state: int = 42):
    """Train several candidate models and return comparison results.

    Returns a dict with keys:
    - best_model: trained sklearn estimator
    - best_model_name: readable name
    - comparison_df: pandas DataFrame with metrics per model
    - confusion_matrix: numpy array for best model
    - classification_report: str
    """
    import pandas as pd

    # Load and prepare data
    df = load_dataset()
    X, y = prepare_features_and_target(df)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Candidate models
    candidates = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "LogisticRegression": LogisticRegression(max_iter=500, random_state=random_state),
        "GradientBoosting": GradientBoostingClassifier(random_state=random_state),
    }

    records = []
    trained_models = {}

    for name, model in candidates.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)

            records.append({
                "Model": name,
                "Accuracy": acc,
                "Precision": precision,
                "Recall": recall,
                "F1 score": f1,
            })
            trained_models[name] = model
        except Exception as e:
            records.append({
                "Model": name,
                "Accuracy": 0.0,
                "Precision": 0.0,
                "Recall": 0.0,
                "F1 score": 0.0,
            })

    comparison_df = pd.DataFrame(records)
    # Choose best by Accuracy (tie-break by F1)
    comparison_df = comparison_df.sort_values(by=["Accuracy", "F1 score"], ascending=False).reset_index(drop=True)
    best_row = comparison_df.iloc[0]
    best_name = best_row["Model"]
    best_model = trained_models[best_name]

    # Compute confusion matrix and classification report for best model
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    cr = classification_report(y_test, y_pred_best)

    return {
        "best_model": best_model,
        "best_model_name": best_name,
        "comparison_df": comparison_df,
        "confusion_matrix": cm,
        "classification_report": cr,
    }
