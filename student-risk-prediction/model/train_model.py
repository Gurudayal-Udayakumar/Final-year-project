"""Train and save the best model for student burnout/dropout risk prediction."""

from pathlib import Path
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from model.model_comparison import compare_models
from utils.preprocessing import load_dataset, prepare_features_and_target

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model" / "student_risk_model.joblib"


def train_and_select_best_model(test_size: float = 0.2, random_state: int = 42):
    """Train candidate models, select best by accuracy, and return training artifacts."""
    # 1) Load data
    df = load_dataset()

    # 2) Build X/y using the shared preprocessing pipeline
    X, y = prepare_features_and_target(df)

    # 3) Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 4) Compare models and pick best
    comparison_df, fitted_models = compare_models(X_train, X_test, y_train, y_test)
    best_model_name = comparison_df.iloc[0]["Model"]
    best_model = fitted_models[best_model_name]

    # 5) Evaluate the selected model
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=["Low", "Medium", "High"])
    report = classification_report(y_test, y_pred)

    return {
        "comparison_df": comparison_df,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "confusion_matrix": cm,
        "classification_report": report,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def save_model(model, model_path: Path = MODEL_PATH):
    """Save trained model to disk using joblib."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def main():
    """CLI entrypoint: python -m model.train_model"""
    results = train_and_select_best_model()
    save_model(results["best_model"])

    print("Model Comparison:")
    print(results["comparison_df"].to_string(index=False))
    print(f"\nBest model: {results['best_model_name']}")
    print("\nClassification Report:\n")
    print(results["classification_report"])
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
