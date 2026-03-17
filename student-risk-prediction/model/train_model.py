"""Train a simple RandomForest model for student burnout/dropout risk."""

from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from utils.preprocessing import load_dataset, prepare_features_and_target


# Path where the trained model file will be saved.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model" / "student_risk_model.joblib"


def main():
    """Main training workflow: load data, train model, evaluate, save model."""
    # 1) Load data from CSV
    df = load_dataset()

    # 2) Prepare feature matrix (X) and labels (y)
    X, y = prepare_features_and_target(df)

    # 3) Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4) Train a RandomForest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 5) Evaluate model on test data
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Model Accuracy:", round(accuracy * 100, 2), "%")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # 6) Save trained model so dashboard can use it later
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
