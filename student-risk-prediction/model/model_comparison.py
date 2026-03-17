"""Model comparison utilities for student risk classification."""

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def build_candidate_models():
    """Return a dictionary of baseline models to compare."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }


def compare_models(X_train, X_test, y_train, y_test):
    """Train all candidate models and return metrics + fitted models."""
    models = build_candidate_models()
    rows = []
    fitted_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        accuracy = accuracy_score(y_test, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, preds, average="weighted", zero_division=0
        )

        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 score": f1,
            }
        )
        fitted_models[name] = model

    results_df = pd.DataFrame(rows).sort_values(by="Accuracy", ascending=False)
    return results_df, fitted_models
