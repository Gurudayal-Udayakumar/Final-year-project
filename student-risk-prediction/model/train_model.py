"""Train a multiclass LightGBM academic-risk model optimized with Mealpy SCSO."""

from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from mealpy import FloatVar
from mealpy.swarm_based.SCSO import OriginalSCSO
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    load_dataset,
    prepare_features_and_target,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model" / "student_risk_model.joblib"
RANDOM_STATE = 42


def main():
    """Train the optimized LightGBM pipeline and save it to disk."""
    results = train_and_select_best_model()
    print("Best model:", results["best_model_name"])
    print("Best parameters:", results["best_params"])
    print("ROC-AUC:", f"{results['roc_auc']:.4f}")
    save_model(results["best_model"])
    print(f"Model saved to: {MODEL_PATH}")


def save_model(model) -> Path:
    """Persist the trained model to disk and return the path."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return MODEL_PATH


def train_and_select_best_model(random_state: int = RANDOM_STATE):
    """Optimize and train a multiclass LightGBM academic-risk pipeline."""
    df = load_dataset()
    X, y = prepare_features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    best_params, optimization_auc = optimize_lightgbm_with_scso(
        X_train,
        X_test,
        y_train,
        y_test,
        random_state=random_state,
    )

    best_model = build_lgbm_pipeline(best_params, random_state=random_state)
    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)
    high_risk_proba = _high_class_probabilities(best_model, X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    roc_auc = roc_auc_score(y_test, y_proba, labels=list(best_model.classes_), multi_class="ovr", average="weighted")

    comparison_df = pd.DataFrame(
        [
            {
                "Model": "SCSO-LightGBM",
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 score": f1,
                "ROC-AUC": roc_auc,
                "Optimization ROC-AUC": optimization_auc,
                "High Risk ROC-AUC": roc_auc_score((y_test == "High").astype(int), high_risk_proba),
            }
        ]
    )

    return {
        "best_model": best_model,
        "best_model_name": "SCSO-LightGBM",
        "best_params": best_params,
        "comparison_df": comparison_df,
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=list(best_model.classes_)),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
        "classes": list(best_model.classes_),
        "roc_auc": roc_auc,
    }


def optimize_lightgbm_with_scso(
    X_train,
    X_test,
    y_train,
    y_test,
    random_state: int = RANDOM_STATE,
):
    """Use Mealpy's Sand Cat Swarm Optimization to tune LightGBM.

    Mealpy sends a continuous solution array. The first two values are rounded
    to integer hyperparameters because LightGBM expects integer tree shape
    controls. The third value remains a float for learning_rate.
    """

    def objective_function(solution):
        params = _solution_to_lgbm_params(solution)
        pipeline = build_lgbm_pipeline(params, random_state=random_state)
        pipeline.fit(X_train, y_train)

        y_proba = pipeline.predict_proba(X_test)
        auc = roc_auc_score(
            y_test,
            y_proba,
            labels=list(pipeline.classes_),
            multi_class="ovr",
            average="weighted",
        )

        # Mealpy minimizes, so return negative ROC-AUC to maximize ROC-AUC.
        return -auc

    problem = {
        "bounds": FloatVar(
            lb=(20.0, 3.0, 0.01),
            ub=(100.0, 12.0, 0.20),
            name="lgbm_leafwise_params",
        ),
        "minmax": "min",
        "obj_func": objective_function,
    }

    optimizer = OriginalSCSO(epoch=30, pop_size=20)
    best_agent = optimizer.solve(problem)
    best_solution = _extract_best_solution(best_agent, optimizer)
    best_params = _solution_to_lgbm_params(best_solution)
    best_auc = -float(objective_function(best_solution))
    return best_params, best_auc


def _solution_to_lgbm_params(solution) -> dict:
    """Map Mealpy's continuous array to valid LightGBM hyperparameters."""
    # SCSO searches in continuous space, but these two LightGBM settings are
    # integer-valued. Round them, then clamp to the requested search bounds.
    num_leaves = int(round(solution[0]))
    max_depth = int(round(solution[1]))

    # learning_rate is naturally continuous, so keep it as a float and clamp.
    learning_rate = float(solution[2])

    return {
        "num_leaves": max(20, min(100, num_leaves)),
        "max_depth": max(3, min(12, max_depth)),
        "learning_rate": max(0.01, min(0.20, learning_rate)),
    }


def build_lgbm_pipeline(params: dict, random_state: int = RANDOM_STATE) -> Pipeline:
    """Build the full preprocessing + LightGBM sklearn pipeline."""
    classifier = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=250,
        class_weight="balanced",
        random_state=random_state,
        verbosity=-1,
        **params,
    )
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", classifier),
        ]
    )


def build_preprocessor() -> ColumnTransformer:
    """Build preprocessing for OULAD numeric and categorical features."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    preprocessor.set_output(transform="pandas")
    return preprocessor


def _high_class_probabilities(pipeline: Pipeline, X):
    """Return predicted probabilities for the High-risk class."""
    classes = list(pipeline.classes_)
    high_index = classes.index("High")
    return pipeline.predict_proba(X)[:, high_index]


def _extract_best_solution(best_agent, optimizer):
    """Read the best solution across Mealpy versions."""
    if hasattr(best_agent, "solution"):
        return best_agent.solution
    if isinstance(best_agent, (list, tuple)) and len(best_agent) > 0:
        return best_agent[0]
    if hasattr(optimizer, "g_best") and hasattr(optimizer.g_best, "solution"):
        return optimizer.g_best.solution
    raise RuntimeError("Unable to extract the best solution from Mealpy optimizer.")


if __name__ == "__main__":
    main()
