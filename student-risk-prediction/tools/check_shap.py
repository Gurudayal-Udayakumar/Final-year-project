"""Check SHAP compatibility for the saved sklearn pipeline."""

from pathlib import Path
import sys
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import shap

from model.train_model import MODEL_PATH
from utils.preprocessing import load_dataset, prepare_features_and_target


print("shap version", shap.__version__)
print("Loading model from", MODEL_PATH)
pipeline = joblib.load(MODEL_PATH)
print("PIPELINE TYPE:", type(pipeline))
print("PIPELINE CLASSES:", getattr(pipeline, "classes_", None))

df = load_dataset()
X, _ = prepare_features_and_target(df)
sample = X.sample(min(5, len(X)), random_state=1)
print("RAW SAMPLE SHAPE:", sample.shape)

try:
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    transformed = preprocessor.transform(sample)
    print("TRANSFORMED SAMPLE SHAPE:", transformed.shape)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(transformed)
    print("shap_values type:", type(shap_values))
    if isinstance(shap_values, (list, tuple)):
        print("len(shap_values):", len(shap_values))
        print("first array shape:", shap_values[0].shape)
    else:
        print("shap_values array shape:", getattr(shap_values, "shape", None))
except Exception:
    traceback.print_exc()
    sys.exit(1)

print("SHAP check completed successfully")
