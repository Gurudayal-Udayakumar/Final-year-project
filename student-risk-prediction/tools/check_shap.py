import joblib
import sys
import traceback
import pandas as pd
import shap
from utils.preprocessing import load_dataset, prepare_features_and_target

print('shap version', shap.__version__)
model_path = 'model/student_risk_model.joblib'
print('Loading model from', model_path)
model = joblib.load(model_path)
print('MODEL TYPE:', type(model))
print('HAS feature_names_in_:', hasattr(model,'feature_names_in_'))
print('feature_names_in_ ->', getattr(model,'feature_names_in_', None))

df = load_dataset()
X, _ = prepare_features_and_target(df)
print('DATA X cols ->', X.columns.tolist())

sample = X.sample(min(5, len(X)), random_state=1)
print('SAMPLE SHAPE', sample.shape)
print('MODEL ATTRS EXCERPT:', [a for a in dir(model) if a in ('predict_proba','feature_importances_','coef_','classes_','feature_names_in_')])

try:
    if hasattr(model, 'feature_importances_'):
        print('Using TreeExplainer')
        expl = shap.TreeExplainer(model)
    elif hasattr(model, 'coef_'):
        print('Using LinearExplainer')
        expl = shap.LinearExplainer(model, sample, feature_perturbation='interventional')
    else:
        print('Using KernelExplainer (this may be slow)')
        expl = shap.KernelExplainer(lambda x: model.predict_proba(pd.DataFrame(x, columns=sample.columns)), sample.sample(min(20, len(sample)), random_state=1))

    sv = expl.shap_values(sample)
    print('shap_values type:', type(sv))
    if isinstance(sv, (list, tuple)):
        print('len(shap_values)=', len(sv))
        print('first array shape', sv[0].shape)
    else:
        import numpy as np
        print('shap_values array shape', getattr(sv, 'shape', None))
except Exception:
    traceback.print_exc()
    sys.exit(1)

print('SHAP check completed successfully')
