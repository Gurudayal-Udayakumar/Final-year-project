import os
import sys
from pathlib import Path

print('CWD:', os.getcwd())
print('Script:', Path(__file__).resolve())
print('\n-- sys.path --')
for p in sys.path:
    print(p)

print('\n-- try import --')
try:
    from utils.preprocessing import load_dataset, prepare_model_dataframe
    print('IMPORT OK:', load_dataset.__name__, prepare_model_dataframe.__name__)
except Exception as e:
    print('IMPORT FAILED:', type(e).__name__, e)
    import traceback
    traceback.print_exc()
