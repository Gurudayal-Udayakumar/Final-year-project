r"""Simple checker that prints the Python executable and verifies plotly and other key packages are importable.

Run with the project's virtualenv python to confirm the environment is correct:
& ".\.venv\Scripts\python.exe" tools\check_plotly.py
"""
import sys

packages = ["plotly", "streamlit", "pandas"]

print("Python executable:", sys.executable)
print("Python version:", sys.version.replace('\n', ' '))

for pkg in packages:
    try:
        m = __import__(pkg)
        ver = getattr(m, "__version__", "<unknown>")
        print(f"{pkg} imported successfully, version={ver}")
    except Exception as e:
        print(f"{pkg} import FAILED: {e}")
