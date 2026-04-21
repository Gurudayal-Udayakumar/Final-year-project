# Data Setup

This project expects the OULAD CSV files in:

```text
data/oulad/
```

Required files:

```text
assessments.csv
courses.csv
studentAssessment.csv
studentInfo.csv
studentRegistration.csv
studentVle.csv
vle.csv
```

The raw OULAD folder and generated DuckDB database are ignored by Git because
the files are large. After placing the CSV files, build the local DuckDB database:

```powershell
.\.venv\Scripts\python.exe -m utils.duckdb_loader
```

This creates:

```text
data/student_risk.duckdb
```
