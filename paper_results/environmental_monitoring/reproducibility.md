# Reproducibility

## Environment

Use the project virtual environment:

```powershell
cd D:\agro2
.\.venv\Scripts\python.exe
```

## Reproduce preprocessing

```powershell
.\.venv\Scripts\python.exe environmental_monitoring\preprocess.py
```

## Reproduce original model evaluation

```powershell
.\.venv\Scripts\python.exe environmental_monitoring\train_models.py
```

## Reproduce this paper-analysis package

```powershell
.\.venv\Scripts\python.exe paper_results\build_environmental_paper_results.py
```

## Model audit

```powershell
.\\.venv\\Scripts\\python.exe paper_results\\audit_environmental_models.py
.\\.venv\\Scripts\\python.exe paper_results\\run_environmental_cv_audit.py
```

These commands verify feature-pattern overlap, class distributions, timestamp provenance, and five-fold TimeSeriesSplit metrics. The resulting CSV files are `model_audit_summary.csv` and `temporal_cv_summary.csv`.

## Inputs

- `data/environmental/processed/predictive_irrigation.csv`
- `data/environmental/processed/motor_monitor.csv`
- `data/environmental/processed/smartagro_iot.csv`
- `environmental_monitoring/results/*_metrics.json`
- `environmental_monitoring/datasets/dataset_metadata.json`

## Procedures

- Processed numerical missing values are excluded or interpolated as documented in `preprocess.py`.
- Stress categories use `environmental_monitoring/stress_detection.py` with its default configurable thresholds.
- The original model evaluation uses a 70%/15%/15% chronological-style split for both datasets. The motor-monitor timestamps are synthetic and not real chronology.
- All figures are regenerated at 300 DPI PNG and PDF.
