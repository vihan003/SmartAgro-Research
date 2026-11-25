# Environmental Monitoring – Verified Results Summary

## Datasets

- Predictive irrigation: 960 real rows, 30-minute smart-garden observations, 932 pump-off and 28 pump-on readings.
- Motor-monitor archive: 3,000 real rows, 1,431 pump-off and 1,569 pump-on readings.
- SmartAgro IoT sensor archive: 101 real timestamped readings; no valid crop-stress or pump target.
- USDA Cook Agronomy Farm sensor network: downloaded for documentation; not merged because its feature set differs and it has no air-humidity/light or validated stress label.

## Supervised pump-status models

Both datasets were evaluated with chronological splits. The predictive-irrigation test set contained 142 pump-off and 2 pump-on observations. Logistic regression, random forest, gradient boosting, and XGBoost all reported 1.000 accuracy, macro-F1, ROC-AUC, and PR-AUC on that test slice. These values are reported, but the positive class sample is extremely small and should not be interpreted as general field performance.

The motor-monitor test set contained 208 pump-off and 242 pump-on observations and also reported perfect or near-perfect model metrics. The source CSV does not contain timestamps, so its chronological split is based only on synthetic playback timestamps and should not be treated as real event chronology.

Full metrics are in `environmental_monitoring/results/environmental_model_results.json`; trained models are in `environmental_monitoring/saved_models/`; plots are in `environmental_monitoring/results/`.

## Monitoring and stress detection

The dashboard provides historical-data playback, configurable speed, pause/resume/reset, sensor cards, a trend chart, a stress category, and conservative alerts. Stress categories are rule-generated and not crop-specific diagnoses. No SmartAgro IoT stress target was available.

## Dashboard and APIs

The dashboard is available at `http://127.0.0.1:5050` while `environmental_monitoring/app.py` is running.

Available APIs:
- `/api/datasets`
- `/api/monitor`
- `/api/history`
- `/api/predict`
- `/api/metrics`
- `/api/alerts`
- `/api/stress`

## Verification

- Full project test suite: 9 tests passed.
- Pylance diagnostics: no errors in `environmental_monitoring/`.
- Dashboard and API smoke checks: page, metadata, monitor, history, metrics, alerts, and prediction endpoints returned HTTP 200.

## Limitations

- No physical sensors or live field devices were used.
- Recorded playback does not establish live field validity.
- Dataset classes and sensor ranges may differ across sources.
- Stress thresholds are configurable demonstrations, not clinical or agronomic recommendations.
- The ideal future integration requires paired sensor readings, crop phenotype, stress labels, and physical field validation.
