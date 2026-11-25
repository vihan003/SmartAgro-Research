# Scientific Limitations and Evidence Boundaries

1. Satellite detections are VIIRS active-fire hotspots or thermal anomalies, not confirmed wildfire or crop-fire records.
2. The model does not identify the cause of a detection.
3. No cropland mask or agricultural ownership map is used.
4. The spatial resolution is 0.5° × 0.625° grid cells, not an individual field or farm.
5. NASA POWER values are gridded observed-weather records. They are used as an optimistic stand-in for forecast variables.
6. The target horizon is next-day occurrence within the same regional grid cell.
7. Calendar-year splitting is chronological but does not create an independent spatial holdout. Neighboring cells share environmental and temporal structure.
8. The 80% validation recall threshold is an operating criterion, not an independently validated warning threshold.
9. Block-bootstrap confidence intervals account for weekly test blocks but do not remove spatial or dataset bias.
10. AUPRC and Brier scores do not demonstrate operational safety, timeliness, or cost-benefit performance.
11. No physical fire validation, fire-type classification, burned-area product, or ground-confirmed event dataset was used.
12. Auxiliary Random Forest and XGBoost results should not be claimed as calibrated operational probabilities.
13. The fire-risk experiment should be reported as retrospective satellite-detection-risk ranking for an Indian regional study area, not as a global or real-time wildfire system.
