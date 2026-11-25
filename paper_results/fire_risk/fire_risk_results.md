# Fire-Risk Prediction and Satellite Hotspot Experimental Results

## 1. Objective and target definition

The experiment estimates the probability that at least one non-low-confidence VIIRS satellite hotspot will occur in a 0.5° × 0.625° grid cell on day $t+1$, using information available through day $t$. The positive label is therefore a satellite-detected thermal anomaly, not a confirmed wildfire or a confirmed agricultural-fire event.

The spatial study area is 74.0–77.6°E and 28.4–32.6°N. The grid contains 54 cells. The evaluated calendar period is 2017–2023, with October–November retained as the seasonal post-harvest period of interest.

## 2. Data sources

### NASA FIRMS

NASA Fire Information for Resource Management System (FIRMS) VIIRS/S-NPP active-fire files were obtained for 91 five-day API windows. After removing low-confidence detections and duplicate observations, 445,482 hotspot records were available in the cache used for this analysis. The cache endpoint and source are documented in `reproducibility.md`. FIRMS observations indicate thermal anomalies and should not be interpreted as confirmed fires.

### NASA POWER

NASA POWER daily meteorological records were obtained for 54 grid points. The cached data cover 2017-01-01 through 2023-12-31 and contain temperature, relative humidity, wind speed, and corrected precipitation. The weather values are retrospective observations, not forecasts.

### Dataset matching

FIRMS point detections were assigned to a grid cell using floor-based 0.5° latitude and 0.625° longitude bins. They were counted by cell and date. The resulting counts were left-joined to daily NASA POWER grid-cell observations. The reconstructed dataset contains 23,058 October–November cell-days.

## 3. Experimental design

The temporal split was:

- Training: years 2017–2020: 13,176 rows, 6,984 positive, 6,192 negative.
- Validation: year 2021: 3,294 rows, 1,805 positive, 1,489 negative.
- Test: years 2022–2023: 6,588 rows, 3,264 positive, 3,324 negative.

The test positive prevalence was 49.5446%. Features include observed temperature, relative humidity, wind speed, precipitation, three-day precipitation, dry-day state, today's detection, previous detection counts, day-of-year, latitude, and longitude. The target uses only the following day's detection count. No target-derived value is included in the feature vector.

The test split is chronological. It does not randomize individual grid cells or dates. A spatial holdout was not available, and spatially adjacent cells remain in each calendar-year partition.

## 4. Reproduced model results

The exact existing reference metrics reproduced from `outputs/results_c/cache` are:

| Model | AUPRC | Precision | Recall | False alarms/100 negative cell-days | Missed event rate | Brier |
|---|---:|---:|---:|---:|---:|---:|
| Persistence | 0.740879 | 0.802026 | 0.800551 | 19.4043 | 0.199449 | — |
| Climatology | 0.545201 | 0.510244 | 0.793505 | 74.7894 | 0.206495 | 0.250013 |
| Hot/dry/windy | 0.436928 | 0.501587 | 0.629596 | 61.4320 | 0.370404 | — |
| Logistic regression | 0.853035 | 0.850391 | 0.700061 | 12.0939 | 0.299939 | 0.134565 |
| Gradient-boosted trees | 0.872330 | 0.836778 | 0.731924 | 14.0193 | 0.268076 | 0.128490 |

The gradient-boosted-tree 95% block-bootstrap AUPRC interval is 0.801349–0.913083. The logistic-regression interval is 0.797220–0.893575.

Additional 0.5-threshold evaluations were produced for comparison only:

| Model | AUPRC | ROC-AUC | Precision | Recall | F1 | Brier |
|---|---:|---:|---:|---:|---:|---:|
| Logistic regression | 0.853035 | 0.885248 | 0.793820 | 0.842218 | 0.817303 | 0.134565 |
| Histogram gradient boosting | 0.872330 | 0.895198 | 0.785794 | 0.860907 | 0.821637 | 0.128490 |
| Random forest | 0.884159 | 0.899158 | 0.789904 | 0.858150 | 0.822614 | 0.127026 |
| XGBoost | 0.875662 | 0.896164 | 0.786798 | 0.858150 | 0.820926 | 0.127724 |

The auxiliary random forest and XGBoost runs are additional 0.5-threshold comparisons. They were not used to select the original reported alert threshold and should not replace the existing threshold-calibrated logistic/gradient-boosted metrics as the primary validation result.

## 5. Hotspot analysis

FIRMS cache files contained point observations across 455 unique acquisition dates. The plotted hotspot distribution is a geographic and temporal description of thermal detections in the study region. It does not separately identify crop residue burning, wildfire, urban fire, or another land-use fire source.

October–November monthly counts and the over-time trend are shown in the publication figures. Weather distributions are based on NASA POWER cell-day observations and are not independent field measurements.

## 6. Scientific interpretation

The experiment demonstrates that historical VIIRS detections and observed meteorology can rank next-day regional satellite detections better than persistence, climatology, and a basic hot/dry/windy heuristic. The strongest reproduced reference result is gradient-boosted trees with AUPRC 0.872330 and Brier score 0.128490; logistic regression has AUPRC 0.853035.

These results should be described as regional cell-day next-day detection-risk estimates. They do not establish operational wildfire warning performance, confirmed agricultural-fire prediction, or field-level crop-risk prediction.

## 7. Limitations

- Hotspots are satellite thermal detections, not confirmed fires.
- No cropland mask or agricultural-fire verification label was used.
- Features use observed weather; forecast weather would normally have additional uncertainty.
- NASA POWER is a gridded modelled/observational product and not an in-field sensor network.
- The split uses calendar years and retains spatial correlation across neighboring grid cells.
- There is no independent spatial holdout or field-verified ground truth.
- A 0.5 probability threshold is a comparison convention, not a scientifically validated warning threshold.
- Confidence intervals are block-bootstrap intervals over test weeks, not guaranteed operational uncertainty intervals.
