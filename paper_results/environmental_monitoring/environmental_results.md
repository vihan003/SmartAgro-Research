# Experimental Results: Environmental Monitoring and Crop-Stress Detection

## 3.1 Data description

Three public datasets were used. The predictive-irrigation dataset contained 960 timestamped observations collected at 30-minute intervals over 20 days. It included air temperature, relative humidity, soil moisture, light intensity, and a binary pump-status label. The motor-monitor dataset contained 3,000 source records with soil moisture, air temperature, air humidity, and pump-status information. Its original source CSV did not provide timestamps; timestamps were generated only for playback and split construction. The SmartAgro IoT sensor archive contained 101 matched timestamped observations for air temperature, relative humidity, light intensity, soil moisture, soil pH, and soil temperature, but no stress or pump-status ground-truth label.

No datasets were merged because their variable semantics and collection environments differed. The USDA/ARS Cook Agronomy Farm archive was downloaded and documented but was not included in the current monitoring or supervised analysis because it did not provide compatible air temperature, humidity, light, or validated stress labels.

## 3.2 Recorded sensor statistics

In the SmartAgro IoT archive, air temperature ranged from 26.10 to 26.90 °C (mean 26.35 °C), relative humidity ranged from 95.00 to 95.00% (mean 95.00%), soil moisture ranged from 0.91 to 64.22% (mean 60.96%), and soil pH ranged from 6.10 to 6.10 (mean 6.10). The dashboard playback is an interpretation of these historical records, not live sensor acquisition.

## 3.3 Rule-based environmental stress analysis

The existing configurable rule system was applied to the SmartAgro IoT records. The resulting categories were: High stress: 4; Low stress: 97. These classes are rule-generated labels and must not be interpreted as diagnosed crop stress or validated agronomic outcomes.

## 3.4 Machine-learning evaluation

For pump-status prediction, the predictive-irrigation dataset contained 932 pump-off and 28 pump-on observations. A chronological 70%/15%/15% split produced 672 training, 144 validation, and 144 test observations. The test set contained 142 pump-off and only 2 pump-on observations. All existing models returned perfect point estimates on this test slice, but the positive class support is insufficient for a statistically meaningful claim. Precision, recall, ROC-AUC, PR-AUC, and confidence in generalization are therefore not reported as reliable field performance.

The motor-monitor dataset contained 1,431 pump-off and 1,569 pump-on observations. Chronological-style partitioning produced 2,100 training, 450 validation, and 450 test observations. The test set contained 208 pump-off and 242 pump-on observations, and the existing models showed perfect or near-perfect test results. This is an exploratory in-dataset result only: the source lacks original timestamps, the timestamps are synthetic for playback, and label validity/field conditions are not established. It should not be presented as a validated irrigation or crop-stress model.

## 3.5 Scientific interpretation and limitations

The implementation demonstrates preprocessing, historical playback, interpretable rule triggering, and pump-status model execution on public datasets. It does not demonstrate live IoT deployment, physical sensor communication, crop-specific stress diagnosis, or field-level agricultural effectiveness. Rule thresholds are configurable and crop/soil-dependent; they require agronomic and field validation. The SmartAgro archive has no valid crop-stress target, while the pump-status datasets provide irrigation motor labels rather than crop-stress ground truth.

## 3.6 Corrected temporal cross-validation audit

A five-fold `TimeSeriesSplit` audit was also performed from the processed data. The predictive-irrigation dataset had 4, 4, 6, 4, and 4 positive pump-on observations in successive fold test windows. The random forest achieved mean accuracy, macro-F1, and PR-AUC of 1.000 across these folds. However, this result is not reliable field evidence because the positive support remains small and the reported point estimates still test the same 20-day smart-garden testbed. The logistic regression was less perfect (accuracy 0.9725 ± 0.0344; macro-F1 0.8555 ± 0.1210; PR-AUC 1.0000 ± 0.0000), while gradient boosting and XGBoost also varied across folds. The exact fold results are available in `temporal_cv_summary.csv`.

For the motor-monitor dataset, five chronological-style folds with synthetic timestamps yielded 257, 267, 246, 258, and 269 positive observations. The random forest and gradient-boosting models achieved approximately 1.000 accuracy and PR-AUC across folds, while logistic regression achieved mean accuracy 0.9932 ± 0.0077 and PR-AUC 0.999976 ± 0.000046. These values are not evidence of field validity because the timestamps are synthetic, no field ground-truth provenance was verified, and soil moisture shows strong class separation (pump-off mean 830.56 versus pump-on mean 509.07 in the current processed archive).

## 3.7 Recorded-data anomalies

The SmartAgro IoT archive contains an anomalously low soil-moisture reading of 0.91% relative to the remaining range of approximately 61.58–64.22%. This may represent a sensor anomaly or a transient value and was retained rather than silently deleted. The archive has constant relative humidity (95.00%) and soil pH (6.10), so these values have no variability and should not be used to infer stress patterns. Temperature varied only from 26.10 to 26.90 °C, and soil temperature varied only from 18.10 to 18.20 °C.

## Research-ready subsection

Environmental monitoring was evaluated using publicly available, timestamped or timestamp-reconstructable sensor records. The SmartAgro IoT archive was used for recorded playback and rule-based stress analysis. In 101 matched readings, air temperature ranged from 26.10–26.90 °C (mean 26.35 °C), relative humidity ranged from 95.00–95.00% (mean 95.00%), soil moisture ranged from 0.91–64.22% (mean 60.96%), and soil pH ranged from 6.10–6.10 (mean 6.10). The existing rule engine produced High stress: 4; Low stress: 97. These categories were rule-generated and were not validated as crop-stress diagnoses.

Pump-status classification was assessed separately on two datasets. On the predictive-irrigation dataset, the chronological test set contained only 2 pump-on observations among 144 test readings, so reported perfect scores were excluded from research claims. On the motor-monitor dataset, the test set contained 208 pump-off and 242 pump-on observations; perfect in-dataset metrics were retained only as exploratory, because original timestamps and field ground-truth validity were unavailable. Accordingly, this work establishes a reproducible software pipeline and recorded-data analysis, but not a clinically or agronomically validated stress-monitoring system.
