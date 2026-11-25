# SmartAgro Environmental Monitoring Module

This module uses public environmental sensor datasets for recorded-data playback, rule-based stress detection, irrigation pump-status prediction, and conservative recommendations. It is not connected to physical sensors and must not be presented as live field monitoring.

## Run

```powershell
cd D:\agro2
.\.venv\Scripts\python.exe environmental_monitoring\preprocess.py
.\.venv\Scripts\python.exe environmental_monitoring\train_models.py
.\.venv\Scripts\python.exe environmental_monitoring\app.py
```

Open http://127.0.0.1:5050.

## Real datasets

1. **Predictive Irrigation Sensor Dataset** – Mendeley Data, CC BY 4.0, 960 rows, 30-minute smart-garden observations, with a pump-status target.
   - Source: https://data.mendeley.com/datasets/jtsyb593fy/1
2. **Soil Moisture, Air Temperature, Humidity and Motor on/off Monitoring** – Mendeley Data, CC BY-NC 3.0, 3,000 rows, with a pump-status target.
   - Source: https://data.mendeley.com/datasets/fpdwmm7nrb/1
3. **Smart Agriculture and Plant Health Monitoring using IoT** – Mendeley Data, CC BY 4.0, 101 readings, temperature, humidity, light, soil moisture, pH, and soil temperature; no validated stress or pump label.
   - Source: https://data.mendeley.com/datasets/65jxyrxv7b/1
4. **USDA/ARS Cook Agronomy Farm Soil Sensor Network** – public-domain resource, 42 locations and soil moisture/soil temperature at five depths. Downloaded and documented, but not merged into the dashboard because it lacks compatible air-temperature, humidity, and light fields.
   - Source: https://catalog.data.gov/dataset/data-from-a-field-scale-sensor-network-data-set-for-monitoring-and-modeling-the-spatial-an

Metadata is in `environmental_monitoring/datasets/dataset_metadata.json`.

## Method

- Missing numeric readings are linearly interpolated only after parsing; missing pump values use the mode.
- Duplicate rows are removed.
- The predictive-irrigation and motor-monitor datasets are kept separate; they are not merged.
- Pump-status models use chronological test splits. The motor dataset has no original timestamps, so synthetic timestamps are playback chronology only and do not represent real event timing.
- Stress categories are configurable rule-generated labels, not diagnosed crop stress.
- A pump-status prediction is a dataset-derived ML output, not a field irrigation prescription.

## Caution about high metric values

The predictive-irrigation test set contains only 2 pump-on examples, while the motor-monitor test set contains 242 pump-on examples out of 450. Perfect or near-perfect scores must therefore be interpreted cautiously because of class structure and possible deterministic relationships. No claim of physical field accuracy is made.

## Outputs

- Processed CSVs: `data/environmental/processed/`
- Models: `environmental_monitoring/saved_models/`
- Metrics and plots: `environmental_monitoring/results/`
- Dashboard: `environmental_monitoring/templates/index.html`
- APIs: `/api/datasets`, `/api/monitor`, `/api/history`, `/api/predict`, `/api/metrics`, `/api/alerts`, `/api/stress`
