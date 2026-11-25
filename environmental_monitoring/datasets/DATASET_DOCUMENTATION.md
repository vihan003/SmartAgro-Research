# Environmental Dataset Documentation

Download date: 20 November 2026.

## 1. Predictive Irrigation Sensor Dataset

- Source: https://data.mendeley.com/datasets/jtsyb593fy/1
- DOI: 10.17632/jtsyb593fy.1
- License: CC BY 4.0
- Downloaded archive: `data/environmental/raw/predictive_irrigation.zip`
- Extracted source: `data/environmental/extracted/predictive_irrigation/.../Edge_IoT_Predictive_Irrigation_Dataset.xlsx`
- Processed file: `data/environmental/processed/predictive_irrigation.csv`
- Records: 960
- Columns after preprocessing: 14
- Features: temperature, humidity, soil moisture, light, hour, day of week, lagged moisture changes, and rolling temperature/humidity
- Target: `pump_status` (0/1)
- Target counts: 932 off, 28 on
- Missing values after preprocessing: 0%
- Limitation: 20-day smart-garden testbed; no crop or soil identity; not field-validated.

## 2. Soil Moisture, Air Temperature, Humidity and Motor Monitoring

- Source: https://data.mendeley.com/datasets/fpdwmm7nrb/1
- DOI: 10.17632/fpdwmm7nrb.1
- License: CC BY-NC 3.0
- Downloaded archive: `data/environmental/raw/motor_monitor.zip`
- Processed file: `data/environmental/processed/motor_monitor.csv`
- Records: 3,000
- Columns after preprocessing: 6
- Features: soil-moisture reading, air temperature, humidity
- Target: `pump_status` (0/1)
- Target counts: 1,431 off, 1,569 on
- Missing values after preprocessing: 0%
- Limitations: source has no timestamps; synthetic timestamps are playback-only; no crop/soil identity; NC license is restrictive.

## 3. Smart Agriculture and Plant Health Monitoring using IoT

- Source: https://data.mendeley.com/datasets/65jxyrxv7b/1
- DOI: 10.17632/65jxyrxv7b.1
- License: CC BY 4.0
- Downloaded archive: `data/environmental/raw/smartagro_iot.zip`
- Processed file: `data/environmental/processed/smartagro_iot.csv`
- Records: 101 timestamped readings
- Columns after preprocessing: 7
- Features: temperature, humidity, light, soil moisture, soil pH, soil temperature
- Valid target: none
- Missing values after preprocessing: 0% after joining available timestamped files
- Limitation: historical testbed/archive; no paired crop stress ground truth; used only for monitoring and rule demonstrations.

## 4. USDA/ARS Cook Agronomy Farm Soil Sensor Network

- Source: https://catalog.data.gov/dataset/data-from-a-field-scale-sensor-network-data-set-for-monitoring-and-modeling-the-spatial-an
- Resource: https://ndownloader.figshare.com/files/44335805
- License: public domain/USA.gov public-domain label as reported by Data.gov
- Downloaded archive: `data/environmental/raw/usda_caf.zip`
- Extracted data: `data/environmental/extracted/usda_caf/caf_sensors/`
- Features: location, date/time, volumetric water content and soil temperature at five depths
- Target: none
- Limitation: no air temperature, humidity, light, or validated stress labels. Not merged with the other datasets because of incompatible feature semantics. Archived for future soil-moisture and soil-temperature research.

## Integrity statement

The downloaded datasets are real public datasets. Playback is explicitly labeled as recorded/simulated. Rule-generated stress labels are exploratory. Pump-status metrics evaluate a supervised target; they are not evidence of real-world field efficacy. No physical sensor or live field readings were used.
