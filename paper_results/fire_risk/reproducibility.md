# Fire-Risk Reproducibility

## Real datasets

The experiment uses files already cached in `outputs/results_c/cache/`:

- 91 `firms_YYYYMMDD.csv` NASA FIRMS VIIRS/S-NPP hotspot windows
- 54 `power_lat_lon.json` NASA POWER daily weather point files

Source information:

- NASA FIRMS: https://firms.modaps.eosdis.nasa.gov/
- NASA POWER: https://power.larc.nasa.gov/
- FIRMS request date in run log: 2026-10-22
- Study area: 74.0–77.6°E, 28.4–32.6°N
- Study period: 2017–2023
- Seasonal period: October–November

## Rebuild the experiment

From the project directory:

```powershell
cd D:\agro2
.\.venv\Scripts\python.exe paper_results\build_fire_risk_results.py
```

The generator imports `exp_c_fire_risk.py`, reads only the cache directory, reconstructs the point-to-grid labels, and performs the exact year-based split.

## Existing full reproduction command

The original real-data benchmark can be rerun only if the FIRMS API key is available:

```powershell
cd D:\agro2
.\.venv\Scripts\python.exe exp_c_fire_risk.py --map-key <MAP_KEY> --years 2017-2023 --out .\outputs\results_c
```

Do not place the MAP_KEY in source files or paper deliverables.

## Target

Label each grid cell and date as 1 if it has at least one non-low-confidence VIIRS hotspot on day $t+1$, otherwise 0.

## Features

- T2M
- RH2M
- WS2M
- PRECTOTCORR
- Three-day precipitation sum
- Consecutive dry-day count
- Detection today
- Previous three-day detection count
- Previous seven-day detection count
- Day of year
- Latitude
- Longitude

## Split

- Train: 2017–2020
- Validation: 2021
- Test: 2022–2023
- Random seed: 0

No test-set tuning is used by the reported original method. Alert thresholds are derived from validation predictions for 80% cumulative positive recall.

## Additional models

Random Forest and XGBoost 0.5-threshold comparisons are generated as auxiliary evidence only. They do not modify or replace the reproduced reference metrics.
