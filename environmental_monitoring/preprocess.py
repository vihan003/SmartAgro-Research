from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXTRACTED = ROOT / "data" / "environmental" / "extracted"
PROCESSED = ROOT / "data" / "environmental" / "processed"
REPORTS = ROOT / "environmental_monitoring" / "results"


def slug(value: str) -> str:
    value = re.sub(r"[^0-Za-z0-9]+", "_", str(value).strip()).strip("_").lower()
    return value


def find_file(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern} under {directory}")
    return matches[0]


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(r"[^0-9+-.]", "", regex=True), errors="coerce")


def preprocess_predictive_irrigation() -> tuple[pd.DataFrame, dict]:
    src = find_file(EXTRACTED / "predictive_irrigation", "*.xlsx")
    raw = pd.read_excel(src)
    df = raw.rename(columns={slug(c): c for c in raw.columns}).copy()
    df.columns = [slug(c) for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for c in ("temperature_c", "humidity_pct", "soil_moisture_pct", "light_intensity", "pump_status"):
        df[c] = clean_numeric(df[c])
    df = df.drop_duplicates(subset=["timestamp"])
    df = df[df["timestamp"].notna()].copy()

    numeric = ["temperature_c", "humidity_pct", "soil_moisture_pct", "light_intensity"]
    for c in numeric:
        df[c] = df[c].interpolate(method="linear", limit_direction="both")
    df["pump_status"] = df["pump_status"].fillna(df["pump_status"].mode().iloc[0]).clip(0, 1).round().astype(int)

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["moisture_change_1"] = df["soil_moisture_pct"].diff(1)
    df["moisture_change_3"] = df["soil_moisture_pct"].diff(3)
    df["temperature_rolling_3"] = df["temperature_c"].rolling(3, min_periods=1).mean()
    df["humidity_rolling_3"] = df["humidity_pct"].rolling(3, min_periods=1).mean()

    feature_cols = ["temperature_c", "humidity_pct", "soil_moisture_pct", "light_intensity",
                    "hour", "day_of_week", "moisture_change_1", "moisture_change_3",
                    "temperature_rolling_3", "humidity_rolling_3"]
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True))

    meta = {
        "dataset_name": "Environmental Sensor Dataset for LSTM-Based Predictive Irrigation",
        "dataset_id": "predictive_irrigation_jtsyb593fy",
        "source_url": "https://data.mendeley.com/datasets/jtsyb593fy/1",
        "license": "CC BY 4.0",
        "doi": "10.17632/jtsyb593fy.1",
        "download_date": pd.Timestamp.now(tz="UTC").isoformat(),
        "raw_source_file": str(src.relative_to(ROOT)),
        "rows": int(len(df)), "columns": int(len(df.columns)),
        "features": feature_cols,
        "target": "pump_status",
        "timestamp_column": "timestamp",
        "missing_value_percent": float(df.isna().mean().mean() * 100),
        "pump_status_counts": df["pump_status"].value_counts().sort_index().astype(int).to_dict(),
        "limitations": ["20-day smart-garden testbed, 30-minute interval", "Not field-validated across crops or soils", "Pump status is the only explicit outcome label"]
    }
    return df, meta


def preprocess_motor_monitor() -> tuple[pd.DataFrame, dict]:
    src = find_file(EXTRACTED / "motor_monitor", "*.csv")
    raw = pd.read_csv(src)
    df = raw.rename(columns={slug(c): c for c in raw.columns}).copy()
    df.columns = [slug(c) for c in df.columns]
    mapping = {"soil_moisture": "soil_moisture_raw", "temperature": "temperature_c",
               "air_humidity": "humidity_pct", "pump_data": "pump_status"}
    df = df.rename(columns=mapping)
    df["soil_moisture_raw"] = clean_numeric(df["soil_moisture_raw"])
    df["soil_moisture_pct"] = df["soil_moisture_raw"]
    df["temperature_c"] = clean_numeric(df["temperature_c"])
    df["humidity_pct"] = clean_numeric(df["humidity_pct"])
    df["pump_status"] = clean_numeric(df["pump_status"]).clip(0, 1).round().astype(int)
    df = df.drop_duplicates().reset_index(drop=True)
    df["timestamp"] = pd.date_range("2023-01-01", periods=len(df), freq="30min")
    for c in ("soil_moisture_pct", "temperature_c", "humidity_pct"):
        df[c] = df[c].interpolate(method="linear", limit_direction="both")
    feature_cols = ["soil_moisture_pct", "temperature_c", "humidity_pct"]
    meta = {
        "dataset_name": "Soil Moisture, Air Temperature, Humidity and Motor on/off Monitoring data",
        "dataset_id": "motor_monitor_fpdwmm7nrb",
        "source_url": "https://data.mendeley.com/datasets/fpdwmm7nrb/1",
        "license": "CC BY NC 3.0",
        "doi": "10.17632/fpdwmm7nrb.1",
        "download_date": pd.Timestamp.now(tz="UTC").isoformat(),
        "raw_source_file": str(src.relative_to(ROOT)),
        "rows": int(len(df)), "columns": int(len(df.columns)),
        "features": feature_cols,
        "target": "pump_status",
        "timestamp_column": "timestamp",
        "missing_value_percent": float(df.isna().mean().mean() * 100),
        "pump_status_counts": df["pump_status"].value_counts().sort_index().astype(int).to_dict(),
        "limitations": ["No original timestamps in the source CSV", "Synthetic timestamps are playback-only chronology", "No crop or soil-class identity", "CC BY-NC limits commercial use"]
    }
    return df, meta


def preprocess_smartagro_iot() -> tuple[pd.DataFrame, dict]:
    base = EXTRACTED / "smartagro_iot"
    names = {
        "temperature_c": "Environment Temperature.xlsx",
        "humidity_pct": "Environment Humidity.xlsx",
        "light_lux": "Environment Light Intensity.xlsx",
        "soil_moisture_pct": "Soil Moisture.xlsx",
        "soil_ph": "Soil pH.xlsx",
        "soil_temperature_c": "Soil Temperature.xlsx",
    }
    frames = []
    for out_name, file_name in names.items():
        p = find_file(base, file_name)
        raw = pd.read_excel(p, header=None)
        values = raw.iloc[2:, 2] if raw.shape[1] >= 3 else raw.iloc[2:, 1]
        timestamps = raw.iloc[2:, 0]
        d = pd.DataFrame({"timestamp": pd.to_datetime(timestamps, errors="coerce", utc=True), out_name: clean_numeric(values)})
        d = d.dropna(subset=["timestamp"])
        frames.append(d)
    df = frames[0]
    for d in frames[1:]:
        df = df.merge(d, on="timestamp", how="outer")
    df = df.sort_values("timestamp").reset_index(drop=True)
    for c in [n for n in names if n != "timestamp"]:
        df[c] = df[c].interpolate(method="linear", limit_direction="both")
    meta = {
        "dataset_name": "Smart Agriculture and Plant Health Monitoring using IoT",
        "dataset_id": "smartagro_iot_65jxyrxv7b",
        "source_url": "https://data.mendeley.com/datasets/65jxyrxv7b/1",
        "license": "CC BY 4.0",
        "doi": "10.17632/65jxyrxv7b.1",
        "download_date": pd.Timestamp.now(tz="UTC").isoformat(),
        "raw_source_file": str(base.relative_to(ROOT)),
        "rows": int(len(df)), "columns": int(len(df.columns)),
        "features": [n for n in names],
        "target": None,
        "timestamp_column": "timestamp",
        "missing_value_percent": float(df.isna().mean().mean() * 100),
        "limitations": ["Sensor files contain 100 data readings and are joined by timestamp", "No pump or validated stress label", "Historical sensor archive, not live field sensing"]
    }
    return df, meta


def main() -> dict:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    meta = {}
    processors = {
        "predictive_irrigation": preprocess_predictive_irrigation,
        "motor_monitor": preprocess_motor_monitor,
        "smartagro_iot": preprocess_smartagro_iot,
    }
    for name, processor in processors.items():
        df, info = processor()
        df.to_csv(PROCESSED / f"{name}.csv", index=False)
        meta[name] = info
    (ROOT / "environmental_monitoring" / "datasets" / "dataset_metadata.json").parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "environmental_monitoring" / "datasets" / "dataset_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
