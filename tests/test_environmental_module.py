from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_processed_environmental_files_exist():
    for name in ("predictive_irrigation", "motor_monitor", "smartagro_iot"):
        p = ROOT / "data" / "environmental" / "processed" / f"{name}.csv"
        assert p.exists(), p
        df = pd.read_csv(p)
        assert len(df) > 0


def test_dataset_metadata_is_real():
    p = ROOT / "environmental_monitoring" / "datasets" / "dataset_metadata.json"
    assert p.exists()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["predictive_irrigation"]["target"] == "pump_status"
    assert data["motor_monitor"]["target"] == "pump_status"
    assert data["smartagro_iot"]["target"] is None


def test_environmental_model_results_exist_and_are_bounded():
    p = ROOT / "environmental_monitoring" / "results" / "environmental_model_results.json"
    assert p.exists()
    results = json.loads(p.read_text(encoding="utf-8"))
    for dataset in results.values():
        for metrics in dataset["models"].values():
            assert 0 <= metrics["accuracy"] <= 1
            assert 0 <= metrics["f1_macro"] <= 1


def test_flask_endpoints():
    from environmental_monitoring.app import app
    client = app.test_client()
    assert client.get("/").status_code == 200
    assert client.get("/api/datasets").status_code == 200
    assert client.get("/api/monitor?dataset=smartagro_iot&index=10").status_code == 200
    assert client.get("/api/history?dataset=predictive_irrigation&limit=5").status_code == 200
    assert client.get("/api/metrics").status_code == 200


def test_stress_engine():
    from environmental_monitoring.stress_detection import stress_for_row
    normal = stress_for_row({"soil_moisture_pct": 55, "temperature_c": 28, "humidity_pct": 65})
    assert normal["category"] == "Normal"
    stressed = stress_for_row({"soil_moisture_pct": 20, "temperature_c": 38, "humidity_pct": 90})
    assert stressed["category"] in ("Moderate stress", "High stress")
