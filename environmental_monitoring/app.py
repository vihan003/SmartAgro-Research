from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environmental_monitoring.stress_detection import stress_for_row

app = Flask(__name__)
DATA_DIR = ROOT / "data" / "environmental" / "processed"
META_FILE = ROOT / "environmental_monitoring" / "datasets" / "dataset_metadata.json"
RESULT_DIR = ROOT / "environmental_monitoring" / "results"
MODEL_DIR = ROOT / "environmental_monitoring" / "saved_models"

metadata = json.loads(META_FILE.read_text(encoding="utf-8"))
data_cache = {
    "smartagro_iot": pd.read_csv(DATA_DIR / "smartagro_iot.csv", parse_dates=["timestamp"]),
    "predictive_irrigation": pd.read_csv(DATA_DIR / "predictive_irrigation.csv", parse_dates=["timestamp"]),
    "motor_monitor": pd.read_csv(DATA_DIR / "motor_monitor.csv", parse_dates=["timestamp"]),
}
alert_history = []


def clean_json_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    return str(value)


def row_payload(row: pd.Series):
    return {c: clean_json_value(v) for c, v in row.items()}


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/datasets")
def datasets():
    return jsonify({"datasets": metadata, "playback_label": "Recorded Dataset – Simulated Monitoring"})


@app.get("/api/monitor")
def monitor():
    dataset = request.args.get("dataset", "smartagro_iot")
    if dataset not in data_cache:
        return jsonify({"error": "Unknown dataset"}), 404
    df = data_cache[dataset]
    try:
        index = max(0, min(int(request.args.get("index", len(df) - 1)), len(df) - 1))
    except ValueError:
        index = len(df) - 1
    row = row_payload(df.iloc[index])
    stress = stress_for_row(row)
    for alert in stress["alerts"]:
        alert_history.append({
            "timestamp": row.get("timestamp"),
            "dataset": metadata[dataset]["dataset_name"],
            "alert_type": alert["type"],
            "severity": alert["severity"],
            "reason": alert["reason"],
            "suggested_action": alert["action"],
            "triggering_readings": {k: row.get(k) for k in ("temperature_c", "humidity_pct", "soil_moisture_pct", "soil_temperature_c", "soil_ph", "light_lux") if k in row},
        })
    if len(alert_history) > 500:
        del alert_history[:-500]
    return jsonify({
        "dataset": dataset,
        "dataset_name": metadata[dataset]["dataset_name"],
        "source_url": metadata[dataset]["source_url"],
        "license": metadata[dataset]["license"],
        "playback_label": "Recorded Dataset – Simulated Monitoring",
        "index": index,
        "total": len(df),
        "reading": row,
        "stress": stress,
        "pump_status": row.get("pump_status"),
    })


@app.get("/api/history")
def history():
    dataset = request.args.get("dataset", "smartagro_iot")
    if dataset not in data_cache:
        return jsonify({"error": "Unknown dataset"}), 404
    try:
        limit = max(1, min(int(request.args.get("limit", 200)), 2000))
    except ValueError:
        limit = 200
    df = data_cache[dataset].tail(limit)
    rows = [row_payload(r) for _, r in df.iterrows()]
    return jsonify({"dataset": dataset, "rows": rows})


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    dataset = payload.pop("dataset", "predictive_irrigation")
    if dataset not in ("predictive_irrigation", "motor_monitor"):
        return jsonify({"error": "Prediction is only available for pump-status datasets"}), 400
    model_path = MODEL_DIR / f"{dataset}_logistic_regression.joblib"
    meta_path = MODEL_DIR / f"{dataset}_metadata.joblib"
    if not model_path.exists():
        return jsonify({"error": "Model not trained"}), 404
    model = joblib.load(model_path)
    model_meta = joblib.load(meta_path)
    features = model_meta["features"]
    defaults = data_cache[dataset][features].median(numeric_only=True).to_dict()
    x = pd.DataFrame([{f: payload.get(f, defaults[f]) for f in features}], columns=features)
    prediction = int(model.predict(x)[0])
    probability = float(model.predict_proba(x)[0][1]) if 1 in model.classes_ else 0.0
    return jsonify({"dataset": dataset, "pump_status_prediction": prediction,
                    "pump_probability": probability,
                    "features_used": features,
                    "disclaimer": "Dataset-derived prediction; not a recommendation for physical irrigation equipment."})


@app.get("/api/metrics")
def metrics():
    result_files = sorted(RESULT_DIR.glob("*_metrics.json"))
    output = {}
    for p in result_files:
        name = p.stem.replace("_metrics", "")
        output[name] = json.loads(p.read_text(encoding="utf-8"))
    return jsonify(output)


@app.get("/api/alerts")
def alerts():
    return jsonify({"alerts": list(reversed(alert_history[-200:])), "count": len(alert_history)})


@app.get("/api/stress")
def stress_api():
    dataset = request.args.get("dataset", "smartagro_iot")
    if dataset not in data_cache:
        return jsonify({"error": "Unknown dataset"}), 404
    df = data_cache[dataset]
    try:
        index = max(0, min(int(request.args.get("index", len(df)-1)), len(df)-1))
    except ValueError:
        index = len(df)-1
    return jsonify(stress_for_row(row_payload(df.iloc[index])))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
