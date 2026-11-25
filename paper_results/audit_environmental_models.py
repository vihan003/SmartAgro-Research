from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "environmental" / "processed"
OUT = ROOT / "paper_results" / "environmental_monitoring"
OUT.mkdir(parents=True, exist_ok=True)

feature_sets = {
    "predictive_irrigation": ["temperature_c", "humidity_pct", "soil_moisture_pct", "light_intensity", "hour", "day_of_week", "moisture_change_1", "moisture_change_3", "temperature_rolling_3", "humidity_rolling_3"],
    "motor_monitor": ["soil_moisture_pct", "temperature_c", "humidity_pct"],
}

rows = []
for key, features in feature_sets.items():
    df = pd.read_csv(PROCESSED / f"{key}.csv", parse_dates=["timestamp"])
    n = len(df)
    train_n = int(n * 0.70)
    val_n = int(n * 0.15)
    tr = df.iloc[:train_n]
    va = df.iloc[train_n:train_n + val_n]
    te = df.iloc[train_n + val_n:] 

    feature_exact_duplicates = int(df.duplicated(subset=features).sum())
    feature_target_counts = df.groupby(features)["pump_status"].nunique()
    conflicting = int((feature_target_counts > 1).sum())

    train_feature_keys = set(map(tuple, tr[features].round(12).to_numpy()))
    test_feature_keys = set(map(tuple, te[features].round(12).to_numpy()))
    overlap = len(train_feature_keys & test_feature_keys)

    class_stats = df.groupby("pump_status")[features].agg(["count", "min", "mean", "max"])
    corr = df[features + ["pump_status"]].corr(numeric_only=True)["pump_status"].drop("pump_status")

    rows.append({
        "dataset": key,
        "records": n,
        "feature_exact_duplicate_rows": feature_exact_duplicates,
        "feature_patterns_with_conflicting_labels": conflicting,
        "train_test_feature_pattern_overlap": overlap,
        "train_test_feature_pattern_overlap_percent_of_test": overlap / len(te) * 100,
        "train_class_0": int(tr.pump_status.eq(0).sum()),
        "train_class_1": int(tr.pump_status.eq(1).sum()),
        "validation_class_0": int(va.pump_status.eq(0).sum()),
        "validation_class_1": int(va.pump_status.eq(1).sum()),
        "test_class_0": int(te.pump_status.eq(0).sum()),
        "test_class_1": int(te.pump_status.eq(1).sum()),
        "highest_absolute_feature_correlation": float(corr.abs().max()),
        "most_correlated_feature": str(corr.abs().idxmax()),
        "source_timestamp_available": key == "predictive_irrigation",
        "labels_real_or_rule_generated": "Explicit source pump-status label; field validity not independently verified",
        "reliability_assessment": (
            "Not reliable for generalization: test positive support is only 2" if key == "predictive_irrigation"
            else "Exploratory only: timestamps synthetic, and perfect metrics may reflect strong separability or repeated sensor patterns"
        ),
    })

    class_stats.to_csv(OUT / f"{key}_class_feature_stats.csv")

pd.DataFrame(rows).to_csv(OUT / "model_audit_summary.csv", index=False)
print(json.dumps(rows, indent=2))
