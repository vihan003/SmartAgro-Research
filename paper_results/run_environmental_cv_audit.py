from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "environmental" / "processed"
OUT = ROOT / "paper_results" / "environmental_monitoring"

sets = {
    "predictive_irrigation": ["temperature_c", "humidity_pct", "soil_moisture_pct", "light_intensity", "hour", "day_of_week", "moisture_change_1", "moisture_change_3", "temperature_rolling_3", "humidity_rolling_3"],
    "motor_monitor": ["soil_moisture_pct", "temperature_c", "humidity_pct"],
}

rows = []
for key, features in sets.items():
    df = pd.read_csv(PROCESSED / f"{key}.csv", parse_dates=["timestamp"]).sort_values("timestamp")
    X = df[features]
    y = df["pump_status"]
    models = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        "random_forest": RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42),
    }
    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", random_state=42, n_jobs=-1)

    for model_name, model in models.items():
        fold_metrics = []
        positive_support = []
        for fold, (tr, te) in enumerate(TimeSeriesSplit(n_splits=5).split(X)):
            if len(set(y.iloc[tr])) < 2:
                continue
            model.fit(X.iloc[tr], y.iloc[tr])
            pred = model.predict(X.iloc[te])
            prob = model.predict_proba(X.iloc[te])[:, 1]
            tp = int(((pred == 1) & (y.iloc[te].to_numpy() == 1)).sum())
            positive_support.append(int(y.iloc[te].sum()))
            m = {
                "accuracy": accuracy_score(y.iloc[te], pred),
                "precision_macro": precision_score(y.iloc[te], pred, average="macro", zero_division=0),
                "recall_macro": recall_score(y.iloc[te], pred, average="macro", zero_division=0),
                "f1_macro": f1_score(y.iloc[te], pred, average="macro", zero_division=0),
                "pr_auc": average_precision_score(y.iloc[te], prob) if y.iloc[te].nunique() == 2 else np.nan,
                "roc_auc": roc_auc_score(y.iloc[te], prob) if y.iloc[te].nunique() == 2 else np.nan,
            }
            fold_metrics.append(m)
        if fold_metrics:
            for metric in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "pr_auc", "roc_auc"):
                vals = [m[metric] for m in fold_metrics if not np.isnan(m[metric])]
                rows.append({
                    "dataset": key,
                    "model": model_name,
                    "metric": metric,
                    "mean": float(np.mean(vals)) if vals else np.nan,
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "folds": len(vals),
                    "positive_support_per_fold": ";".join(map(str, positive_support)),
                    "timestamp_type": "source timestamp" if key == "predictive_irrigation" else "synthetic playback timestamp",
                })

pd.DataFrame(rows).to_csv(OUT / "temporal_cv_summary.csv", index=False)
print(json.dumps({"rows": len(rows), "output": str(OUT / "temporal_cv_summary.csv")}, indent=2))
