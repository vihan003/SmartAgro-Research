from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             average_precision_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "environmental" / "processed"
MODEL_DIR = ROOT / "environmental_monitoring" / "saved_models"
RESULT_DIR = ROOT / "environmental_monitoring" / "results"


def chronological_split(df: pd.DataFrame, date_col: str):
    d = df.sort_values(date_col).reset_index(drop=True)
    n = len(d)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    return d.iloc[:n_train], d.iloc[n_train:n_train+n_val], d.iloc[n_train+n_val:]


def plot_confusion(cm: np.ndarray, labels: list[str], title: str, path: Path):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def train_dataset(name: str, df: pd.DataFrame, features: list[str], target: str):
    tr, va, te = chronological_split(df, "timestamp")
    if tr[target].nunique() < 2 or va[target].nunique() < 2 or te[target].nunique() < 2:
        raise ValueError(f"Insufficient chronological class coverage for {name}")

    models = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=250, learning_rate=0.05, max_depth=3, random_state=42),
    }
    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05,
                                           subsample=0.9, colsample_bytree=0.9,
                                           eval_metric="logloss", random_state=42, n_jobs=-1)

    results = {"dataset": name, "features": features, "target": target,
               "splits": {"train": len(tr), "validation": len(va), "test": len(te)},
               "train_end": str(tr.timestamp.max()), "validation_end": str(va.timestamp.max()),
               "test_start": str(te.timestamp.min()), "test_end": str(te.timestamp.max()),
               "models": {}}

    best = None
    for model_name, model in models.items():
        model.fit(tr[features], tr[target])
        pred = model.predict(te[features])
        prob = model.predict_proba(te[features])[:, 1] if len(model.classes_) == 2 else None
        report = {
            "accuracy": float(accuracy_score(te[target], pred)),
            "precision_macro": float(precision_score(te[target], pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(te[target], pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(te[target], pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(te[target], pred, average="weighted", zero_division=0)),
        }
        if prob is not None:
            report["roc_auc"] = float(roc_auc_score(te[target], prob))
            report["pr_auc"] = float(average_precision_score(te[target], prob))
        cm = confusion_matrix(te[target], pred, labels=sorted(te[target].unique()))
        report["confusion_matrix"] = cm.tolist()
        report["classification_report"] = classification_report(te[target], pred, output_dict=True, zero_division=0)
        results["models"][model_name] = report

        joblib.dump(model, MODEL_DIR / f"{name}_{model_name}.joblib")
        plot_confusion(cm, [str(x) for x in sorted(te[target].unique())], f"{name}: {model_name}",
                       RESULT_DIR / f"{name}_{model_name}_confusion_matrix.png")

        if best is None or report["pr_auc"] > results["models"][best]["pr_auc"]:
            best = model_name

    results["best_model"] = best
    joblib.dump({"features": features, "target": target, "dataset": name}, MODEL_DIR / f"{name}_metadata.joblib")
    (RESULT_DIR / f"{name}_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    names = list(results["models"])
    fig, ax = plt.subplots(figsize=(9, 5))
    pr = [results["models"][n].get("pr_auc", 0) for n in names]
    ax.bar(names, pr, color="#3977A8")
    ax.set_ylim(0, 1); ax.set_ylabel("PR-AUC"); ax.set_title(f"{name}: chronological test PR-AUC")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout(); fig.savefig(RESULT_DIR / f"{name}_model_comparison.png", dpi=180); plt.close(fig)

    best_model = joblib.load(MODEL_DIR / f"{name}_{best}.joblib")
    if hasattr(best_model, "feature_importances_"):
        imp = best_model.feature_importances_
    elif hasattr(best_model, "named_steps") and hasattr(best_model.named_steps[list(best_model.named_steps)[-1]], "feature_importances_"):
        imp = best_model.named_steps[list(best_model.named_steps)[-1]].feature_importances_
    else:
        imp = None
    if imp is not None:
        order = np.argsort(imp)[::-1]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh([features[i] for i in order], imp[order], color="#27877B")
        ax.invert_yaxis(); ax.set_title(f"{name}: feature importance")
        fig.tight_layout(); fig.savefig(RESULT_DIR / f"{name}_feature_importance.png", dpi=180); plt.close(fig)
    return results


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}
    predictive = pd.read_csv(PROCESSED / "predictive_irrigation.csv", parse_dates=["timestamp"])
    predictive_features = ["temperature_c", "humidity_pct", "soil_moisture_pct", "light_intensity",
                           "hour", "day_of_week", "moisture_change_1", "moisture_change_3",
                           "temperature_rolling_3", "humidity_rolling_3"]
    all_results["predictive_irrigation"] = train_dataset("predictive_irrigation", predictive,
                                                           predictive_features, "pump_status")

    motor = pd.read_csv(PROCESSED / "motor_monitor.csv", parse_dates=["timestamp"])
    motor_features = ["soil_moisture_pct", "temperature_c", "humidity_pct"]
    all_results["motor_monitor"] = train_dataset("motor_monitor", motor, motor_features, "pump_status")

    (RESULT_DIR / "environmental_model_results.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
