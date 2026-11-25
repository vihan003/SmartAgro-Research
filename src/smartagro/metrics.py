from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def compute_classification_metrics(y_true, y_pred, classes: List[str]) -> Dict[str, object]:
    """Computes multiclass metrics and per-class metrics for classification outputs."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(classes)))),
        "per_class": {},
    }

    for i, label in enumerate(classes):
        mask = y_true == i
        pred_for_class = y_pred[mask]
        tp = np.sum(pred_for_class == i)
        fp = np.sum((y_pred != i) & (y_true != i))
        fn = np.sum((y_pred != i) & (y_true == i))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        metrics["per_class"][label] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(mask.sum()),
        }

    return metrics
