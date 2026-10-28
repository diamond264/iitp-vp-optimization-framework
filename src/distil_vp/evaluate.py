"""Evaluation utilities: binary metrics (accuracy, F1, precision, recall, AUC)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    y_prob: list[float] | None = None,
) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None:
        try:
            metrics["auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["auc"] = float("nan")
    return metrics


def format_metrics(metrics: dict[str, float]) -> str:
    return "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
