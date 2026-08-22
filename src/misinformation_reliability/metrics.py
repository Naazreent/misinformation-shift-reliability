"""Binary classification and calibration metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)


def expected_calibration_error(
    y_true: np.ndarray,
    probability_true: np.ndarray,
    bins: int = 10,
) -> float:
    """Equal-width ECE for the positive-class probability."""

    y = np.asarray(y_true, dtype=float)
    probability = np.asarray(probability_true, dtype=float)
    if len(y) == 0 or len(y) != len(probability):
        raise ValueError("y_true and probability_true must be non-empty and aligned")
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("Probabilities must be between 0 and 1")
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.digitize(probability, edges[1:-1], right=True)
    ece = 0.0
    for bin_id in range(bins):
        mask = bin_ids == bin_id
        if not mask.any():
            continue
        observed = float(y[mask].mean())
        confidence = float(probability[mask].mean())
        ece += float(mask.mean()) * abs(observed - confidence)
    return float(ece)


def binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probability_true: np.ndarray,
) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    prediction = np.asarray(y_pred, dtype=int)
    probability = np.asarray(probability_true, dtype=float)
    matrix = confusion_matrix(y, prediction, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_precision": float(precision_score(y, prediction, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, prediction, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, prediction, average="weighted", zero_division=0)),
        "brier_score": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, np.column_stack([1.0 - probability, probability]), labels=[0, 1])),
        "ece_10": expected_calibration_error(y, probability, bins=10),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

