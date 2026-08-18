"""
Evaluation metrics for unsupervised anomaly detection.

Every metric here is computed against the `Is Fraud?` label, which is
**never** used during model fitting (see `TransactionCleaner.extract_label`
and `src/pipeline.py`) — this module is the only place in the codebase
that is allowed to look at it, and only for honest, after-the-fact
evaluation of how well each unsupervised anomaly ranking agrees with real
fraud outcomes.

Because fraud is a rare-event problem (typically <1% positive), accuracy is
not reported as a headline metric; ROC-AUC, average precision, and
precision/recall@k are used instead, since they are meaningful under
severe class imbalance.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

from src.models.base import BaseAnomalyModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Of the top-k highest-scored (most anomalous) transactions, what
    fraction are actually fraud? Directly answers the business question:
    "if an analyst only has time to review k transactions today, how good
    is this ranking?" """
    if k <= 0 or len(scores) == 0:
        return float("nan")
    k = min(k, len(scores))
    top_k_idx = np.argsort(scores)[::-1][:k]
    return float(y_true[top_k_idx].sum() / k)


def recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Of all actual fraud cases, what fraction appear in the top-k
    highest-scored transactions?"""
    n_positive = int(y_true.sum())
    if n_positive == 0 or len(scores) == 0:
        return float("nan")
    k = min(k, len(scores))
    top_k_idx = np.argsort(scores)[::-1][:k]
    return float(y_true[top_k_idx].sum() / n_positive)


def compute_ranking_metrics(
    y_true: np.ndarray, scores: np.ndarray, k_values: Optional[List[int]] = None
) -> Dict[str, Any]:
    """Core ranking-quality metrics for a single model's anomaly scores
    against the held-out fraud label. Returns NaN for metrics that are
    undefined when there are zero (or all) positive labels, rather than
    raising, so evaluation can still run on small/edge-case samples."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    k_values = k_values or [50, 100, 500]

    result: Dict[str, Any] = {}

    n_pos = int(y_true.sum())
    if 0 < n_pos < len(y_true):
        result["roc_auc"] = float(roc_auc_score(y_true, scores))
        result["average_precision"] = float(average_precision_score(y_true, scores))
    else:
        logger.warning(
            "Cannot compute ROC-AUC/average precision: y_true has %d positive out of %d rows.",
            n_pos, len(y_true),
        )
        result["roc_auc"] = float("nan")
        result["average_precision"] = float("nan")

    for k in k_values:
        result[f"precision_at_{k}"] = precision_at_k(y_true, scores, k)
        result[f"recall_at_{k}"] = recall_at_k(y_true, scores, k)

    return result


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Precision/recall/F1 of the model's *binary* anomaly flag
    (thresholded via `contamination`) against the fraud label — a coarser,
    complementary view to the ranking metrics above."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n_flagged": int(y_pred.sum()),
        "n_actual_fraud": int(y_true.sum()),
        "n_flagged_correctly": int(((y_pred == 1) & (y_true == 1)).sum()),
    }


def get_roc_curve_data(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, List[float]]:
    """Return FPR/TPR arrays for plotting the ROC curve in the dashboard/
    report. Empty lists if the label is degenerate (all-0 or all-1)."""
    y_true = np.asarray(y_true).astype(int)
    if 0 < y_true.sum() < len(y_true):
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds.tolist()}
    return {"fpr": [], "tpr": [], "thresholds": []}


def get_pr_curve_data(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, List[float]]:
    """Return precision/recall arrays for plotting the PR curve — more
    informative than ROC under heavy class imbalance."""
    y_true = np.asarray(y_true).astype(int)
    if 0 < y_true.sum() < len(y_true):
        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        return {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": thresholds.tolist(),
        }
    return {"precision": [], "recall": [], "thresholds": []}


def time_inference(model: BaseAnomalyModel, X: np.ndarray) -> float:
    """Wall-clock time (seconds) for `model.predict(X)` on the full array —
    used for the model-comparison runtime table."""
    start = time.perf_counter()
    model.predict(X)
    return round(time.perf_counter() - start, 4)


def evaluate_model(
    model: BaseAnomalyModel,
    X_test: np.ndarray,
    y_test: np.ndarray,
    k_values: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Full single-model evaluation: ranking metrics, classification
    metrics on the thresholded flag, and runtime — everything needed for
    one row of the model-comparison table."""
    scores = model.score_samples(X_test)
    preds = model.predict(X_test)

    metrics: Dict[str, Any] = {"model": model.name}
    metrics.update(compute_ranking_metrics(y_test, scores, k_values))
    metrics.update(compute_classification_metrics(y_test, preds))
    metrics["train_time_seconds"] = model.train_time_seconds_
    metrics["prediction_time_seconds"] = time_inference(model, X_test)
    metrics["n_test_rows"] = len(X_test)
    return metrics