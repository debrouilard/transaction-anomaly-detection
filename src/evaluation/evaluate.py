"""
Runs full evaluation for every trained/loaded model and persists results —
the counterpart to `src/models/train_models.py` in the evaluation stage.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate_model, get_pr_curve_data, get_roc_curve_data
from src.models.base import BaseAnomalyModel
from src.utils.config import Config, get_config
from src.utils.helpers import save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_all_models(
    models: Dict[str, BaseAnomalyModel],
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: Optional[Config] = None,
    save: bool = True,
) -> Dict[str, Dict]:
    """Evaluate every model in `models` on the same held-out (X_test,
    y_test), returning {model_name: metrics_dict}. Also computes and
    persists ROC/PR curve data per model for dashboard plotting."""
    cfg = config or get_config()
    k_values = cfg.models.evaluation.get("k_values", [50, 100, 500])
    y_test_arr = np.asarray(y_test)

    results: Dict[str, Dict] = {}
    curves: Dict[str, Dict] = {}

    for name, model in models.items():
        logger.info("Evaluating model: %s", name)
        metrics = evaluate_model(model, X_test, y_test_arr, k_values)
        results[name] = metrics

        scores = model.score_samples(X_test)
        curves[name] = {
            "roc": get_roc_curve_data(y_test_arr, scores),
            "pr": get_pr_curve_data(y_test_arr, scores),
        }

        logger.info(
            "  %-22s ROC-AUC=%.4f  AP=%.4f  P@100=%.4f  n_flagged=%d",
            name,
            metrics.get("roc_auc", float("nan")),
            metrics.get("average_precision", float("nan")),
            metrics.get("precision_at_100", float("nan")),
            metrics.get("n_flagged", 0),
        )

    if save:
        save_json(results, cfg.paths.metrics_path)
        save_json(curves, cfg.paths.reports_dir / "roc_pr_curves.json")

    return results


def metrics_to_dataframe(results: Dict[str, Dict]) -> pd.DataFrame:
    """Flatten the per-model metrics dicts into a tidy comparison
    dataframe, sorted by ROC-AUC descending (best model first)."""
    df = pd.DataFrame(list(results.values()))
    if "roc_auc" in df.columns:
        df = df.sort_values("roc_auc", ascending=False).reset_index(drop=True)
    return df