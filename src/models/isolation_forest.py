"""
Isolation Forest anomaly detector.

Selected because it scales near-linearly to millions of rows, needs no
distance/kernel computation (unlike LOF/OCSVM), handles the mixed-scale
numeric features produced by feature engineering gracefully, and its
"few splits to isolate = anomalous" mechanism directly matches the
intuition that fraud/anomalous transactions sit in sparse regions of
behavior-space.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import IsolationForest

from src.models.base import BaseAnomalyModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class IsolationForestModel(BaseAnomalyModel):
    """Wraps `sklearn.ensemble.IsolationForest`.

    Key hyperparameters:
    - `n_estimators`: number of isolation trees; more trees stabilize the
      anomaly score at higher compute cost.
    - `max_samples`: rows sampled per tree; smaller values increase
      isolation sensitivity and speed, at some variance cost.
    - `contamination`: expected anomaly proportion, used to set the
      internal (and our shared) decision threshold.
    - `max_features`: fraction of features sampled per tree/split.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = dict(config or {})
        cfg.pop("enabled", None)
        contamination = cfg.pop("contamination", 0.02)
        random_state = cfg.pop("random_state", 42)
        cfg.pop("hyperparameter_search", None)
        super().__init__(name="isolation_forest", contamination=contamination, random_state=random_state)
        self._sk_params = cfg
        self.model: Optional[IsolationForest] = None

    def _fit_model(self, X: np.ndarray) -> None:
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            **self._sk_params,
        )
        self.model.fit(X)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        # sklearn's decision_function: higher = more normal. We negate so
        # higher = more anomalous, matching the shared BaseAnomalyModel
        # convention used across every model in this project.
        return -self.model.decision_function(X)

    def feature_importances(self) -> Optional[np.ndarray]:
        """Isolation Forest has no native feature_importances_; average
        path-length sensitivity is approximated in
        `src/explainability/explain.py` via permutation importance
        instead. Kept here as an explicit None for interface clarity."""
        return None