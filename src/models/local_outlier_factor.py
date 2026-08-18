"""
Local Outlier Factor (novelty mode) anomaly detector.

Selected because it captures *local density* anomalies that global methods
(Isolation Forest, OCSVM with a single global boundary) can miss — e.g. a
$300 transaction is unremarkable globally but highly anomalous for a
customer whose typical spend is $15. `novelty=True` is required (rather
than the default outlier-detection mode) so the fitted model exposes a
`.predict()` usable on new/unseen transactions at inference time, which a
production scoring service requires.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from src.models.base import BaseAnomalyModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LocalOutlierFactorModel(BaseAnomalyModel):
    """Wraps `sklearn.neighbors.LocalOutlierFactor(novelty=True)`.

    Key hyperparameters:
    - `n_neighbors`: size of the local neighborhood used to estimate
      density; small values are noise-sensitive, large values blur local
      structure into a near-global estimate.
    - `contamination`: expected anomaly proportion (decision threshold).
    - `metric`/`p`: distance metric for neighbor search (Minkowski/Euclidean
      by default), which matters because features are on different scales
      even after standardization.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = dict(config or {})
        contamination = cfg.pop("contamination", 0.02)
        random_state = cfg.pop("random_state", 42)  # LOF itself is deterministic; kept for interface parity
        cfg.pop("hyperparameter_search", None)
        cfg["novelty"] = True  # required for .predict()/.decision_function() on new data
        super().__init__(name="local_outlier_factor", contamination=contamination, random_state=random_state)
        self._sk_params = cfg
        self.model: Optional[LocalOutlierFactor] = None

    def _fit_model(self, X: np.ndarray) -> None:
        self.model = LocalOutlierFactor(contamination=self.contamination, **self._sk_params)
        self.model.fit(X)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        # decision_function: higher = more normal (LOF score near 1 = inlier,
        # << 0 = outlier). Negate for the shared "higher = more anomalous"
        # convention.
        return -self.model.decision_function(X)