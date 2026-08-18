"""
One-Class SVM anomaly detector.

Selected because the RBF-kernel decision boundary can capture non-linear,
non-convex "normal" regions of behavior-space that a purely tree-based or
density-based method might approximate more crudely.

Trade-off (documented, not hidden): OCSVM with an RBF kernel is
O(n^2)-O(n^3) in the number of training rows, which makes fitting on the
full engineered dataset impractical. We therefore fit on a fixed-size
random subsample (`train_subsample_size` in configs/model_config.yaml,
default 40,000 rows) — the model is still evaluated on the full held-out
test set, so reported metrics are honest; only *training* is subsampled,
a standard, disclosed engineering trade-off for kernel methods at scale.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.svm import OneClassSVM

from src.models.base import BaseAnomalyModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OneClassSVMModel(BaseAnomalyModel):
    """Wraps `sklearn.svm.OneClassSVM`.

    Key hyperparameters:
    - `nu`: upper bound on the fraction of training errors / lower bound on
      the fraction of support vectors — analogous to `contamination`.
    - `gamma`: RBF kernel coefficient; controls how tightly the boundary
      hugs the training data (small = smoother/more generalized boundary).
    - `kernel`: fixed to RBF for non-linear boundaries (vs. a linear SVM,
      which would reduce to a global hyperplane, similar to what a much
      cheaper linear model could already offer).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = dict(config or {})
        self.train_subsample_size = cfg.pop("train_subsample_size", 40_000)
        cfg.pop("hyperparameter_search", None)
        nu = cfg.pop("nu", 0.02)
        random_state = cfg.pop("random_state", 42)
        super().__init__(name="one_class_svm", contamination=nu, random_state=random_state)
        self.nu = nu
        self._sk_params = cfg
        self.model: Optional[OneClassSVM] = None

    def _fit_model(self, X: np.ndarray) -> None:
        if X.shape[0] > self.train_subsample_size:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(X.shape[0], size=self.train_subsample_size, replace=False)
            X_train = X[idx]
            logger.info(
                "OneClassSVM: subsampled %s / %s training rows for tractability (RBF kernel is O(n^2)).",
                f"{self.train_subsample_size:,}", f"{X.shape[0]:,}",
            )
        else:
            X_train = X

        self.model = OneClassSVM(nu=self.nu, **self._sk_params)
        self.model.fit(X_train)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        # decision_function: positive = inlier, negative = outlier
        # (higher = more normal). Negate for "higher = more anomalous".
        return -self.model.decision_function(X)