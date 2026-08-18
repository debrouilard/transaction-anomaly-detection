"""
Common interface for every anomaly-detection model in this project.

Design: every model produces a `score_samples(X)` anomaly score where
**higher = more anomalous** (a uniform convention across Isolation Forest,
LOF, One-Class SVM, and the Autoencoder, even though their underlying
libraries use different/opposite raw conventions — each subclass is
responsible for normalizing sign). `fit()` additionally computes a
decision threshold from the training scores (via `contamination`, or an
explicit percentile for the Autoencoder), so `predict()` is a simple,
consistent `score >= threshold_` across all models — this is what makes a
fair, apples-to-apples model comparison in `src/evaluation/` possible.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.utils.helpers import save_artifact, load_artifact
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseAnomalyModel(ABC):
    """Abstract base class for all anomaly detection model wrappers."""

    def __init__(self, name: str, contamination: float = 0.02, random_state: int = 42):
        self.name = name
        self.contamination = contamination
        self.random_state = random_state
        self.is_fitted: bool = False
        self.threshold_: Optional[float] = None
        self.train_time_seconds_: Optional[float] = None
        self.n_features_: Optional[int] = None

    # ------------------------------------------------------------------ #
    # Subclasses must implement these two methods.
    # ------------------------------------------------------------------ #

    @abstractmethod
    def _fit_model(self, X: np.ndarray) -> None:
        """Fit the underlying estimator on X. Must set `self.is_fitted`
        is handled by the public `fit()` wrapper, not here."""
        raise NotImplementedError

    @abstractmethod
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return an anomaly score per row of X, where higher = more
        anomalous. Must be callable after `fit()`."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Shared, concrete behavior.
    # ------------------------------------------------------------------ #

    def fit(self, X: np.ndarray, threshold_percentile: Optional[float] = None) -> "BaseAnomalyModel":
        X = np.asarray(X, dtype=float)
        self.n_features_ = X.shape[1]
        logger.info("Fitting %s on %s rows x %d features...", self.name, f"{X.shape[0]:,}", X.shape[1])

        start = time.perf_counter()
        self._fit_model(X)
        self.train_time_seconds_ = round(time.perf_counter() - start, 4)
        self.is_fitted = True

        train_scores = self.score_samples(X)
        pct = threshold_percentile if threshold_percentile is not None else 100 * (1 - self.contamination)
        self.threshold_ = float(np.percentile(train_scores, pct))

        logger.info(
            "%s fit in %.3fs; decision threshold set at the %.2f percentile (%.6f).",
            self.name, self.train_time_seconds_, pct, self.threshold_,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Binary anomaly labels: 1 = anomalous, 0 = normal."""
        self._check_fitted()
        scores = self.score_samples(np.asarray(X, dtype=float))
        return (scores >= self.threshold_).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Min-max normalized anomaly score in [0, 1], for ranking /
        dashboard display. Not a calibrated probability."""
        self._check_fitted()
        scores = self.score_samples(np.asarray(X, dtype=float))
        lo, hi = scores.min(), scores.max()
        if hi - lo < 1e-12:
            return np.zeros_like(scores)
        return (scores - lo) / (hi - lo)

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} has not been fit yet. Call fit(X) first.")

    def get_params(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "contamination": self.contamination,
            "random_state": self.random_state,
            "threshold": self.threshold_,
            "n_features": self.n_features_,
            "train_time_seconds": self.train_time_seconds_,
        }

    def save(self, path: Path) -> None:
        """Default persistence via joblib; overridden by AutoencoderModel
        (PyTorch state dicts aren't cleanly joblib-picklable)."""
        save_artifact(self, path)

    @classmethod
    def load(cls, path: Path) -> "BaseAnomalyModel":
        return load_artifact(path)