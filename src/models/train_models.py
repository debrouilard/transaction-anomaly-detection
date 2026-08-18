"""
Model registry and training orchestration.

`train_all_models` is the single place that knows how to instantiate every
configured model from `configs/model_config.yaml`, fit it, time it, and
persist it — used by both `scripts/train.py` and `src/pipeline.py` so the
two never drift out of sync.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.models.autoencoder import AutoencoderModel
from src.models.base import BaseAnomalyModel
from src.models.isolation_forest import IsolationForestModel
from src.models.local_outlier_factor import LocalOutlierFactorModel
from src.models.one_class_svm import OneClassSVMModel
from src.utils.config import Config, get_config
from src.utils.helpers import save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EllipticEnvelopeModel(BaseAnomalyModel):
    """Optional 5th model (bonus): assumes normal transactions follow a
    single Gaussian distribution in feature space and flags points with
    large Mahalanobis distance from its robust (MCD) center/covariance.
    Fast and interpretable, but — unlike the other three models — assumes
    a single unimodal, elliptical "normal" region, which is a much
    stronger (and often wrong) assumption for multi-segment customer
    behavior; included for comparison purposes when enabled in
    `configs/model_config.yaml` (`elliptic_envelope.enabled: true`)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        from sklearn.covariance import EllipticEnvelope

        cfg = dict(config or {})
        cfg.pop("enabled", None)
        contamination = cfg.pop("contamination", 0.02)
        random_state = cfg.pop("random_state", 42)
        super().__init__(name="elliptic_envelope", contamination=contamination, random_state=random_state)
        self._sk_params = cfg
        self._cls = EllipticEnvelope
        self.model = None

    def _fit_model(self, X: np.ndarray) -> None:
        self.model = self._cls(
            contamination=self.contamination, random_state=self.random_state, **self._sk_params
        )
        self.model.fit(X)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return -self.model.decision_function(X)


MODEL_REGISTRY = {
    "isolation_forest": (IsolationForestModel, "isolation_forest", "isolation_forest.pkl"),
    "local_outlier_factor": (LocalOutlierFactorModel, "local_outlier_factor", "lof.pkl"),
    "one_class_svm": (OneClassSVMModel, "one_class_svm", "one_class_svm.pkl"),
    "autoencoder": (AutoencoderModel, "autoencoder", "autoencoder.pt"),
    "elliptic_envelope": (EllipticEnvelopeModel, "elliptic_envelope", "elliptic_envelope.pkl"),
}


def get_model_class(name: str):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name][0]


def build_model(name: str, config: Optional[Config] = None) -> BaseAnomalyModel:
    """Instantiate (but do not fit) a single model by name, wired up with
    its hyperparameters from configs/model_config.yaml."""
    cfg = config or get_config()
    model_cls, cfg_key, _ = MODEL_REGISTRY[name]
    model_cfg = cfg.models[cfg_key].to_dict()
    return model_cls(model_cfg)


def active_model_names(config: Optional[Config] = None) -> list:
    """Return models explicitly enabled in model_config.yaml."""
    cfg = config or get_config()

    names = []

    for name in [
        "isolation_forest",
        "local_outlier_factor",
        "one_class_svm",
        "autoencoder",
        "elliptic_envelope",
    ]:
        model_cfg = cfg.models.get(name)

        if model_cfg and model_cfg.get("enabled", False):
            names.append(name)

    return names


def train_all_models(
    X_train: np.ndarray,
    config: Optional[Config] = None,
    save: bool = True,
    model_names: Optional[list] = None,
) -> Dict[str, BaseAnomalyModel]:
    """Instantiate, fit, and (optionally) persist every active model.

    Returns a dict of {model_name: fitted BaseAnomalyModel}.
    """
    cfg = config or get_config()
    names = model_names or active_model_names(cfg)
    fitted: Dict[str, BaseAnomalyModel] = {}
    timing: Dict[str, float] = {}

    for name in names:
        logger.info("=" * 70)
        logger.info("Training model: %s", name)
        model = build_model(name, cfg)
        model.fit(X_train)
        fitted[name] = model
        timing[name] = model.train_time_seconds_

        if save:
            _, _, filename = MODEL_REGISTRY[name]
            save_path: Path = cfg.paths.models_dir / filename
            model.save(save_path)

    if save:
        save_json(timing, cfg.paths.reports_dir / "training_times.json")

    logger.info("=" * 70)
    logger.info("Finished training %d model(s): %s", len(fitted), list(fitted.keys()))
    return fitted


def load_all_models(
    config: Optional[Config] = None, model_names: Optional[list] = None
) -> Dict[str, BaseAnomalyModel]:
    """Load every persisted model from `models/` for evaluation/inference."""
    cfg = config or get_config()
    names = model_names or active_model_names(cfg)
    loaded: Dict[str, BaseAnomalyModel] = {}
    for name in names:
        model_cls, _, filename = MODEL_REGISTRY[name]
        path = cfg.paths.models_dir / filename
        if not path.exists():
            logger.warning("No saved artifact for '%s' at %s — skipping.", name, path)
            continue
        loaded[name] = model_cls.load(path)
    return loaded