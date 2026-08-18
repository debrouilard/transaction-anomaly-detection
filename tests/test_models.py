"""Unit tests for src/models/*.py — each model wrapper's fit/predict/score
contract, plus save/load round-trips."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.isolation_forest import IsolationForestModel
from src.models.local_outlier_factor import LocalOutlierFactorModel
from src.models.one_class_svm import OneClassSVMModel
from src.models.train_models import (
    active_model_names,
    build_model,
    train_all_models,
)
from src.utils.config import get_config

torch = pytest.importorskip("torch", reason="PyTorch not installed in this environment")
from src.models.autoencoder import AutoencoderModel  # noqa: E402


@pytest.fixture(scope="module")
def config():
    return get_config()


@pytest.fixture
def synthetic_data():
    """200 'normal' points clustered near the origin + 10 far-away
    'anomalous' points, so every model should reliably flag the latter."""
    rng = np.random.default_rng(0)
    normal = rng.normal(loc=0.0, scale=1.0, size=(200, 5))
    anomalies = rng.normal(loc=15.0, scale=1.0, size=(10, 5))
    X = np.vstack([normal, anomalies])
    is_anomaly_true = np.array([0] * 200 + [1] * 10)
    return X, is_anomaly_true


MODEL_CLASSES_AND_CONFIG_KEYS = [
    (IsolationForestModel, "isolation_forest"),
    (LocalOutlierFactorModel, "local_outlier_factor"),
    (OneClassSVMModel, "one_class_svm"),
]


class TestSklearnBackedModels:
    @pytest.mark.parametrize("model_cls,cfg_key", MODEL_CLASSES_AND_CONFIG_KEYS)
    def test_fit_predict_score_contract(self, config, synthetic_data, model_cls, cfg_key):
        X, y_true = synthetic_data
        model_cfg = config.models[cfg_key].to_dict()
        model = model_cls(model_cfg)

        model.fit(X)
        assert model.is_fitted
        assert model.threshold_ is not None

        scores = model.score_samples(X)
        assert scores.shape == (X.shape[0],)

        preds = model.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})

        # The synthetic anomalies should score higher on average than normals.
        assert scores[y_true == 1].mean() > scores[y_true == 0].mean()

    @pytest.mark.parametrize("model_cls,cfg_key", MODEL_CLASSES_AND_CONFIG_KEYS)
    def test_predict_before_fit_raises(self, config, model_cls, cfg_key):
        model_cfg = config.models[cfg_key].to_dict()
        model = model_cls(model_cfg)
        with pytest.raises(RuntimeError):
            model.predict(np.zeros((5, 5)))

    @pytest.mark.parametrize("model_cls,cfg_key", MODEL_CLASSES_AND_CONFIG_KEYS)
    def test_save_load_roundtrip(self, config, synthetic_data, model_cls, cfg_key, tmp_path):
        X, _ = synthetic_data
        model_cfg = config.models[cfg_key].to_dict()
        model = model_cls(model_cfg)
        model.fit(X)

        path = tmp_path / f"{cfg_key}.pkl"
        model.save(path)
        loaded = model_cls.load(path)

        np.testing.assert_allclose(model.score_samples(X), loaded.score_samples(X))
        np.testing.assert_array_equal(model.predict(X), loaded.predict(X))

    def test_predict_proba_is_normalized(self, config, synthetic_data):
        X, _ = synthetic_data
        model = IsolationForestModel(config.models["isolation_forest"].to_dict())
        model.fit(X)
        proba = model.predict_proba(X)
        assert proba.min() >= 0.0
        assert proba.max() <= 1.0 + 1e-9


class TestAutoencoderModel:
    def test_fit_predict_score_contract(self, config, synthetic_data):
        X, y_true = synthetic_data
        ae_cfg = config.models["autoencoder"].to_dict()
        ae_cfg["training"] = dict(ae_cfg["training"])
        ae_cfg["training"]["epochs"] = 5  # keep the test fast
        model = AutoencoderModel(ae_cfg)

        model.fit(X.astype(np.float32))
        assert model.is_fitted

        scores = model.score_samples(X.astype(np.float32))
        assert scores.shape == (X.shape[0],)
        assert scores[y_true == 1].mean() >= 0  # sanity: non-negative reconstruction error

        preds = model.predict(X.astype(np.float32))
        assert set(np.unique(preds)).issubset({0, 1})

    def test_save_load_roundtrip(self, config, synthetic_data, tmp_path):
        X, _ = synthetic_data
        ae_cfg = config.models["autoencoder"].to_dict()
        ae_cfg["training"] = dict(ae_cfg["training"])
        ae_cfg["training"]["epochs"] = 3
        model = AutoencoderModel(ae_cfg)
        model.fit(X.astype(np.float32))

        path = tmp_path / "autoencoder.pt"
        model.save(path)
        loaded = AutoencoderModel.load(path)

        np.testing.assert_allclose(
            model.score_samples(X.astype(np.float32)),
            loaded.score_samples(X.astype(np.float32)),
            rtol=1e-4,
        )

    def test_reconstruction_error_per_feature_shape(self, config, synthetic_data):
        X, _ = synthetic_data
        ae_cfg = config.models["autoencoder"].to_dict()
        ae_cfg["training"] = dict(ae_cfg["training"])
        ae_cfg["training"]["epochs"] = 3
        model = AutoencoderModel(ae_cfg)
        model.fit(X.astype(np.float32))

        per_feature = model.reconstruction_error_per_feature(X.astype(np.float32))
        assert per_feature.shape == X.shape


class TestTrainModelsOrchestration:
    def test_active_model_names_includes_all_four_required(self, config):
        names = active_model_names(config)
        for required in ["isolation_forest", "local_outlier_factor", "one_class_svm", "autoencoder"]:
            assert required in names

    def test_build_model_returns_correct_type(self, config):
        model = build_model("isolation_forest", config)
        assert isinstance(model, IsolationForestModel)

    def test_train_all_models_fits_every_active_model(self, config, synthetic_data):
        X, _ = synthetic_data
        # save=False: exercises the fit orchestration without writing into
        # the real models/ folder (persistence itself is covered by each
        # model's own save/load round-trip test above).
        fitted = train_all_models(
            X, config, save=False, model_names=["isolation_forest", "local_outlier_factor"]
        )
        assert set(fitted.keys()) == {"isolation_forest", "local_outlier_factor"}
        for model in fitted.values():
            assert model.is_fitted