"""Integration tests for src/pipeline.py — exercises the full
load-agnostic path (clean -> engineer -> split -> encode -> train ->
score) against an in-memory synthetic raw dataframe matching the real
schema, without touching disk-based raw data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pipeline import AnomalyDetectionPipeline
from src.utils.config import get_config


def _make_raw_df(n_users: int = 15, txns_per_user: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    merchants = list(range(50))
    for user in range(n_users):
        base_amount = rng.uniform(10, 100)
        day_cursor = 1
        for i in range(txns_per_user):
            day_cursor += rng.integers(0, 3)
            month = 1 + (day_cursor - 1) // 28
            day = 1 + (day_cursor - 1) % 28
            amount = max(0.5, rng.normal(base_amount, base_amount * 0.3))
            if rng.random() < 0.03:
                amount *= 50
            rows.append(
                {
                    "User": user,
                    "Card": 0,
                    "Year": 2020,
                    "Month": min(month, 12),
                    "Day": day,
                    "Time": f"{rng.integers(0, 24):02d}:{rng.integers(0, 60):02d}",
                    "Amount": f"${amount:.2f}",
                    "Use Chip": rng.choice(["Chip Transaction", "Swipe Transaction", "Online Transaction"]),
                    "Merchant Name": int(rng.choice(merchants)),
                    "Merchant City": rng.choice(["NYC", "LA", "SF", "ONLINE"]),
                    "Merchant State": rng.choice(["NY", "CA", None]),
                    "Zip": 10001.0,
                    "MCC": int(rng.choice([5411, 5732, 4829, 5999])),
                    "Errors?": None,
                    "Is Fraud?": "Yes" if rng.random() < 0.02 else "No",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def config():
    return get_config()


@pytest.fixture
def raw_df():
    return _make_raw_df()


class TestPipelineDataStages:
    def test_clean_and_engineer_returns_expected_shapes(self, config, raw_df):
        pipeline = AnomalyDetectionPipeline(config)
        engineered_df, y = pipeline.clean_and_engineer(raw_df.copy())
        assert len(engineered_df) == len(y)
        for col in pipeline.feature_engineer.engineered_columns():
            assert col in engineered_df.columns
        assert "Is Fraud?" not in engineered_df.columns

    def test_split_preserves_row_count_and_label_alignment(self, config, raw_df):
        pipeline = AnomalyDetectionPipeline(config)
        engineered_df, y = pipeline.clean_and_engineer(raw_df.copy())
        train_df, test_df, y_train, y_test = pipeline.split(engineered_df, y)
        assert len(train_df) == len(y_train)
        assert len(test_df) == len(y_test)
        assert len(train_df) + len(test_df) == len(engineered_df)

    def test_encode_produces_numeric_matrices_without_nans(self, config, raw_df):
        pipeline = AnomalyDetectionPipeline(config)
        engineered_df, y = pipeline.clean_and_engineer(raw_df.copy())
        train_df, test_df, y_train, y_test = pipeline.split(engineered_df, y)
        X_train, X_test, feature_cols = pipeline.encode(train_df, test_df)

        assert X_train.shape[0] == len(train_df)
        assert X_test.shape[0] == len(test_df)
        assert X_train.shape[1] == len(feature_cols)
        assert not np.isnan(X_train).any()
        assert not np.isnan(X_test).any()

    def test_encoder_fit_only_on_train_split(self, config, raw_df):
        """The FeatureEncoder must be fit using only train_df — verified
        by checking the fitted scaler's mean roughly matches train_df's
        Amount distribution (encode() is only ever passed train_df/test_df
        separately; fit() is called on train_df alone inside encode())."""
        pipeline = AnomalyDetectionPipeline(config)
        engineered_df, y = pipeline.clean_and_engineer(raw_df.copy())
        train_df, test_df, y_train, y_test = pipeline.split(engineered_df, y)
        pipeline.encode(train_df, test_df)
        assert pipeline.encoder.is_fitted
        amount_idx = pipeline.encoder.numeric_columns_.index("Amount")
        fitted_mean = pipeline.encoder.scaler.mean_[amount_idx]
        assert fitted_mean == pytest.approx(train_df["Amount"].mean(), rel=0.15)


class TestPipelineTrainAndScore:
    def _fit_pipeline(self, config, raw_df):
        pipeline = AnomalyDetectionPipeline(config)
        engineered_df, y = pipeline.clean_and_engineer(raw_df.copy())
        train_df, test_df, y_train, y_test = pipeline.split(engineered_df, y)
        X_train, X_test, _ = pipeline.encode(train_df, test_df)
        pipeline.train(X_train, save=False)
        return pipeline

    def test_train_fits_all_active_models(self, config, raw_df):
        pipeline = self._fit_pipeline(config, raw_df)
        for name in ["isolation_forest", "local_outlier_factor", "one_class_svm"]:
            assert name in pipeline.models
            assert pipeline.models[name].is_fitted

    def test_score_end_to_end_on_new_raw_data(self, config, raw_df):
        pipeline = self._fit_pipeline(config, raw_df)
        new_raw = raw_df.iloc[:10].copy()
        scored = pipeline.score(new_raw, model_name="isolation_forest")

        assert "anomaly_score" in scored.columns
        assert "anomaly_score_normalized" in scored.columns
        assert "is_anomaly" in scored.columns
        assert set(scored["is_anomaly"].unique()).issubset({0, 1})
        assert len(scored) == len(new_raw)

    def test_score_with_unknown_model_name_raises(self, config, raw_df):
        pipeline = self._fit_pipeline(config, raw_df)
        with pytest.raises(ValueError):
            pipeline.score(raw_df.iloc[:5], model_name="not_a_real_model")