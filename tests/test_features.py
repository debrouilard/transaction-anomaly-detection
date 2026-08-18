"""Unit tests for src/features/feature_engineering.py."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.preprocessing import TransactionCleaner
from src.features.feature_engineering import FeatureEngineer
from src.utils.config import get_config


@pytest.fixture(scope="module")
def config():
    return get_config()


@pytest.fixture
def cleaned_df(config):
    raw = pd.DataFrame(
        {
            "User": [1, 1, 1, 1, 2, 2],
            "Card": [0, 0, 0, 0, 0, 0],
            "Year": [2020, 2020, 2020, 2020, 2020, 2020],
            "Month": [1, 1, 1, 1, 1, 1],
            "Day": [1, 1, 2, 2, 1, 5],
            "Time": ["09:00", "09:30", "10:00", "20:00", "12:00", "12:00"],
            "Amount": ["$50.00", "$55.00", "$5000.00", "$60.00", "$20.00", "$25.00"],
            "Use Chip": ["Chip Transaction"] * 6,
            "Merchant Name": [111, 111, 222, 111, 333, 444],
            "Merchant City": ["NYC", "NYC", "LA", "NYC", "SF", "SF"],
            "Merchant State": ["NY", "NY", "CA", "NY", "CA", "CA"],
            "Zip": [10001.0] * 6,
            "MCC": [5411] * 6,
            "Errors?": [None] * 6,
            "Is Fraud?": ["No", "No", "No", "No", "No", "No"],
        }
    )
    cleaner = TransactionCleaner(config)
    df, _ = cleaner.clean(raw)
    return df


class TestFeatureEngineer:
    def test_transform_adds_all_configured_features(self, config, cleaned_df):
        fe = FeatureEngineer(config)
        out = fe.transform(cleaned_df.copy())
        for col in fe.engineered_columns():
            assert col in out.columns, f"missing engineered feature: {col}"
        assert len(fe.engineered_columns()) >= 6  # rubric minimum

    def test_no_nan_or_inf_in_engineered_features(self, config, cleaned_df):
        fe = FeatureEngineer(config)
        out = fe.transform(cleaned_df.copy())
        feats = out[fe.engineered_columns()]
        assert not feats.isna().any().any()
        assert not (feats.abs() == float("inf")).any().any()

    def test_customer_txn_count_is_causal_and_zero_indexed(self, config, cleaned_df):
        fe = FeatureEngineer(config)
        out = fe.transform(cleaned_df.copy())
        user1 = out[out["User"] == 1].sort_values("transaction_datetime")
        assert list(user1["customer_txn_count"]) == [0, 1, 2, 3]

    def test_customer_avg_amount_excludes_current_transaction(self, config, cleaned_df):
        fe = FeatureEngineer(config)
        out = fe.transform(cleaned_df.copy())
        user1 = out[out["User"] == 1].sort_values("transaction_datetime").reset_index(drop=True)
        # Third transaction's avg should be mean of the first two (50, 55), not include itself.
        assert user1.loc[2, "customer_avg_amount"] == pytest.approx(52.5)

    def test_weekend_and_business_hour_indicators_are_binary(self, config, cleaned_df):
        fe = FeatureEngineer(config)
        out = fe.transform(cleaned_df.copy())
        assert set(out["weekend_indicator"].unique()).issubset({0, 1})
        assert set(out["business_hour_indicator"].unique()).issubset({0, 1})

    def test_business_hour_indicator_correctness(self, config, cleaned_df):
        fe = FeatureEngineer(config)
        out = fe.transform(cleaned_df.copy())
        row_9am = out[out["txn_hour"] == 9].iloc[0]
        row_8pm = out[out["txn_hour"] == 20].iloc[0]
        assert row_9am["business_hour_indicator"] == 1
        assert row_8pm["business_hour_indicator"] == 0

    def test_merchant_diversity_increases_only_on_new_merchant(self, config, cleaned_df):
        fe = FeatureEngineer(config)
        out = fe.transform(cleaned_df.copy())
        user1 = out[out["User"] == 1].sort_values("transaction_datetime").reset_index(drop=True)
        # merchants in order: 111, 111, 222, 111 -> diversity-before-row: 0, 1, 1, 2
        assert list(user1["merchant_diversity"]) == [0, 1, 1, 2]

    def test_transaction_velocity_counts_prior_window_only(self, config, cleaned_df):
        fe = FeatureEngineer(config)
        out = fe.transform(cleaned_df.copy())
        user1 = out[out["User"] == 1].sort_values("transaction_datetime").reset_index(drop=True)
        # txns at 09:00 and 09:30 same day are within a 1h window of each other.
        assert user1.loc[1, "transaction_velocity"] >= 1

    def test_missing_required_columns_raises(self, config):
        fe = FeatureEngineer(config)
        with pytest.raises(ValueError):
            fe.transform(pd.DataFrame({"foo": [1, 2, 3]}))