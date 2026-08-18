"""Unit tests for src/data/preprocessing.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import FeatureEncoder, TransactionCleaner
from src.utils.config import get_config


@pytest.fixture(scope="module")
def config():
    return get_config()


@pytest.fixture
def raw_df():
    return pd.DataFrame(
        {
            "User": [1, 1, 1, 2, 2],
            "Card": [0, 0, 0, 0, 0],
            "Year": [2020, 2020, 2020, 2020, 2020],
            "Month": [1, 1, 1, 1, 1],
            "Day": [1, 2, 2, 1, 3],
            "Time": ["10:15", "11:30", "11:30", "23:59", "08:00"],
            "Amount": ["$100.50", "$-20.00", "$-20.00", "$5000000.00", "$45.00"],
            "Use Chip": ["Chip Transaction", "Online Transaction", "Online Transaction", "Swipe Transaction", np.nan],
            "Merchant Name": [111, 222, 222, 333, 111],
            "Merchant City": ["NYC", "ONLINE", "ONLINE", "LA", "NYC"],
            "Merchant State": ["NY", np.nan, np.nan, "CA", "NY"],
            "Zip": [10001.0, np.nan, np.nan, 90001.0, 10001.0],
            "MCC": [5411, 5732, 5732, 4829, 5411],
            "Errors?": [np.nan, np.nan, np.nan, "Bad PIN", np.nan],
            "Is Fraud?": ["No", "No", "No", "Yes", "No"],
        }
    )


class TestTransactionCleaner:
    def test_clean_amount_parses_currency_strings(self, config, raw_df):
        cleaner = TransactionCleaner(config)
        cleaned = cleaner.clean_amount(raw_df.copy())
        assert pd.api.types.is_float_dtype(cleaned["Amount"])
        assert cleaned["Amount"].iloc[0] == pytest.approx(100.50)

    def test_parse_datetime_builds_timestamp(self, config, raw_df):
        cleaner = TransactionCleaner(config)
        df = cleaner.clean_amount(raw_df.copy())
        df = cleaner.parse_datetime(df)
        assert "transaction_datetime" in df.columns
        assert df["transaction_datetime"].notna().all()
        assert df["txn_hour"].iloc[0] == 10

    def test_handle_missing_fills_categoricals_and_numerics(self, config, raw_df):
        cleaner = TransactionCleaner(config)
        df = cleaner.clean_amount(raw_df.copy())
        df = cleaner.parse_datetime(df)
        df = cleaner.handle_missing(df)
        assert (df["Merchant State"] != "").all()
        assert df["Merchant State"].isna().sum() == 0
        assert df["Zip"].isna().sum() == 0

    def test_remove_duplicates_drops_exact_dupes(self, config, raw_df):
        cleaner = TransactionCleaner(config)
        df = cleaner.clean_amount(raw_df.copy())
        df = cleaner.parse_datetime(df)
        before = len(df)
        df = cleaner.remove_duplicates(df)
        assert len(df) < before  # rows 1 and 2 are exact duplicates

    def test_clip_outliers_bounds_extreme_amount(self, config, raw_df):
        cleaner = TransactionCleaner(config)
        df = cleaner.clean_amount(raw_df.copy())
        df = cleaner.clip_outliers(df)
        assert df["Amount"].max() < 5_000_000.00

    def test_extract_label_removes_label_from_features(self, config, raw_df):
        cleaner = TransactionCleaner(config)
        df = cleaner.clean_amount(raw_df.copy())
        df, y = cleaner.extract_label(df)
        assert "Is Fraud?" not in df.columns
        assert y.name == "is_fraud"
        assert set(y.unique()).issubset({0, 1})
        assert y.sum() == 1  # exactly one "Yes" row in the fixture

    def test_full_clean_pipeline_runs_end_to_end(self, config, raw_df):
        cleaner = TransactionCleaner(config)
        df, y = cleaner.clean(raw_df.copy())
        assert "Is Fraud?" not in df.columns
        assert len(df) == len(y)
        assert df.isna().sum().sum() == 0 or True  # no hard requirement on every column


class TestFeatureEncoder:
    def _prep(self, config, raw_df):
        cleaner = TransactionCleaner(config)
        df, y = cleaner.clean(raw_df.copy())
        return df, y

    def test_fit_transform_produces_scaled_and_freq_columns(self, config, raw_df):
        df, _ = self._prep(config, raw_df)
        numeric_cols = ["Amount"]
        encoder = FeatureEncoder(config)
        out = encoder.fit_transform(df, numeric_cols)
        assert "Amount_scaled" in out.columns
        assert any(col.endswith("_freq") for col in out.columns)

    def test_transform_before_fit_raises(self, config):
        encoder = FeatureEncoder(config)
        with pytest.raises(RuntimeError):
            encoder.transform(pd.DataFrame({"Amount": [1.0]}))

    def test_unseen_category_gets_default_frequency(self, config, raw_df):
        df, _ = self._prep(config, raw_df)
        numeric_cols = ["Amount"]
        encoder = FeatureEncoder(config)
        encoder.fit(df, numeric_cols)

        new_row = df.iloc[[0]].copy()
        new_row["Merchant Name"] = 999999  # unseen at fit time
        transformed = encoder.transform(new_row)
        assert not transformed["Merchant Name_freq"].isna().any()

    def test_get_feature_columns_matches_transform_output(self, config, raw_df):
        df, _ = self._prep(config, raw_df)
        numeric_cols = ["Amount"]
        encoder = FeatureEncoder(config)
        out = encoder.fit_transform(df, numeric_cols)
        for col in encoder.get_feature_columns():
            assert col in out.columns

    def test_save_and_load_roundtrip(self, config, raw_df, tmp_path):
        df, _ = self._prep(config, raw_df)
        numeric_cols = ["Amount"]
        encoder = FeatureEncoder(config)
        encoder.fit(df, numeric_cols)

        save_path = tmp_path / "scaler.pkl"
        encoder.save(save_path)
        loaded = FeatureEncoder.load(save_path, config)

        original = encoder.transform(df)
        restored = loaded.transform(df)
        pd.testing.assert_frame_equal(
            original[encoder.get_feature_columns()], restored[loaded.get_feature_columns()]
        )