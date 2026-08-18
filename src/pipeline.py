"""
`AnomalyDetectionPipeline` — the single orchestrator that ties data
loading, validation, cleaning, feature engineering, encoding, model
training, and inference-time feature preparation together. `scripts/*.py`
are thin CLI wrappers around this class; the Streamlit dashboard uses it
directly for live scoring, so training and serving can never drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.loader import load_raw_data, save_dataframe, load_dataframe
from src.data.preprocessing import FeatureEncoder, TransactionCleaner
from src.data.validator import DataValidator
from src.features.feature_engineering import FeatureEngineer
from src.models.base import BaseAnomalyModel
from src.models.train_models import load_all_models, train_all_models
from src.utils.config import Config, get_config
from src.utils.helpers import set_seed, Timer
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SplitData:
    """Container for a train/test split, keeping the raw engineered
    dataframe, the encoded feature matrices, and the (evaluation-only)
    fraud labels together so nothing gets mismatched downstream."""

    train_df: pd.DataFrame
    test_df: pd.DataFrame
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: pd.Series
    y_test: pd.Series
    feature_columns: List[str]


class AnomalyDetectionPipeline:
    """End-to-end orchestrator for the transaction anomaly detection
    system. Typical usage:

    >>> pipeline = AnomalyDetectionPipeline()
    >>> split = pipeline.prepare_training_data()
    >>> models = pipeline.train(split.X_train)
    >>> pipeline.save_engineered_data(split)
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        set_seed(self.config.project.get("random_seed", 42))

        self.validator = DataValidator(self.config)
        self.cleaner = TransactionCleaner(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.encoder = FeatureEncoder(self.config)
        self.models: Dict[str, BaseAnomalyModel] = {}

    # ------------------------------------------------------------------ #
    # Data stages
    # ------------------------------------------------------------------ #

    def load_data(self, sample_frac: Optional[float] = None, nrows: Optional[int] = None) -> pd.DataFrame:
        return load_raw_data(self.config, sample_frac=sample_frac, nrows=nrows)

    def validate(self, df: pd.DataFrame):
        report = self.validator.validate(df)
        self.validator.save_report(report)
        return report

    def clean_and_engineer(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Run cleaning + behavioral feature engineering. Returns
        (engineered_df, y) where `y` is the evaluation-only fraud label,
        never merged back into the feature dataframe."""
        with Timer("clean_and_engineer"):
            clean_df, y = self.cleaner.clean(df)
            engineered_df = self.feature_engineer.transform(clean_df)
        return engineered_df, y

    def split(self, engineered_df: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        test_size = self.config.preprocessing.get("test_size", 0.2)
        seed = self.config.project.get("random_seed", 42)
        stratify = y if y.sum() > 0 and y.sum() < len(y) else None
        train_df, test_df, y_train, y_test = train_test_split(
            engineered_df, y, test_size=test_size, random_state=seed, stratify=stratify
        )
        return (
            train_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
            y_train.reset_index(drop=True),
            y_test.reset_index(drop=True),
        )

    def encode(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Fit the FeatureEncoder on the training split only, transform
        both splits, and return the numeric matrices the models consume."""
        numeric_cols = self.feature_engineer.engineered_columns() + [self.config.data_schema.amount_column]
        self.encoder.fit(train_df, numeric_cols)
        train_encoded = self.encoder.transform(train_df)
        test_encoded = self.encoder.transform(test_df)

        feature_cols = self.encoder.get_feature_columns()
        X_train = train_encoded[feature_cols].values
        X_test = test_encoded[feature_cols].values
        return X_train, X_test, feature_cols

    def prepare_training_data(
        self, sample_frac: Optional[float] = None, nrows: Optional[int] = None
    ) -> SplitData:
        """Run the full data stage: load -> validate -> clean -> engineer
        -> split -> fit encoder -> encode. Persists the encoder for reuse
        at inference time."""
        raw_df = self.load_data(sample_frac=sample_frac, nrows=nrows)
        self.validate(raw_df)
        engineered_df, y = self.clean_and_engineer(raw_df)

        save_dataframe(
            pd.concat([engineered_df, y.rename("is_fraud")], axis=1),
            self.config.paths.engineered_data,
        )

        train_df, test_df, y_train, y_test = self.split(engineered_df, y)
        X_train, X_test, feature_cols = self.encode(train_df, test_df)
        self.encoder.save()

        return SplitData(train_df, test_df, X_train, X_test, y_train, y_test, feature_cols)

    # ------------------------------------------------------------------ #
    # Model stages
    # ------------------------------------------------------------------ #

    def train(self, X_train: np.ndarray, save: bool = True) -> Dict[str, BaseAnomalyModel]:
        self.models = train_all_models(X_train, self.config, save=save)
        return self.models

    def load_models(self) -> Dict[str, BaseAnomalyModel]:
        self.models = load_all_models(self.config)
        return self.models

    # ------------------------------------------------------------------ #
    # Inference (shared by scripts/predict.py and the dashboard)
    # ------------------------------------------------------------------ #

    def prepare_inference_features(self, raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Apply the exact same cleaning + feature engineering + encoding
        used in training to new/unseen raw transactions, using the
        *persisted* encoder (loaded if not already in memory). Returns the
        engineered dataframe (for display) and the numeric feature matrix
        (for scoring)."""
        if not self.encoder.is_fitted:
            self.encoder = FeatureEncoder.load(config=self.config)

        clean_df, _ = self.cleaner.clean(raw_df.copy())
        engineered_df = self.feature_engineer.transform(clean_df)
        encoded_df = self.encoder.transform(engineered_df)
        feature_cols = self.encoder.get_feature_columns()
        X = encoded_df[feature_cols].values
        return engineered_df, X

    def score(self, raw_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
        """Score new transactions with a named model; returns the
        engineered dataframe with `anomaly_score`, `anomaly_score_normalized`,
        and `is_anomaly` columns appended."""
        if not self.models:
            self.load_models()
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' is not loaded/trained. Available: {list(self.models)}")

        engineered_df, X = self.prepare_inference_features(raw_df)
        model = self.models[model_name]

        result = engineered_df.copy()
        result["anomaly_score"] = model.score_samples(X)
        result["anomaly_score_normalized"] = model.predict_proba(X)
        result["is_anomaly"] = model.predict(X)
        return result

    # ------------------------------------------------------------------ #
    # Full run
    # ------------------------------------------------------------------ #

    def run_full_pipeline(self, sample_frac: Optional[float] = None) -> Dict[str, Any]:
        """Convenience method used by scripts/run_pipeline.py: data prep +
        training in one call. Evaluation is intentionally kept in
        `scripts/evaluate.py` / `src/evaluation/` so this method never
        needs to know about the fraud label beyond passing it through."""
        split = self.prepare_training_data(sample_frac=sample_frac)
        models = self.train(split.X_train)
        return {
            "split": split,
            "models": models,
            "n_train": len(split.X_train),
            "n_test": len(split.X_test),
            "n_features": len(split.feature_columns),
        }