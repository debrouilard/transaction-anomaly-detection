"""
Preprocessing for the Credit Card Transactions dataset, split into two
deliberately separate stages:

1. `TransactionCleaner` — stateless cleaning applied identically to every
   row regardless of train/val/test membership: currency parsing, datetime
   construction, missing-value imputation, duplicate removal, and
   winsorizing extreme `Amount` outliers. Safe to run once on the whole
   dataset before any split.

2. `FeatureEncoder` — a *fittable* transformer (frequency-encodes
   categoricals, standard-scales numeric columns) that must be fit on the
   training split only and then applied to validation/test/inference data,
   to avoid leaking distributional information across the split boundary.
   Fitted state (frequency maps + scaler) is persisted via
   `save`/`load` so training and inference always use identical encodings.

Justification for each preprocessing decision is inline as docstrings/
comments and mirrored in the README/technical report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.utils.config import Config, get_config
from src.utils.helpers import load_artifact, save_artifact, save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SCALERS = {"standard": StandardScaler, "robust": RobustScaler, "minmax": MinMaxScaler}


class TransactionCleaner:
    """Stateless cleaning steps applied once to the raw dataframe."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.schema = self.config.data_schema
        self.pp_cfg = self.config.preprocessing

    def clean_amount(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse the currency-formatted `Amount` string (e.g. "$14.57")
        into a float column. Justification: the raw field is a string with
        a leading '$', which every downstream numeric step (scaling,
        feature engineering) requires as a float."""
        col = self.schema.amount_column
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(float)
        else:
            df[col] = (
                df[col].astype(str).str.replace(r"[\$,]", "", regex=True).astype(float)
            )
        return df

    def parse_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Combine Year/Month/Day/Time into a single `transaction_datetime`
        column (and derived hour/day-of-week fields), which every temporal
        feature (business-hour, weekend, rolling/velocity features) needs.
        Justification: the raw parts-based timestamp is not directly usable
        for time-delta or ordering operations."""
        hh_mm = df["Time"].astype(str).str.split(":", expand=True)
        hour = pd.to_numeric(hh_mm[0], errors="coerce").fillna(0).astype(int).clip(0, 23)
        minute = pd.to_numeric(hh_mm[1], errors="coerce").fillna(0).astype(int).clip(0, 59)

        df["transaction_datetime"] = pd.to_datetime(
            dict(
                year=df["Year"].astype(int),
                month=df["Month"].astype(int),
                day=df["Day"].astype(int),
                hour=hour,
                minute=minute,
            ),
            errors="coerce",
        )
        # Rows whose Y/M/D/Time combination is not a valid calendar date
        # (rare data-entry artifacts) get NaT and are dropped, logged.
        n_bad = int(df["transaction_datetime"].isna().sum())
        if n_bad:
            logger.warning("Dropping %d rows with an invalid transaction date/time.", n_bad)
            df = df.loc[df["transaction_datetime"].notna()].copy()

        df["txn_hour"] = df["transaction_datetime"].dt.hour
        df["txn_day_of_week"] = df["transaction_datetime"].dt.dayofweek
        return df

    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values. Justification: `Merchant State`/`Zip` are
        structurally missing for online transactions (no physical location)
        rather than data-quality defects, so categoricals are filled with an
        explicit 'UNKNOWN' sentinel (preserving the "missingness" as
        information the model can use) instead of being dropped; the rare
        numeric gaps use the median, which is robust to the heavy right
        skew of transaction amounts."""
        cat_cols = [c for c in self.schema.categorical_columns if c in df.columns]
        for col in cat_cols:
            df[col] = df[col].fillna("UNKNOWN").astype(str)

        if "Merchant State" in df.columns:
            df["Merchant State"] = df["Merchant State"].fillna("UNKNOWN")
        if "Zip" in df.columns:
            df["Zip"] = df["Zip"].fillna(-1)

        amount_col = self.schema.amount_column
        if df[amount_col].isna().any():
            median_val = df[amount_col].median()
            df[amount_col] = df[amount_col].fillna(median_val)

        return df

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop exact-duplicate rows. Justification: duplicate rows most
        likely reflect an ingestion artifact (e.g. re-sent batch file) —
        keeping them would double-count behavior and bias frequency-based
        features."""
        subset = self.pp_cfg.get("duplicate_subset", None)
        before = len(df)
        df = df.drop_duplicates(subset=subset).reset_index(drop=True)
        removed = before - len(df)
        if removed:
            logger.info("Removed %d duplicate rows (%.3f%%).", removed, 100 * removed / before)
        return df

    def clip_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Winsorize `Amount` at configured quantiles. Justification: a
        handful of extreme values (data artifacts or legitimate but rare
        large purchases) can dominate distance/variance-based models
        (OCSVM, LOF, Autoencoder reconstruction loss) if left unclipped;
        winsorizing preserves rank order while bounding influence, and is
        applied identically at inference time using stored quantile bounds."""
        col = self.schema.amount_column
        lo_q, hi_q = self.pp_cfg.get("amount_outlier_clip_quantiles", [0.001, 0.999])
        lo, hi = df[col].quantile(lo_q), df[col].quantile(hi_q)
        n_clipped = int(((df[col] < lo) | (df[col] > hi)).sum())
        df[col] = df[col].clip(lower=lo, upper=hi)
        if n_clipped:
            logger.info(
                "Winsorized %d Amount values outside [%.2f, %.2f] (q%.3f-q%.3f).",
                n_clipped, lo, hi, lo_q, hi_q,
            )
        return df

    def extract_label(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Split off the fraud label into its own Series and drop it from
        the working dataframe, so it is structurally impossible for it to
        leak into features or model training downstream. The label is
        returned separately for evaluation-only use."""
        label_col = self.schema.get("label_column")
        if label_col and label_col in df.columns:
            y = (df[label_col] == self.schema.label_positive_value).astype(int)
            y.name = "is_fraud"
            df = df.drop(columns=[label_col])
        else:
            y = pd.Series(np.zeros(len(df), dtype=int), name="is_fraud", index=df.index)
            logger.warning("No label column found; returning an all-zero placeholder label.")
        return df, y

    def clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Run the full stateless cleaning sequence and return
        (cleaned_features_df, label_series)."""
        df = df.copy()
        df = self.clean_amount(df)
        df = self.parse_datetime(df)
        df = self.handle_missing(df)
        df = self.remove_duplicates(df)
        df = self.clip_outliers(df)
        df, y = self.extract_label(df)
        logger.info("Cleaning complete: %s rows remaining.", f"{len(df):,}")
        return df, y


class FeatureEncoder:
    """Fittable categorical frequency-encoding + numeric scaling.

    Frequency encoding (rather than one-hot) is used because `Merchant
    Name`/`Merchant City` have tens of thousands of levels — one-hot would
    explode dimensionality and is meaningless for distance-based models
    (LOF, OCSVM) and gradient-based ones (Autoencoder) alike. Frequency
    encoding is a compact, monotonic, well-behaved numeric proxy for
    merchant/category popularity.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.pp_cfg = self.config.preprocessing
        self.categorical_columns: List[str] = list(self.config.data_schema.categorical_columns)
        self.merchant_column: str = self.config.data_schema.merchant_column
        self._freq_maps: Dict[str, Dict[str, float]] = {}
        self.scaler = None
        self.numeric_columns_: List[str] = []
        self.is_fitted: bool = False

    def _freq_encode_column(self, series: pd.Series, freq_map: Dict[str, float]) -> pd.Series:
        default = min(freq_map.values()) if freq_map else 0.0
        return series.astype(str).map(freq_map).fillna(default).astype(float)

    def fit(self, df: pd.DataFrame, numeric_columns: List[str]) -> "FeatureEncoder":
        cols_to_encode = [c for c in [*self.categorical_columns, self.merchant_column] if c in df.columns]
        for col in cols_to_encode:
            counts = df[col].astype(str).value_counts(normalize=True)
            self._freq_maps[col] = counts.to_dict()

        self.numeric_columns_ = list(numeric_columns)
        scaler_name = self.pp_cfg.get("scaler", "standard")
        self.scaler = _SCALERS.get(scaler_name, StandardScaler)()
        self.scaler.fit(df[self.numeric_columns_].values)

        self.is_fitted = True
        logger.info(
            "FeatureEncoder fit on %d categorical columns and %d numeric columns using a %s scaler.",
            len(cols_to_encode), len(self.numeric_columns_), scaler_name,
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("FeatureEncoder.transform() called before fit(). Call fit() first.")
        df = df.copy()
        for col, freq_map in self._freq_maps.items():
            if col in df.columns:
                df[f"{col}_freq"] = self._freq_encode_column(df[col], freq_map)

        missing_numeric = [c for c in self.numeric_columns_ if c not in df.columns]
        if missing_numeric:
            raise ValueError(f"Columns expected by the fitted scaler are missing: {missing_numeric}")

        scaled = self.scaler.transform(df[self.numeric_columns_].values)
        scaled_df = pd.DataFrame(
            scaled, columns=[f"{c}_scaled" for c in self.numeric_columns_], index=df.index
        )
        return pd.concat([df, scaled_df], axis=1)

    def fit_transform(self, df: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
        return self.fit(df, numeric_columns).transform(df)

    def get_feature_columns(self) -> List[str]:
        """Return the final numeric model-input column names produced by
        this encoder (frequency-encoded categoricals + scaled numerics)."""
        freq_cols = [f"{c}_freq" for c in self._freq_maps]
        scaled_cols = [f"{c}_scaled" for c in self.numeric_columns_]
        return freq_cols + scaled_cols

    def save(self, path: Optional[Path] = None) -> None:
        path = path or self.config.paths.scaler_path
        save_artifact(
            {
                "freq_maps": self._freq_maps,
                "scaler": self.scaler,
                "numeric_columns": self.numeric_columns_,
                "categorical_columns": self.categorical_columns,
                "merchant_column": self.merchant_column,
            },
            path,
        )
        save_json(
            {"feature_columns": self.get_feature_columns()},
            self.config.paths.feature_list_path,
        )

    @classmethod
    def load(cls, path: Optional[Path] = None, config: Optional[Config] = None) -> "FeatureEncoder":
        cfg = config or get_config()
        path = path or cfg.paths.scaler_path
        state = load_artifact(path)
        encoder = cls(cfg)
        encoder._freq_maps = state["freq_maps"]
        encoder.scaler = state["scaler"]
        encoder.numeric_columns_ = state["numeric_columns"]
        encoder.categorical_columns = state["categorical_columns"]
        encoder.merchant_column = state["merchant_column"]
        encoder.is_fitted = True
        return encoder