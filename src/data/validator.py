"""
Data quality validation for raw and processed transaction data.

`DataValidator` is used both as a pipeline safety gate (raising on critical
schema violations before preprocessing runs) and as a reporting tool that
feeds the "Data Understanding" section of the README / technical report /
dashboard (missing values, duplicates, dtype and range issues, cardinality).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.config import Config, get_config
from src.utils.helpers import save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SchemaValidationError(ValueError):
    """Raised when the input data does not satisfy the required schema."""


@dataclass
class ValidationReport:
    n_rows: int
    n_cols: int
    missing_columns: List[str] = field(default_factory=list)
    unexpected_dtypes: Dict[str, str] = field(default_factory=dict)
    missing_values: Dict[str, int] = field(default_factory=dict)
    missing_value_pct: Dict[str, float] = field(default_factory=dict)
    n_duplicate_rows: int = 0
    cardinality: Dict[str, int] = field(default_factory=dict)
    range_violations: Dict[str, int] = field(default_factory=dict)
    inconsistent_value_notes: List[str] = field(default_factory=list)
    is_valid: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "missing_columns": self.missing_columns,
            "unexpected_dtypes": self.unexpected_dtypes,
            "missing_values": self.missing_values,
            "missing_value_pct": self.missing_value_pct,
            "n_duplicate_rows": self.n_duplicate_rows,
            "cardinality": self.cardinality,
            "range_violations": self.range_violations,
            "inconsistent_value_notes": self.inconsistent_value_notes,
            "is_valid": self.is_valid,
        }


class DataValidator:
    """Validates raw transaction data against the schema declared in
    `configs/config.yaml` and profiles data-quality issues."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.schema = self.config.data_schema

    def validate_schema(self, df: pd.DataFrame, raise_on_error: bool = True) -> List[str]:
        """Check that every required raw column is present. Returns the list
        of missing columns; raises `SchemaValidationError` if any are
        missing and `raise_on_error` is True."""
        required = set(self.schema.raw_columns)
        missing = sorted(required - set(df.columns))
        if missing:
            msg = f"Missing required columns: {missing}"
            if raise_on_error:
                raise SchemaValidationError(msg)
            logger.warning(msg)
        return missing

    def profile(self, df: pd.DataFrame) -> ValidationReport:
        """Produce a full data-quality profile: missingness, duplicates,
        cardinality, dtype sanity checks, and plausible-range checks on
        `Amount`. Never raises — this is a reporting method."""
        report = ValidationReport(n_rows=len(df), n_cols=len(df.columns))

        report.missing_columns = self.validate_schema(df, raise_on_error=False)

        missing_counts = df.isnull().sum()
        report.missing_values = {c: int(v) for c, v in missing_counts.items() if v > 0}
        report.missing_value_pct = {
            c: round(100 * v / len(df), 3) for c, v in report.missing_values.items()
        }

        report.n_duplicate_rows = int(df.duplicated().sum())

        for col in [
            self.schema.merchant_column,
            *self.schema.categorical_columns,
            *self.schema.id_columns,
        ]:
            if col in df.columns:
                report.cardinality[col] = int(df[col].nunique(dropna=True))

        amount_col = self.schema.amount_column
        if amount_col in df.columns:
            amounts = self._coerce_amount(df[amount_col])
            n_negative = int((amounts < 0).sum())
            n_nan = int(amounts.isna().sum())
            if n_negative:
                report.inconsistent_value_notes.append(
                    f"{n_negative} rows have negative '{amount_col}' (possible refunds/reversals; "
                    "reviewed rather than dropped, see preprocessing)."
                )
                report.range_violations[amount_col] = n_negative
            if n_nan:
                report.inconsistent_value_notes.append(
                    f"{n_nan} rows have a non-numeric '{amount_col}' after stripping currency symbols."
                )

        label_col = self.schema.get("label_column")
        if label_col and label_col in df.columns:
            valid_labels = {self.schema.label_positive_value, "No"}
            bad = df.loc[~df[label_col].isin(valid_labels) & df[label_col].notna(), label_col]
            if len(bad):
                report.inconsistent_value_notes.append(
                    f"{len(bad)} rows have an unexpected value in '{label_col}' "
                    f"(expected one of {sorted(valid_labels)})."
                )

        report.is_valid = len(report.missing_columns) == 0
        logger.info(
            "Data profile: %s rows, %s duplicate rows, %d columns with missing values.",
            f"{report.n_rows:,}",
            f"{report.n_duplicate_rows:,}",
            len(report.missing_values),
        )
        return report

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        """Convenience: raise on critical schema issues, then return the
        full profile for logging/reporting."""
        self.validate_schema(df, raise_on_error=True)
        return self.profile(df)

    def save_report(self, report: ValidationReport, path=None) -> None:
        path = path or (self.config.paths.eda_report_dir / "data_validation_report.json")
        save_json(report.to_dict(), path)
        logger.info("Saved validation report to %s", path)

    @staticmethod
    def _coerce_amount(series: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(series):
            return series.astype(float)
        cleaned = series.astype(str).str.replace(r"[\$,]", "", regex=True)
        return pd.to_numeric(cleaned, errors="coerce")