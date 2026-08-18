"""
Data loading for the IBM/Kaggle Credit Card Transactions dataset.

The full raw file has ~24M rows, so a naive `pd.read_csv` is wasteful during
iteration. `load_raw_data` samples a configurable fraction of *users* (not
rows) up front — preserving each sampled user's complete transaction
history, which every downstream behavioral/rolling feature depends on — and
then reads the file in chunks, keeping only rows for sampled users.

Processed/engineered dataframes are persisted as Parquet (fast, typed,
much smaller than CSV) via `save_dataframe` / `load_dataframe`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from src.utils.config import Config, get_config
from src.utils.helpers import Timer, ensure_dir, format_bytes
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _sample_user_ids(raw_path: Path, id_column: str, sample_frac: float, seed: int) -> np.ndarray:
    """Read only the user-id column (cheap, single-column scan) and return a
    reproducible random subset of unique user ids covering `sample_frac` of
    the user population."""
    logger.info("Scanning '%s' for unique user ids (this reads one column only)...", id_column)
    users = pd.read_csv(raw_path, usecols=[id_column])[id_column].unique()
    rng = np.random.default_rng(seed)
    n_keep = max(1, int(round(len(users) * sample_frac)))
    sampled = rng.choice(users, size=n_keep, replace=False)
    logger.info(
        "Sampling %d / %d users (%.1f%%) for this run.", n_keep, len(users), 100 * sample_frac
    )
    return sampled


def load_raw_data(
    config: Optional[Config] = None,
    sample_frac: Optional[float] = None,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load the raw transactions CSV described by `configs/config.yaml`.

    Parameters
    ----------
    config : Config, optional
        Defaults to `get_config()`.
    sample_frac : float, optional
        Fraction of *users* to keep (overrides `data_loading.sample_frac`).
        Pass 1.0 (or set it in config) to load the full dataset.
    nrows : int, optional
        Hard cap on rows read (mainly for tests / smoke runs); applied after
        user-sampling, on the concatenated result.

    Returns
    -------
    pd.DataFrame with the raw schema columns from `configs/config.yaml`.
    """
    cfg = config or get_config()
    raw_path: Path = cfg.paths.raw_data
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {raw_path}. Download it from "
            "https://www.kaggle.com/datasets/ealtman2019/credit-card-transactions "
            "and place it at this path, or update `paths.raw_data` in configs/config.yaml."
        )

    schema = cfg.data_schema
    id_column = schema.id_columns[0]  # "User"
    frac = sample_frac if sample_frac is not None else cfg.data_loading.get("sample_frac", 1.0)
    chunksize = cfg.data_loading.get("chunksize", 500_000)
    seed = cfg.project.get("random_seed", 42)

    with Timer("load_raw_data"):
        if frac is not None and frac < 1.0:
            sampled_users = set(_sample_user_ids(raw_path, id_column, frac, seed))
            chunks = []
            total_rows = 0
            for chunk in pd.read_csv(raw_path, chunksize=chunksize, low_memory=False):
                filtered = chunk[chunk[id_column].isin(sampled_users)]
                if not filtered.empty:
                    chunks.append(filtered)
                    total_rows += len(filtered)
                if nrows is not None and total_rows >= nrows:
                    break
            if not chunks:
                raise ValueError(
                    "User-sampling produced zero rows — check `data_loading.sample_frac` "
                    "and that the id column values match between the id scan and the full read."
                )
            df = pd.concat(chunks, ignore_index=True)
        else:
            logger.info("Loading full raw dataset (sample_frac >= 1.0 or unset).")
            df = pd.read_csv(raw_path, nrows=nrows, low_memory=False)

    if nrows is not None:
        df = df.head(nrows)

    missing_cols = set(schema.raw_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Loaded data is missing expected columns: {sorted(missing_cols)}. "
            "Verify this is the correct dataset / that the CSV header wasn't altered."
        )

    logger.info(
        "Loaded raw data: %s rows x %s cols (%s in memory)",
        f"{len(df):,}",
        len(df.columns),
        format_bytes(df.memory_usage(deep=True).sum()),
    )
    return df[schema.raw_columns]


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Persist a dataframe as Parquet, creating parent directories as needed."""
    path = Path(path)
    ensure_dir(path.parent)
    df.to_parquet(path, index=False)
    logger.info("Saved dataframe (%s rows) to %s", f"{len(df):,}", path)


def load_dataframe(path: Path) -> pd.DataFrame:
    """Load a Parquet dataframe previously written by `save_dataframe`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Expected processed data at {path} but it does not exist. "
            "Run the earlier pipeline stage (e.g. `python scripts/train.py` or "
            "`python scripts/run_pipeline.py`) first."
        )
    df = pd.read_parquet(path)
    logger.info("Loaded dataframe (%s rows) from %s", f"{len(df):,}", path)
    return df


def chunked_reader(path: Path, chunksize: int = 500_000) -> Iterable[pd.DataFrame]:
    """Thin wrapper around `pd.read_csv(..., chunksize=...)` for callers
    (e.g. batch prediction on very large upload files) that need to stream
    a CSV rather than load it fully into memory."""
    yield from pd.read_csv(path, chunksize=chunksize, low_memory=False)