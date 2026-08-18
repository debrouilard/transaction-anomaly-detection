"""
Generic, dependency-light helper functions shared across the pipeline:
reproducibility seeding, execution timing, JSON/joblib persistence, and
small numeric utilities used by more than one module.
"""

from __future__ import annotations

import functools
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar, Union

import joblib
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def set_seed(seed: int = 42) -> None:
    """Seed python's `random`, numpy, and (if importable) torch, for
    reproducible experiments across the whole pipeline."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    logger.debug("Global random seed set to %d", seed)


def timer(func: F) -> F:
    """Decorator that logs the wall-clock execution time of the wrapped
    function and, when the wrapped function returns a dict, injects an
    `elapsed_seconds` key so callers can persist timing alongside results."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info("%s completed in %.3fs", func.__qualname__, elapsed)
        if isinstance(result, dict) and "elapsed_seconds" not in result:
            result["elapsed_seconds"] = round(elapsed, 4)
        return result

    return wrapper  # type: ignore[return-value]


class Timer:
    """Context-manager alternative to the `timer` decorator, for timing
    arbitrary code blocks (e.g. a section inside a larger function).

    >>> with Timer("feature engineering") as t:
    ...     do_work()
    >>> t.elapsed_seconds
    """

    def __init__(self, label: str = "block"):
        self.label = label
        self.elapsed_seconds: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.elapsed_seconds = round(time.perf_counter() - self._start, 4)
        logger.info("[%s] completed in %.3fs", self.label, self.elapsed_seconds)


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create `path` (and parents) if it doesn't exist; return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Dict[str, Any], path: Union[str, Path], indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, default=_json_default)
    logger.debug("Saved JSON artifact to %s", path)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(obj: Any) -> Any:
    """Best-effort serializer for numpy scalar/array types that otherwise
    aren't JSON-serializable."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_artifact(obj: Any, path: Union[str, Path]) -> None:
    """Persist any picklable object (models, scalers, encoders) via joblib."""
    path = Path(path)
    ensure_dir(path.parent)
    joblib.dump(obj, path)
    logger.info("Saved artifact to %s", path)


def load_artifact(path: Union[str, Path]) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}. Run the training pipeline first.")
    return joblib.load(path)


def safe_divide(numerator: np.ndarray, denominator: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Element-wise division that substitutes `fill` wherever the
    denominator is zero, instead of raising / producing inf or nan."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denominator != 0, numerator / denominator, fill)
    return result


def format_bytes(num_bytes: int) -> str:
    """Human-readable byte size, used in logging dataset/model sizes."""
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"