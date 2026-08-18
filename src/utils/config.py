"""
Configuration loader for the Transaction Anomaly Detection pipeline.

Loads configs/config.yaml and configs/model_config.yaml, resolves every
relative path against the project root (so scripts behave identically
regardless of the working directory they're invoked from), and exposes a
single cached `get_config()` accessor used throughout the codebase.

Example
-------
>>> from src.utils.config import get_config
>>> cfg = get_config()
>>> cfg.paths.raw_data
PosixPath('/abs/path/to/project/data/raw/credit_card_transactions.csv')
>>> cfg["preprocessing"]["scaler"]
'standard'
"""

from __future__ import annotations

import copy
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class ConfigNode:
    """
    Read-only, dot-accessible wrapper around a nested dict.

    Supports both attribute access (`cfg.paths.raw_data`) and dict-style
    access (`cfg["paths"]["raw_data"]`) so it's convenient in scripts while
    still behaving predictably (e.g. with `.get()`, `in`, iteration).
    """

    def __init__(self, data: Dict[str, Any]):
        object.__setattr__(self, "_data", data)

    def __getattr__(self, item: str) -> Any:
        try:
            value = self._data[item]
        except KeyError as exc:
            raise AttributeError(
                f"Config has no key '{item}'. Available keys: {list(self._data.keys())}"
            ) from exc
        return self._wrap(value)

    def __getitem__(self, item: str) -> Any:
        return self._wrap(self._data[item])

    def __contains__(self, item: str) -> bool:
        return item in self._data

    def __repr__(self) -> str:
        return f"ConfigNode({self._data!r})"

    def __iter__(self):
        return iter(self._data)

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self._data:
            return default
        return self._wrap(self._data[key])

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._data)

    @staticmethod
    def _wrap(value: Any) -> Any:
        if isinstance(value, dict):
            return ConfigNode(value)
        return value


class Config(ConfigNode):
    """
    Top-level configuration object. Merges config.yaml and model_config.yaml
    under `paths`, `data_schema`, ... and `models` (model_config.yaml is
    nested under the `models` key) respectively, and resolves every entry
    under `paths` into an absolute `pathlib.Path` rooted at the project root.
    """

    def __init__(self, data: Dict[str, Any], project_root: Path):
        super().__init__(data)
        object.__setattr__(self, "project_root", project_root)

    def resolve(self, relative_path: str) -> Path:
        """Resolve a path string relative to the project root."""
        p = Path(relative_path)
        return p if p.is_absolute() else (self.project_root / p)


def find_project_root(start: Optional[Path] = None) -> Path:
    """
    Walk upward from `start` (default: this file's directory) until a
    directory containing both `configs/` and `src/` is found. Falls back to
    two levels up from this file (src/utils/ -> project root) if no marker
    is found, which covers the standard layout even when invoked from an
    unusual working directory.
    """
    current = (start or Path(__file__).resolve().parent)
    for candidate in [current, *current.parents]:
        if (candidate / "configs").is_dir() and (candidate / "src").is_dir():
            return candidate
    # Fallback: src/utils/config.py -> src/utils -> src -> project root
    return Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        content = yaml.safe_load(f) or {}
    if not isinstance(content, dict):
        raise ValueError(f"Configuration file {path} did not parse to a mapping/dict.")
    return content


def _build_config(project_root: Path) -> Config:
    main_cfg = _load_yaml(project_root / "configs" / "config.yaml")
    model_cfg = _load_yaml(project_root / "configs" / "model_config.yaml")

    merged: Dict[str, Any] = copy.deepcopy(main_cfg)
    merged["models"] = model_cfg

    # Resolve every path in `paths:` to an absolute pathlib.Path up front so
    # downstream code never has to think about the working directory.
    resolved_paths = {}
    for key, rel in merged.get("paths", {}).items():
        p = Path(rel)
        resolved_paths[key] = p if p.is_absolute() else (project_root / p)
    merged["paths"] = resolved_paths

    return Config(merged, project_root)


_config_lock = threading.Lock()
_config_instance: Optional[Config] = None


def get_config(force_reload: bool = False, project_root: Optional[Path] = None) -> Config:
    """
    Return the cached, process-wide Config instance, building it on first
    call. Thread-safe. Pass `force_reload=True` to re-read the YAML files
    (useful in tests) or `project_root` to override auto-detection.
    """
    global _config_instance
    with _config_lock:
        if _config_instance is None or force_reload:
            root = project_root or find_project_root()
            _config_instance = _build_config(root)
        return _config_instance