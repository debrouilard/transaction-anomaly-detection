"""
Centralized logging setup for the Transaction Anomaly Detection pipeline.

Every module obtains its logger via `get_logger(__name__)` rather than
calling `logging.basicConfig` itself, so log formatting, levels and file
rotation stay consistent across scripts, tests, and the Streamlit dashboard.

Example
-------
>>> from src.utils.logger import get_logger
>>> logger = get_logger(__name__)
>>> logger.info("Loaded %d rows", 1000)
"""

from __future__ import annotations

import logging
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from src.utils.config import get_config

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_setup_lock = threading.Lock()
_root_configured = False


def _configure_root_logger() -> None:
    """Attach console + rotating file handlers to the package root logger
    ("transaction_anomaly") exactly once per process."""
    global _root_configured
    if _root_configured:
        return

    with _setup_lock:
        if _root_configured:
            return

        try:
            cfg = get_config()
            level_name = cfg.logging.get("level", "INFO")
            log_to_file = cfg.logging.get("log_to_file", True)
            log_filename = cfg.logging.get("log_filename", "pipeline.log")
            max_bytes = cfg.logging.get("max_bytes", 5 * 1024 * 1024)
            backup_count = cfg.logging.get("backup_count", 3)
            logs_dir: Path = cfg.paths.logs_dir
        except Exception:
            # Config not available yet (e.g. very early import cycle) —
            # fall back to sane defaults rather than crashing on import.
            level_name, log_to_file, log_filename = "INFO", True, "pipeline.log"
            max_bytes, backup_count = 5 * 1024 * 1024, 3
            logs_dir = Path("logs")

        root = logging.getLogger("transaction_anomaly")
        root.setLevel(getattr(logging, str(level_name).upper(), logging.INFO))
        root.propagate = False

        if root.handlers:
            _root_configured = True
            return

        formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

        if log_to_file:
            try:
                logs_dir.mkdir(parents=True, exist_ok=True)
                file_handler = RotatingFileHandler(
                    logs_dir / log_filename,
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                    encoding="utf-8",
                )
                file_handler.setFormatter(formatter)
                root.addHandler(file_handler)
            except OSError:
                # e.g. read-only filesystem (some deployment targets) —
                # console logging alone is an acceptable degradation.
                root.warning("Could not create log file at %s; console-only logging.", logs_dir)

        _root_configured = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a logger namespaced under 'transaction_anomaly', configuring the
    shared console/file handlers on first use.
    """
    _configure_root_logger()
    full_name = "transaction_anomaly" if not name else f"transaction_anomaly.{name}"
    return logging.getLogger(full_name)