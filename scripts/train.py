#!/usr/bin/env python
"""
CLI entry point: prepare data and train all anomaly detection models.

Usage
-----
    python scripts/train.py
    python scripts/train.py --sample-frac 0.1
    python scripts/train.py --nrows 50000          # quick smoke run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import AnomalyDetectionPipeline  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("scripts.train")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train all anomaly detection models.")
    parser.add_argument(
        "--sample-frac", type=float, default=None,
        help="Fraction of users to sample (overrides configs/config.yaml). Use 1.0 for the full dataset.",
    )
    parser.add_argument(
        "--nrows", type=int, default=None,
        help="Cap on rows loaded (after user sampling) — useful for a fast smoke test.",
    )
    args = parser.parse_args()

    pipeline = AnomalyDetectionPipeline()

    logger.info("=== Stage 1/2: preparing training data ===")
    split = pipeline.prepare_training_data(sample_frac=args.sample_frac, nrows=args.nrows)
    logger.info(
        "Prepared %s train / %s test rows across %d features.",
        f"{len(split.X_train):,}", f"{len(split.X_test):,}", len(split.feature_columns),
    )

    logger.info("=== Stage 2/2: training models ===")
    models = pipeline.train(split.X_train)

    logger.info("Training complete. Models saved to %s", pipeline.config.paths.models_dir)
    for name, model in models.items():
        logger.info("  - %-22s trained in %.2fs", name, model.train_time_seconds_)


if __name__ == "__main__":
    main()