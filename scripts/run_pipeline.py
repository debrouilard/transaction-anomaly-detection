#!/usr/bin/env python
"""
CLI entry point: run the entire project end-to-end in one command —
data preparation, feature engineering, training all models, and full
evaluation/comparison. This is what `README.md` refers to as the
one-command reproduction path.

Usage
-----
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --sample-frac 1.0     # full 24M-row dataset
    python scripts/run_pipeline.py --nrows 20000          # fast smoke test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.compare_models import run_comparison  # noqa: E402
from src.evaluation.evaluate import evaluate_all_models  # noqa: E402
from src.pipeline import AnomalyDetectionPipeline  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("scripts.run_pipeline")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full anomaly detection pipeline end-to-end.")
    parser.add_argument("--sample-frac", type=float, default=None)
    parser.add_argument("--nrows", type=int, default=None)
    args = parser.parse_args()

    overall_start = time.perf_counter()
    pipeline = AnomalyDetectionPipeline()

    logger.info("############################################################")
    logger.info("STAGE 1/4 — Data preparation (load, validate, clean, engineer, split, encode)")
    logger.info("############################################################")
    split = pipeline.prepare_training_data(sample_frac=args.sample_frac, nrows=args.nrows)
    logger.info(
        "Prepared %s train / %s test rows, %d engineered features.",
        f"{len(split.X_train):,}", f"{len(split.X_test):,}", len(split.feature_columns),
    )

    logger.info("############################################################")
    logger.info("STAGE 2/4 — Model training (Isolation Forest, LOF, One-Class SVM, Autoencoder)")
    logger.info("############################################################")
    models = pipeline.train(split.X_train)

    logger.info("############################################################")
    logger.info("STAGE 3/4 — Evaluation against the held-out fraud label")
    logger.info("############################################################")
    results = evaluate_all_models(models, split.X_test, split.y_test.values, pipeline.config)

    logger.info("############################################################")
    logger.info("STAGE 4/4 — Model comparison & recommendation")
    logger.info("############################################################")
    comparison = run_comparison(results, pipeline.config)

    elapsed = time.perf_counter() - overall_start
    logger.info("############################################################")
    logger.info("PIPELINE COMPLETE in %.1fs", elapsed)
    logger.info("Recommended model: %s", comparison["recommendation"]["recommended_model"])
    logger.info("Reason: %s", comparison["recommendation"]["reason"])
    logger.info(
        "Artifacts: models/*.pkl,*.pt | reports/metrics.json | reports/model_comparison.csv | "
        "reports/recommendation.json | reports/roc_pr_curves.json"
    )
    logger.info("Next: `streamlit run dashboard/app.py` to explore results interactively.")
    logger.info("############################################################")


if __name__ == "__main__":
    main()