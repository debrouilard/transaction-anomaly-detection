#!/usr/bin/env python
"""
CLI entry point: load persisted models + the held-out test split, evaluate
every model against the fraud label (evaluation-only use, per project
design), and write reports/metrics.json, reports/model_comparison.csv,
reports/roc_pr_curves.json, and reports/recommendation.json.

Usage
-----
    python scripts/evaluate.py
    python scripts/evaluate.py --sample-frac 0.1   # re-derive the same split used for training
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.compare_models import run_comparison  # noqa: E402
from src.evaluation.evaluate import evaluate_all_models  # noqa: E402
from src.pipeline import AnomalyDetectionPipeline  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("scripts.evaluate")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate and compare all trained anomaly detection models.")
    parser.add_argument("--sample-frac", type=float, default=None, help="Must match the fraction used in scripts/train.py for a consistent split.")
    parser.add_argument("--nrows", type=int, default=None)
    args = parser.parse_args()

    pipeline = AnomalyDetectionPipeline()

    logger.info("=== Preparing data (re-running the same deterministic split as training) ===")
    split = pipeline.prepare_training_data(sample_frac=args.sample_frac, nrows=args.nrows)

    logger.info("=== Loading trained models ===")
    models = pipeline.load_models()
    if not models:
        raise RuntimeError(
            "No trained models found in models/. Run `python scripts/train.py` first."
        )

    logger.info("=== Evaluating %d model(s) against the held-out fraud label ===", len(models))
    results = evaluate_all_models(models, split.X_test, split.y_test.values, pipeline.config)

    logger.info("=== Building comparison table and recommendation ===")
    comparison = run_comparison(results, pipeline.config)

    print("\n" + "=" * 100)
    print("MODEL COMPARISON")
    print("=" * 100)
    display_cols = [c for c in ["model", "roc_auc", "average_precision", "precision_at_100", "recall_at_100", "f1", "train_time_seconds", "prediction_time_seconds"] if c in comparison["table"].columns]
    print(comparison["table"][display_cols].to_string(index=False))
    print("=" * 100)
    print(f"RECOMMENDATION: {comparison['recommendation']['recommended_model']}")
    print(f"REASON: {comparison['recommendation']['reason']}")
    print("=" * 100)
    logger.info("Full metrics written to %s", pipeline.config.paths.metrics_path)


if __name__ == "__main__":
    main()