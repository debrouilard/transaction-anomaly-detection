#!/usr/bin/env python
"""
CLI entry point: score a CSV of new/raw transactions with a trained model
and write the results (with anomaly_score, anomaly_score_normalized, and
is_anomaly columns appended) to data/prediction/.

Usage
-----
    python scripts/predict.py --input data/raw/sample.csv --model isolation_forest
    python scripts/predict.py --input new_transactions.csv --model autoencoder --output flagged.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.train_models import MODEL_REGISTRY  # noqa: E402
from src.pipeline import AnomalyDetectionPipeline  # noqa: E402
from src.utils.helpers import ensure_dir  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("scripts.predict")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score new transactions for anomalies with a trained model.")
    parser.add_argument("--input", type=str, required=True, help="Path to a CSV of raw transactions (same schema as the training data).")
    parser.add_argument("--model", type=str, default="isolation_forest", choices=list(MODEL_REGISTRY.keys()), help="Which trained model to score with.")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path. Defaults to data/prediction/<input_stem>_<model>_scored.csv")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    pipeline = AnomalyDetectionPipeline()
    pipeline.load_models()

    logger.info("Loading input transactions from %s", input_path)
    raw_df = pd.read_csv(input_path, low_memory=False)

    logger.info("Scoring %s rows with model '%s'...", f"{len(raw_df):,}", args.model)
    scored_df = pipeline.score(raw_df, model_name=args.model)

    output_path = Path(args.output) if args.output else (
        pipeline.config.paths.prediction_dir / f"{input_path.stem}_{args.model}_scored.csv"
    )
    ensure_dir(output_path.parent)
    scored_df.to_csv(output_path, index=False)

    n_flagged = int(scored_df["is_anomaly"].sum())
    logger.info(
        "Scoring complete: %d / %s transactions flagged as anomalous (%.2f%%). Results written to %s",
        n_flagged, f"{len(scored_df):,}", 100 * n_flagged / len(scored_df), output_path,
    )


if __name__ == "__main__":
    main()