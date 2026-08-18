"""
Combines the quantitative metrics from `src/evaluation/evaluate.py` with a
structured qualitative comparison (ease of implementation, computational
cost, strengths/weaknesses, business usefulness) and produces a final,
metrics-justified model recommendation — feeding the "Model Comparison"
dashboard page, the technical report, and the presentation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from src.evaluation.evaluate import metrics_to_dataframe
from src.utils.config import Config, get_config
from src.utils.helpers import save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Static qualitative profile for each model — written from first-hand
# engineering experience building this project (not derived from the
# runtime metrics), used to enrich the comparison table with context
# numbers alone don't capture.
QUALITATIVE_PROFILE: Dict[str, Dict[str, Any]] = {
    "isolation_forest": {
        "ease_of_implementation": "Low effort — few hyperparameters, no scaling requirement in theory (though we scale anyway for consistency), scikit-learn API.",
        "computational_cost": "Low. Near-linear in n_estimators x n_samples; trivially parallelizable across trees.",
        "strengths": "Scales to millions of rows; robust to irrelevant/noisy features; no distance computation so unaffected by the curse of dimensionality; fast train and predict.",
        "weaknesses": "Axis-aligned splits can miss anomalies defined by diagonal/rotated feature relationships; less interpretable per-decision than a distance-based score.",
        "business_usefulness": "Strong default choice for a first production deployment — fast enough for near-real-time scoring at transaction volume.",
    },
    "local_outlier_factor": {
        "ease_of_implementation": "Low-to-moderate — must remember `novelty=True` for a usable inference-time `.predict()`; sensitive to `n_neighbors` choice.",
        "computational_cost": "Moderate-to-high. Neighbor search is roughly O(n log n) to O(n^2) depending on dimensionality/algorithm; predicting on new data requires distances to the training set.",
        "strengths": "Captures *local* density anomalies (e.g. unusual-for-this-customer, not unusual-globally) that global methods miss.",
        "weaknesses": "Performance degrades in high dimensions; prediction cost grows with training-set size (all neighbors must be kept in memory).",
        "business_usefulness": "Valuable as a complementary signal to Isolation Forest, especially for customer-specific behavioral deviations.",
    },
    "one_class_svm": {
        "ease_of_implementation": "Moderate — kernel/gamma/nu tuning is less intuitive than tree-based `contamination`; requires explicit subsampling to stay tractable.",
        "computational_cost": "High. RBF-kernel SVM is O(n^2)-O(n^3) in training-set size; we subsample for tractability (see model docstring).",
        "strengths": "Flexible non-linear decision boundary; well-studied theoretical guarantees (nu bounds).",
        "weaknesses": "Doesn't scale to full production data volume without subsampling or approximation (e.g. SGDOneClassSVM); boundary quality is sensitive to gamma.",
        "business_usefulness": "Useful as a research/comparison baseline and for periodic batch scoring on samples rather than full-volume real-time scoring.",
    },
    "autoencoder": {
        "ease_of_implementation": "Higher effort — requires framework code (PyTorch), architecture/training hyperparameter tuning, and GPU access is helpful (not required) at scale.",
        "computational_cost": "Moderate to train (mini-batch SGD, early stopping); very fast to score once trained (a single forward pass).",
        "strengths": "Learns non-linear feature *interactions* jointly across all inputs; reconstruction error is a natural, continuous anomaly score; per-feature error supports explainability.",
        "weaknesses": "More hyperparameters and less interpretable than tree/kernel methods; needs enough data to avoid overfitting the reconstruction to noise; requires careful early stopping.",
        "business_usefulness": "Strongest candidate when anomalies are expected to arise from *combinations* of features rather than any single one — complements Isolation Forest well in an ensemble.",
    },
    "elliptic_envelope": {
        "ease_of_implementation": "Low — few hyperparameters, fast scikit-learn fit.",
        "computational_cost": "Low for the sample sizes used here; scales less gracefully to very high-dimensional feature sets.",
        "strengths": "Fast, interpretable (Mahalanobis distance from a robust center); good baseline for roughly unimodal, elliptical data.",
        "weaknesses": "Assumes a single Gaussian/elliptical 'normal' cluster — a poor fit for multi-segment customer populations, which is why it's an optional/bonus model here rather than a required one.",
        "business_usefulness": "Useful as a fast sanity-check baseline rather than a primary production model for this dataset.",
    },
}


def build_comparison_table(results: Dict[str, Dict], config: Optional[Config] = None) -> pd.DataFrame:
    """Merge quantitative metrics with the static qualitative profile into
    a single comparison dataframe (one row per model)."""
    df = metrics_to_dataframe(results)
    for field in ["ease_of_implementation", "computational_cost", "strengths", "weaknesses", "business_usefulness"]:
        df[field] = df["model"].map(lambda name: QUALITATIVE_PROFILE.get(name, {}).get(field, ""))
    return df


def recommend_model(results: Dict[str, Dict], config: Optional[Config] = None) -> Dict[str, Any]:
    """Produce a final, metrics-justified recommendation.

    Recommendation logic: rank models by ROC-AUC against the held-out
    fraud label (primary metric, per `configs/model_config.yaml
    evaluation.primary_metric`); among models within 2 percentage points
    of the top ROC-AUC, prefer the one with lower prediction latency,
    since a production fraud-review queue is latency-sensitive. Ties are
    broken by average precision. This logic is deterministic and fully
    reproducible from `reports/metrics.json` — never hand-picked.
    """
    cfg = config or get_config()
    primary_metric = cfg.models.evaluation.get("primary_metric", "roc_auc")

    valid = {
        name: m for name, m in results.items()
        if isinstance(m.get(primary_metric), (int, float)) and m.get(primary_metric) == m.get(primary_metric)
    }
    if not valid:
        return {
            "recommended_model": None,
            "reason": (
                "No model produced a valid "
                f"'{primary_metric}' (label likely had 0 or 1 unique classes in this run's "
                "test split — re-run with a larger sample_frac so the held-out set contains "
                "both fraud and non-fraud examples)."
            ),
        }

    best_score = max(m[primary_metric] for m in valid.values())
    near_best = {
        name: m for name, m in valid.items() if best_score - m[primary_metric] <= 0.02
    }
    recommended = min(
        near_best.items(),
        key=lambda kv: (kv[1].get("prediction_time_seconds", float("inf")), -kv[1].get("average_precision", 0)),
    )[0]

    return {
        "recommended_model": recommended,
        "primary_metric": primary_metric,
        "primary_metric_value": valid[recommended][primary_metric],
        "candidates_within_2pct_of_best": list(near_best.keys()),
        "reason": (
            f"'{recommended}' is within 2 percentage points of the best {primary_metric} "
            f"({best_score:.4f}) among all evaluated models and has the lowest prediction "
            "latency among that group, making it the best fit for near-real-time transaction scoring."
        ),
    }


def run_comparison(
    results: Dict[str, Dict], config: Optional[Config] = None, save: bool = True
) -> Dict[str, Any]:
    """Top-level entry point used by `scripts/evaluate.py`: builds the
    comparison table and the recommendation, persists both."""
    cfg = config or get_config()
    table = build_comparison_table(results, cfg)
    recommendation = recommend_model(results, cfg)

    if save:
        table.to_csv(cfg.paths.reports_dir / "model_comparison.csv", index=False)
        save_json(recommendation, cfg.paths.reports_dir / "recommendation.json")

    logger.info("Final recommendation: %s — %s", recommendation["recommended_model"], recommendation["reason"])
    return {"table": table, "recommendation": recommendation}