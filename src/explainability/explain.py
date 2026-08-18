"""
Explainability for the anomaly detection models.

Provides three complementary views, all model-agnostic except where noted:

1. `permutation_importance` — global feature importance for *any* model
   (works identically for Isolation Forest, LOF, OCSVM, and the
   Autoencoder) by measuring how much shuffling a feature perturbs the
   model's anomaly scores.
2. `explain_instance` — per-transaction explanation: which features
   deviate most from the reference (training) distribution for this
   specific row, in human-readable business language.
3. `autoencoder_reconstruction_breakdown` — Autoencoder-specific: which
   individual features the network reconstructed worst for a given row,
   the most direct explanation available for a reconstruction-error-based
   anomaly score.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.models.autoencoder import AutoencoderModel
from src.models.base import BaseAnomalyModel
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Human-readable descriptions for the engineered features + core inputs,
# used to turn a ranked list of "most deviant features" into a business
# explanation an analyst can act on without knowing the underlying stats.
FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "Amount": "the raw transaction amount",
    "customer_avg_amount": "how it compares to this customer's typical spending",
    "customer_max_amount": "how it compares to the largest amount this customer has spent before",
    "customer_txn_count": "this customer's overall transaction history length",
    "merchant_txn_frequency": "how common/rare this merchant is across all transactions",
    "avg_daily_spending": "this customer's typical daily spending level",
    "weekend_indicator": "whether the transaction occurred on a weekend",
    "business_hour_indicator": "whether the transaction occurred during typical business hours",
    "time_since_prev_txn": "how long it had been since this customer's previous transaction",
    "rolling_avg_amount": "how it compares to this customer's recent (last few) transaction amounts",
    "transaction_velocity": "how many transactions this customer made in a short recent window",
    "merchant_diversity": "how many distinct merchants this customer has used before",
    "customer_activity_frequency": "how often this customer transacts relative to their active days",
    "amount_deviation_from_customer_mean": "how far the amount deviates from this customer's typical spending, in standard deviations",
    "amount_zscore_by_merchant": "how far the amount deviates from this merchant's typical transaction size, in standard deviations",
}


def _base_feature_name(encoded_col: str) -> str:
    """Map an encoded column name (e.g. 'Amount_scaled', 'MCC_freq') back
    to its underlying feature name for lookup in FEATURE_DESCRIPTIONS."""
    for suffix in ("_scaled", "_freq"):
        if encoded_col.endswith(suffix):
            return encoded_col[: -len(suffix)]
    return encoded_col


class AnomalyExplainer:
    """Explainability toolkit wrapping a single fitted model + the
    feature-column names it was trained on."""

    def __init__(self, model: BaseAnomalyModel, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names

    def permutation_importance(
        self, X: np.ndarray, n_repeats: int = 5, sample_size: Optional[int] = 5000, random_state: int = 42
    ) -> pd.DataFrame:
        """Global feature importance: for each feature, shuffle its values
        across rows and measure the average absolute change in anomaly
        score. A feature the model relies on heavily will cause a large
        score shift when scrambled; an irrelevant feature won't."""
        rng = np.random.default_rng(random_state)
        X = np.asarray(X, dtype=float)
        if sample_size is not None and len(X) > sample_size:
            idx = rng.choice(len(X), size=sample_size, replace=False)
            X = X[idx]

        baseline = self.model.score_samples(X)
        importances = np.zeros(X.shape[1])

        for j in range(X.shape[1]):
            deltas = []
            for _ in range(n_repeats):
                X_perturbed = X.copy()
                X_perturbed[:, j] = rng.permutation(X_perturbed[:, j])
                perturbed_scores = self.model.score_samples(X_perturbed)
                deltas.append(np.mean(np.abs(perturbed_scores - baseline)))
            importances[j] = float(np.mean(deltas))

        df = pd.DataFrame({"feature": self.feature_names, "importance": importances})
        df["base_feature"] = df["feature"].apply(_base_feature_name)
        df["description"] = df["base_feature"].map(FEATURE_DESCRIPTIONS).fillna("")
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def explain_instance(
        self,
        x_row: np.ndarray,
        X_reference: np.ndarray,
        top_n: int = 5,
    ) -> Dict[str, Any]:
        """Explain a single transaction by ranking its features by
        |z-score| against the reference (training) distribution — a
        model-agnostic, always-available explanation even for models
        without a native feature-attribution method (Isolation Forest,
        LOF, OCSVM)."""
        x_row = np.asarray(x_row, dtype=float).reshape(-1)
        X_reference = np.asarray(X_reference, dtype=float)

        ref_mean = X_reference.mean(axis=0)
        ref_std = X_reference.std(axis=0)
        ref_std_safe = np.where(ref_std < 1e-9, 1.0, ref_std)
        z_scores = (x_row - ref_mean) / ref_std_safe

        order = np.argsort(-np.abs(z_scores))[:top_n]
        top_features = []
        for idx in order:
            feature_name = self.feature_names[idx]
            base_name = _base_feature_name(feature_name)
            top_features.append(
                {
                    "feature": feature_name,
                    "z_score": round(float(z_scores[idx]), 3),
                    "value": round(float(x_row[idx]), 4),
                    "reference_mean": round(float(ref_mean[idx]), 4),
                    "description": FEATURE_DESCRIPTIONS.get(base_name, base_name),
                }
            )

        anomaly_score = float(self.model.score_samples(x_row.reshape(1, -1))[0])
        return {
            "anomaly_score": anomaly_score,
            "is_anomaly": bool(anomaly_score >= self.model.threshold_) if self.model.threshold_ is not None else None,
            "top_contributing_features": top_features,
            "business_explanation": self._to_business_language(top_features),
        }

    @staticmethod
    def _to_business_language(top_features: List[Dict[str, Any]]) -> str:
        if not top_features:
            return "No feature deviated meaningfully from typical behavior."
        clauses = []
        for feat in top_features[:3]:
            direction = "much higher than usual" if feat["z_score"] > 0 else "much lower than usual"
            clauses.append(f"{feat['description']} ({direction}, z={feat['z_score']:+.2f})")
        return "This transaction is flagged primarily due to: " + "; ".join(clauses) + "."


def autoencoder_reconstruction_breakdown(
    model: AutoencoderModel, x_row: np.ndarray, feature_names: List[str], top_n: int = 5
) -> Dict[str, Any]:
    """Autoencoder-specific explanation: which individual features the
    network reconstructed worst for this transaction — the most direct
    possible explanation for a reconstruction-error-based anomaly score."""
    x_row = np.asarray(x_row, dtype=float).reshape(1, -1)
    per_feature_error = model.reconstruction_error_per_feature(x_row)[0]

    order = np.argsort(-per_feature_error)[:top_n]
    breakdown = [
        {
            "feature": feature_names[idx],
            "description": FEATURE_DESCRIPTIONS.get(_base_feature_name(feature_names[idx]), feature_names[idx]),
            "reconstruction_error": round(float(per_feature_error[idx]), 6),
        }
        for idx in order
    ]
    total_error = float(per_feature_error.sum())
    return {
        "total_reconstruction_error": total_error,
        "top_contributing_features": breakdown,
        "business_explanation": (
            "The Autoencoder struggled most to reconstruct: "
            + "; ".join(f"{b['description']}" for b in breakdown[:3])
            + " — indicating this transaction's combination of behavioral features "
            "doesn't match patterns the model learned as 'normal'."
        ),
    }