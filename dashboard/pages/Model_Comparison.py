"""Streamlit page: Model Comparison — metrics, curves, and runtime
comparison across all trained models."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import get_config  # noqa: E402
from src.utils.helpers import load_json  # noqa: E402
from src.visualization import dashboard_charts as charts  # noqa: E402

cfg = get_config()
st.set_page_config(page_title="Model Comparison", page_icon="⚖️", layout="wide")
st.title("⚖️ Model Comparison")

if not cfg.paths.metrics_path.exists():
    st.warning(
        "No evaluation results found yet. Run `python scripts/evaluate.py` "
        "(after `python scripts/train.py`) from the project root first."
    )
    st.stop()


@st.cache_data(show_spinner="Loading evaluation results...")
def _load_results():
    metrics = load_json(cfg.paths.metrics_path)
    comparison_csv_path = cfg.paths.reports_dir / "model_comparison.csv"
    comparison_df = pd.read_csv(comparison_csv_path) if comparison_csv_path.exists() else pd.DataFrame(metrics.values())
    curves = load_json(cfg.paths.reports_dir / "roc_pr_curves.json") if (cfg.paths.reports_dir / "roc_pr_curves.json").exists() else {}
    try:
        recommendation = load_json(cfg.paths.reports_dir / "recommendation.json")
    except FileNotFoundError:
        recommendation = {}
    return metrics, comparison_df, curves, recommendation


metrics, comparison_df, curves, recommendation = _load_results()

if recommendation.get("recommended_model"):
    st.success(
        f"**Final recommendation: `{recommendation['recommended_model']}`** — {recommendation.get('reason', '')}"
    )

st.subheader("Metrics Table")
display_cols = [
    c for c in [
        "model", "roc_auc", "average_precision", "precision_at_50", "precision_at_100",
        "recall_at_100", "f1", "precision", "recall", "n_flagged", "n_actual_fraud",
        "train_time_seconds", "prediction_time_seconds",
    ] if c in comparison_df.columns
]
st.dataframe(
    comparison_df[display_cols].style.highlight_max(
        subset=[c for c in ["roc_auc", "average_precision", "f1"] if c in display_cols], color="lightgreen"
    ),
    use_container_width=True,
)

st.divider()
st.subheader("Ranking Quality")
metric_choice = st.selectbox(
    "Metric to compare", ["roc_auc", "average_precision", "f1", "precision_at_100", "recall_at_100"], index=0
)
if metric_choice in comparison_df.columns:
    st.plotly_chart(charts.model_comparison_bar_chart(comparison_df, metric_choice), use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    if curves:
        st.plotly_chart(charts.roc_curves_chart(curves), use_container_width=True)
with col2:
    if curves:
        st.plotly_chart(charts.pr_curves_chart(curves), use_container_width=True)

st.divider()
st.subheader("Runtime Comparison")
st.plotly_chart(charts.model_runtime_chart(comparison_df), use_container_width=True)

st.divider()
st.subheader("Qualitative Comparison")
qual_cols = ["model", "ease_of_implementation", "computational_cost", "strengths", "weaknesses", "business_usefulness"]
qual_cols = [c for c in qual_cols if c in comparison_df.columns]
if len(qual_cols) > 1:
    for _, row in comparison_df.iterrows():
        with st.expander(f"**{row['model']}**"):
            for col in qual_cols[1:]:
                st.markdown(f"**{col.replace('_', ' ').title()}:** {row[col]}")

st.caption(
    "ROC-AUC / Average Precision / precision & recall@k are computed against the held-out "
    "`Is Fraud?` label for evaluation only — this label was never used to fit any model."
)