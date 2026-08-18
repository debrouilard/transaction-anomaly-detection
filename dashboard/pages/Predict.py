"""Streamlit page: Predict — upload a CSV of new transactions and score
them live with any trained model."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.explainability.explain import AnomalyExplainer  # noqa: E402
from src.models.train_models import MODEL_REGISTRY  # noqa: E402
from src.pipeline import AnomalyDetectionPipeline  # noqa: E402
from src.utils.config import get_config  # noqa: E402
from src.visualization import dashboard_charts as charts  # noqa: E402

cfg = get_config()
st.set_page_config(page_title="Predict", page_icon="🔎", layout="wide")
st.title("🔎 Live Anomaly Prediction")


@st.cache_resource(show_spinner="Loading pipeline and trained models...")
def _get_pipeline() -> AnomalyDetectionPipeline:
    pipeline = AnomalyDetectionPipeline(cfg)
    pipeline.load_models()
    return pipeline


available_models = [name for name in MODEL_REGISTRY if (cfg.paths.models_dir / MODEL_REGISTRY[name][2]).exists()]
if not available_models:
    st.warning(
        "No trained models found. Run `python scripts/train.py` from the project root first."
    )
    st.stop()

pipeline = _get_pipeline()

st.markdown(
    "Upload a CSV of raw transactions with the same columns as the training data "
    "(`User, Card, Year, Month, Day, Time, Amount, Use Chip, Merchant Name, Merchant City, "
    "Merchant State, Zip, MCC, Errors?`; `Is Fraud?` is optional and ignored for scoring)."
)

col_a, col_b = st.columns([2, 1])
with col_a:
    uploaded_file = st.file_uploader("Upload transactions CSV", type=["csv"])
with col_b:
    model_choice = st.selectbox("Model", available_models, index=available_models.index(cfg.dashboard.get("default_model", available_models[0])) if cfg.dashboard.get("default_model") in available_models else 0)

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file, low_memory=False)
    max_rows = cfg.dashboard.get("max_upload_rows", 200_000)
    if len(raw_df) > max_rows:
        st.warning(f"File has {len(raw_df):,} rows; only the first {max_rows:,} will be scored.")
        raw_df = raw_df.head(max_rows)

    with st.spinner(f"Scoring {len(raw_df):,} transactions with {model_choice}..."):
        try:
            scored_df = pipeline.score(raw_df, model_name=model_choice)
        except Exception as exc:  # surface a clear, actionable error rather than a stack trace
            st.error(f"Scoring failed: {exc}")
            st.stop()

    n_flagged = int(scored_df["is_anomaly"].sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Transactions scored", f"{len(scored_df):,}")
    c2.metric("Flagged as anomalous", f"{n_flagged:,}")
    c3.metric("Anomaly rate", f"{100 * n_flagged / len(scored_df):.2f}%")

    st.subheader("Prediction Results")
    show_only_anomalies = st.checkbox("Show only flagged transactions", value=True)
    display_df = scored_df[scored_df["is_anomaly"] == 1] if show_only_anomalies else scored_df
    display_df = display_df.sort_values("anomaly_score", ascending=False)
    st.dataframe(display_df, use_container_width=True, height=400)

    st.download_button(
        "⬇️ Download prediction results (CSV)",
        data=scored_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_{model_choice}_scored.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("Explore Flagged Transactions")
    feature_cols_for_plot = [
        c for c in ["Amount", "customer_avg_amount", "amount_deviation_from_customer_mean", "transaction_velocity", "merchant_diversity"]
        if c in scored_df.columns
    ]
    if len(feature_cols_for_plot) >= 2:
        c1, c2 = st.columns(2)
        x_feature = c1.selectbox("X-axis feature", feature_cols_for_plot, index=0)
        y_feature = c2.selectbox("Y-axis feature", feature_cols_for_plot, index=1)
        st.plotly_chart(charts.anomaly_scatter_chart(scored_df, x_feature, y_feature), use_container_width=True)

    if n_flagged > 0:
        st.subheader("Explain a Specific Flagged Transaction")
        flagged_indices = scored_df[scored_df["is_anomaly"] == 1].index.tolist()
        selected_idx = st.selectbox("Select a flagged transaction (row index)", flagged_indices)

        engineered_df, X = pipeline.prepare_inference_features(raw_df)
        model = pipeline.models[model_choice]
        feature_names = pipeline.encoder.get_feature_columns()

        explainer = AnomalyExplainer(model, feature_names)
        explanation = explainer.explain_instance(X[selected_idx], X, top_n=5)

        st.info(explanation["business_explanation"])
        st.dataframe(pd.DataFrame(explanation["top_contributing_features"]), use_container_width=True)
else:
    st.info("Upload a CSV to get started, or run `python scripts/predict.py --input <file> --model <name>` from the command line for batch scoring.")