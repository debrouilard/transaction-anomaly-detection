"""Streamlit page: Business Insights — anomaly patterns by merchant,
category, and time, plus an estimated business-impact summary."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.loader import load_dataframe  # noqa: E402
from src.models.train_models import MODEL_REGISTRY  # noqa: E402
from src.pipeline import AnomalyDetectionPipeline  # noqa: E402
from src.utils.config import get_config  # noqa: E402
from src.visualization import dashboard_charts as charts  # noqa: E402

cfg = get_config()
st.set_page_config(page_title="Business Insights", page_icon="💡", layout="wide")
st.title("💡 Business Insights")

if not cfg.paths.engineered_data.exists():
    st.warning("No engineered dataset found yet. Run `python scripts/run_pipeline.py` first.")
    st.stop()

available_models = [name for name in MODEL_REGISTRY if (cfg.paths.models_dir / MODEL_REGISTRY[name][2]).exists()]
if not available_models:
    st.warning("No trained models found. Run `python scripts/train.py` first.")
    st.stop()


@st.cache_resource(show_spinner="Loading pipeline and models...")
def _get_pipeline() -> AnomalyDetectionPipeline:
    pipeline = AnomalyDetectionPipeline(cfg)
    pipeline.load_models()
    return pipeline


@st.cache_data(show_spinner="Scoring the full engineered dataset for business analysis...")
def _score_full_dataset(model_name: str) -> pd.DataFrame:
    engineered = load_dataframe(cfg.paths.engineered_data)
    pipeline = _get_pipeline()
    if not pipeline.encoder.is_fitted:
        from src.data.preprocessing import FeatureEncoder
        pipeline.encoder = FeatureEncoder.load(config=cfg)
    feature_cols = pipeline.encoder.get_feature_columns()
    encoded = pipeline.encoder.transform(engineered)
    X = encoded[feature_cols].values
    model = pipeline.models[model_name]
    result = engineered.copy()
    result["anomaly_score"] = model.score_samples(X)
    result["is_anomaly"] = model.predict(X)
    return result


model_choice = st.selectbox(
    "Model used for this analysis", available_models,
    index=available_models.index(cfg.dashboard.get("default_model")) if cfg.dashboard.get("default_model") in available_models else 0,
)
scored_df = _score_full_dataset(model_choice)

n_flagged = int(scored_df["is_anomaly"].sum())
flagged_amount = scored_df.loc[scored_df["is_anomaly"] == 1, "Amount"].sum()
total_amount = scored_df["Amount"].sum()

st.subheader("Impact Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total transactions", f"{len(scored_df):,}")
c2.metric("Flagged transactions", f"{n_flagged:,}", f"{100 * n_flagged / len(scored_df):.2f}%")
c3.metric("Flagged transaction value", f"${flagged_amount:,.0f}")
c4.metric("Share of total transaction value", f"{100 * flagged_amount / total_amount:.2f}%")

st.caption(
    "\"Flagged transaction value\" is the total dollar amount of transactions the model "
    "marked anomalous — a proxy for exposure an analyst review queue would need to triage, "
    "not a confirmed fraud-loss figure."
)

st.divider()
st.subheader("Where Anomalies Concentrate")
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(charts.mcc_anomaly_rate_chart(scored_df), use_container_width=True)
with col2:
    channel_col = "Use Chip" if "Use Chip" in scored_df.columns else None
    if channel_col:
        rate_by_channel = scored_df.groupby(channel_col)["is_anomaly"].mean().sort_values(ascending=False).reset_index()
        import plotly.express as px
        fig = px.bar(
            rate_by_channel, x=channel_col, y="is_anomaly", template="plotly_white",
            title="Anomaly Rate by Transaction Channel", labels={"is_anomaly": "Anomaly rate"},
        )
        st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    hourly_rate = scored_df.groupby("txn_hour")["is_anomaly"].mean().reset_index()
    import plotly.express as px2
    st.plotly_chart(
        px2.line(hourly_rate, x="txn_hour", y="is_anomaly", markers=True, template="plotly_white",
                  title="Anomaly Rate by Hour of Day", labels={"is_anomaly": "Anomaly rate", "txn_hour": "Hour"}),
        use_container_width=True,
    )
with col2:
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekly_rate = scored_df.groupby("txn_day_of_week")["is_anomaly"].mean().reset_index()
    weekly_rate["day"] = weekly_rate["txn_day_of_week"].apply(lambda d: day_names[int(d)])
    import plotly.express as px3
    st.plotly_chart(
        px3.bar(weekly_rate, x="day", y="is_anomaly", template="plotly_white",
                title="Anomaly Rate by Day of Week", labels={"is_anomaly": "Anomaly rate"}),
        use_container_width=True,
    )

st.divider()
st.subheader("Top Flagged Merchants")
merchant_summary = (
    scored_df[scored_df["is_anomaly"] == 1]
    .groupby("Merchant Name")
    .agg(flagged_count=("is_anomaly", "size"), total_flagged_amount=("Amount", "sum"))
    .sort_values("flagged_count", ascending=False)
    .head(15)
    .reset_index()
)
st.dataframe(merchant_summary, use_container_width=True)

st.info(
    "**Business recommendation:** prioritize analyst review queues by MCC/channel "
    "combinations with both high anomaly rate *and* high transaction volume — a rare "
    "category with a 100% anomaly rate on 2 transactions is lower priority than a "
    "common category with a 5% rate across thousands."
)