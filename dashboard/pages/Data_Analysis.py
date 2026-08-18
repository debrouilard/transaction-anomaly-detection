"""Streamlit page: Data Analysis — interactive EDA on the engineered
transaction dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.loader import load_dataframe  # noqa: E402
from src.data.validator import DataValidator  # noqa: E402
from src.utils.config import get_config  # noqa: E402
from src.visualization import dashboard_charts as charts  # noqa: E402

cfg = get_config()
st.set_page_config(page_title="Data Analysis", page_icon="📊", layout="wide")
st.title("📊 Data Analysis")


@st.cache_data(show_spinner="Loading engineered dataset...")
def _load_engineered() -> pd.DataFrame:
    return load_dataframe(cfg.paths.engineered_data)


if not cfg.paths.engineered_data.exists():
    st.warning(
        "No engineered dataset found yet. Run `python scripts/train.py` or "
        "`python scripts/run_pipeline.py` from the project root first."
    )
    st.stop()

df = _load_engineered()

st.subheader("Dataset Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Columns", f"{df.shape[1]}")
c3.metric("Unique customers", f"{df['User'].nunique():,}")
c4.metric("Unique merchants", f"{df['Merchant Name'].nunique():,}")

with st.expander("Descriptive statistics", expanded=False):
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

with st.expander("Missing values", expanded=False):
    validator = DataValidator(cfg)
    report = validator.profile(df)
    if report.missing_values:
        st.dataframe(
            pd.DataFrame(
                {"column": list(report.missing_values), "missing_count": list(report.missing_values.values())}
            ),
            use_container_width=True,
        )
    else:
        st.success("No missing values in the engineered dataset.")

st.divider()

# --- Filters -----------------------------------------------------------
st.sidebar.header("Filters")
users = sorted(df["User"].unique().tolist())
selected_users = st.sidebar.multiselect("Filter by customer (User ID)", users, default=[])
filtered_df = df[df["User"].isin(selected_users)] if selected_users else df

st.subheader("Transaction Amount")
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(charts.amount_distribution_chart(filtered_df), use_container_width=True)
with col2:
    numeric_cols = [
        "Amount", "customer_avg_amount", "customer_max_amount", "rolling_avg_amount",
        "amount_deviation_from_customer_mean", "amount_zscore_by_merchant",
        "transaction_velocity", "merchant_diversity",
    ]
    numeric_cols = [c for c in numeric_cols if c in filtered_df.columns]
    st.plotly_chart(charts.correlation_heatmap_chart(filtered_df, numeric_cols), use_container_width=True)

st.subheader("Temporal Patterns")
col1, col2, col3 = st.columns(3)
with col1:
    st.plotly_chart(charts.hourly_pattern_chart(filtered_df), use_container_width=True)
with col2:
    st.plotly_chart(charts.weekly_trend_chart(filtered_df), use_container_width=True)
with col3:
    st.plotly_chart(charts.daily_trend_chart(filtered_df), use_container_width=True)

st.subheader("Merchant & Customer Behavior")
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(charts.merchant_distribution_chart(filtered_df), use_container_width=True)
with col2:
    freq = filtered_df["User"].value_counts()
    st.plotly_chart(
        px.histogram(
            x=freq.values, nbins=40, template="plotly_white",
            title="Customer Transaction Frequency Distribution",
            labels={"x": "Transactions per customer"},
        ),
        use_container_width=True,
    )

st.caption(
    "Business interpretation: peak activity hours and weekday/weekend splits calibrate "
    "expectations for 'normal' timing (feeding `business_hour_indicator` / "
    "`weekend_indicator`); merchant/customer concentration motivates the frequency- and "
    "customer-relative features used across all four models."
)