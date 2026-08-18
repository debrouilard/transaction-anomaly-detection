"""
Interactive (Plotly) chart builders used exclusively by the Streamlit
dashboard (`dashboard/`). Kept separate from `src/visualization/plots.py`
(matplotlib, used by notebooks/PDF report) because Streamlit renders
Plotly figures natively with zoom/hover/filter interactivity, which static
images can't offer — but both modules cover the same analytical ground so
notebook and dashboard findings stay consistent.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

_TEMPLATE = "plotly_white"


def amount_distribution_chart(df: pd.DataFrame, amount_col: str = "Amount") -> go.Figure:
    fig = px.histogram(
        df, x=amount_col, nbins=60, template=_TEMPLATE,
        title="Transaction Amount Distribution", labels={amount_col: "Amount ($)"},
    )
    fig.update_layout(bargap=0.02)
    return fig


def correlation_heatmap_chart(df: pd.DataFrame, numeric_columns: Optional[List[str]] = None) -> go.Figure:
    cols = numeric_columns or df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[cols].corr()
    fig = px.imshow(
        corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, template=_TEMPLATE,
        title="Feature Correlation Heatmap", aspect="auto",
    )
    return fig


def hourly_pattern_chart(df: pd.DataFrame, hour_col: str = "txn_hour") -> go.Figure:
    hourly = df[hour_col].value_counts().sort_index().reset_index()
    hourly.columns = ["hour", "count"]
    fig = px.line(
        hourly, x="hour", y="count", markers=True, template=_TEMPLATE,
        title="Transaction Volume by Hour of Day",
    )
    fig.update_xaxes(dtick=2)
    return fig


def daily_trend_chart(df: pd.DataFrame, datetime_col: str = "transaction_datetime") -> go.Figure:
    daily = df[datetime_col].dt.date.value_counts().sort_index().reset_index()
    daily.columns = ["date", "count"]
    fig = px.line(daily, x="date", y="count", template=_TEMPLATE, title="Daily Transaction Volume")
    return fig


def weekly_trend_chart(df: pd.DataFrame, day_of_week_col: str = "txn_day_of_week") -> go.Figure:
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekly = df[day_of_week_col].value_counts().sort_index()
    weekly.index = [day_names[i] for i in weekly.index]
    fig = px.bar(
        x=weekly.index, y=weekly.values, template=_TEMPLATE,
        title="Transaction Volume by Day of Week", labels={"x": "Day", "y": "Count"},
    )
    return fig


def merchant_distribution_chart(df: pd.DataFrame, merchant_col: str = "Merchant Name", top_n: int = 20) -> go.Figure:
    top = df[merchant_col].value_counts().head(top_n).reset_index()
    top.columns = ["merchant", "count"]
    fig = px.bar(
        top, x="count", y="merchant", orientation="h", template=_TEMPLATE,
        title=f"Top {top_n} Merchants by Transaction Count",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return fig


def model_comparison_bar_chart(comparison_df: pd.DataFrame, metric: str = "roc_auc") -> go.Figure:
    ordered = comparison_df.sort_values(metric, ascending=True)
    fig = px.bar(
        ordered, x=metric, y="model", orientation="h", template=_TEMPLATE,
        title=f"Model Comparison — {metric.replace('_', ' ').upper()}",
        color=metric, color_continuous_scale="Blues",
    )
    return fig


def model_runtime_chart(comparison_df: pd.DataFrame) -> go.Figure:
    melted = comparison_df.melt(
        id_vars="model",
        value_vars=[c for c in ["train_time_seconds", "prediction_time_seconds"] if c in comparison_df.columns],
        var_name="stage", value_name="seconds",
    )
    fig = px.bar(
        melted, x="model", y="seconds", color="stage", barmode="group", template=_TEMPLATE,
        title="Training vs. Prediction Time by Model",
    )
    return fig


def roc_curves_chart(curves: Dict[str, Dict]) -> go.Figure:
    """Overlay ROC curves for every model on one chart, from the
    `reports/roc_pr_curves.json` structure produced by
    `src/evaluation/evaluate.py`."""
    fig = go.Figure()
    for model_name, curve_data in curves.items():
        roc = curve_data.get("roc", {})
        if roc.get("fpr"):
            fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name=model_name))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random baseline", line=dict(dash="dash", color="gray")))
    fig.update_layout(
        template=_TEMPLATE, title="ROC Curves — All Models",
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
    )
    return fig


def pr_curves_chart(curves: Dict[str, Dict]) -> go.Figure:
    fig = go.Figure()
    for model_name, curve_data in curves.items():
        pr = curve_data.get("pr", {})
        if pr.get("precision"):
            fig.add_trace(go.Scatter(x=pr["recall"], y=pr["precision"], mode="lines", name=model_name))
    fig.update_layout(
        template=_TEMPLATE, title="Precision-Recall Curves — All Models",
        xaxis_title="Recall", yaxis_title="Precision",
    )
    return fig


def anomaly_score_distribution_chart(scores: np.ndarray, threshold: Optional[float] = None, model_name: str = "") -> go.Figure:
    fig = px.histogram(x=scores, nbins=60, template=_TEMPLATE, title=f"Anomaly Score Distribution — {model_name}")
    fig.update_xaxes(title="Anomaly score (higher = more anomalous)")
    if threshold is not None:
        fig.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text="Decision threshold")
    return fig


def anomaly_scatter_chart(df: pd.DataFrame, x: str, y: str, anomaly_col: str = "is_anomaly") -> go.Figure:
    """2D scatter of two chosen features, colored by anomaly flag — the
    core interactive exploration chart on the Predict page."""
    plot_df = df.copy()
    plot_df[anomaly_col] = plot_df[anomaly_col].map({0: "Normal", 1: "Anomaly"})
    fig = px.scatter(
        plot_df, x=x, y=y, color=anomaly_col, template=_TEMPLATE,
        color_discrete_map={"Normal": "#4C72B0", "Anomaly": "#C44E52"},
        title=f"{x} vs. {y}, colored by anomaly flag", opacity=0.7,
    )
    return fig


def feature_importance_chart(importance_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    top = importance_df.head(top_n).sort_values("importance")
    fig = px.bar(
        top, x="importance", y="feature", orientation="h", template=_TEMPLATE,
        title="Permutation Feature Importance", hover_data=["description"],
    )
    return fig


def mcc_anomaly_rate_chart(df: pd.DataFrame, mcc_col: str = "MCC", anomaly_col: str = "is_anomaly", top_n: int = 15) -> go.Figure:
    """Business-insights chart: anomaly rate by merchant category code —
    highlights which categories warrant tighter review thresholds."""
    grouped = df.groupby(mcc_col)[anomaly_col].agg(["mean", "count"]).reset_index()
    grouped = grouped[grouped["count"] >= 5].sort_values("mean", ascending=False).head(top_n)
    grouped[mcc_col] = grouped[mcc_col].astype(str)
    fig = px.bar(
        grouped, x="mean", y=mcc_col,
        orientation="h", template=_TEMPLATE, title=f"Anomaly Rate by {mcc_col} (min. 5 transactions)",
        labels={"mean": "Anomaly rate"},
    )
    return fig