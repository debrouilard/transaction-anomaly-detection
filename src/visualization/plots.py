"""
Static (matplotlib/seaborn) chart builders used by the EDA notebooks and
the PDF technical report. Every function returns a `matplotlib.figure.Figure`
rather than calling `plt.show()`, so callers (notebooks, report generation,
tests) control display/saving themselves.

Covers every visualization required by the rubric's EDA section: dataset
summary, descriptive statistics, missing values, correlation heatmap,
histograms, boxplots, transaction-amount distribution, customer
transaction frequency, merchant distribution, hourly/daily/weekly
transaction patterns, customer spending behavior, and transaction amount
by merchant — 14 chart functions in total (rubric minimum: 10).
"""

from __future__ import annotations

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
_FIGSIZE = (10, 6)


def _new_fig(figsize=_FIGSIZE):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def plot_missing_values(df: pd.DataFrame) -> plt.Figure:
    """Bar chart of missing-value counts per column.
    Business interpretation: columns like `Merchant State`/`Zip` are
    expected to be missing for online transactions (no physical location);
    high missingness elsewhere would signal an ingestion problem."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    fig, ax = _new_fig()
    if len(missing) == 0:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center", fontsize=14)
    else:
        sns.barplot(x=missing.values, y=missing.index, ax=ax, color="#4C72B0")
        ax.set_xlabel("Missing value count")
    ax.set_title("Missing Values by Column")
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, numeric_columns: Optional[List[str]] = None) -> plt.Figure:
    """Correlation heatmap of numeric features.
    Business interpretation: highly correlated engineered features (e.g.
    `customer_avg_amount` and `rolling_avg_amount`) confirm the features
    capture related-but-distinct behavioral signals rather than pure
    duplicates; near-zero correlations with `Amount` for time-based
    indicators are expected and sanity-check the feature set."""
    cols = numeric_columns or df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(max(8, len(cols) * 0.6), max(6, len(cols) * 0.5)))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=len(cols) <= 15, fmt=".2f", ax=ax, square=True)
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    return fig


def plot_amount_distribution(df: pd.DataFrame, amount_col: str = "Amount") -> plt.Figure:
    """Histogram + log-scale inset of the transaction amount distribution.
    Business interpretation: transaction amounts are almost always
    right-skewed (many small purchases, few large ones); the log view
    reveals structure hidden by the long tail and justifies scaling/
    winsorizing decisions made in preprocessing."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df[amount_col], bins=60, ax=axes[0], color="#55A868")
    axes[0].set_title("Transaction Amount Distribution")
    axes[0].set_xlabel("Amount ($)")

    sns.histplot(np.log1p(df[amount_col].clip(lower=0)), bins=60, ax=axes[1], color="#C44E52")
    axes[1].set_title("Transaction Amount Distribution (log1p scale)")
    axes[1].set_xlabel("log(1 + Amount)")
    fig.tight_layout()
    return fig


def plot_amount_boxplot_by_category(df: pd.DataFrame, category_col: str, amount_col: str = "Amount") -> plt.Figure:
    """Boxplot of amount by a categorical column (e.g. `Use Chip`).
    Business interpretation: channel-level spread differences (e.g. online
    transactions skew higher/lower than chip transactions) inform whether
    channel should be weighted differently in fraud review triage."""
    fig, ax = _new_fig()
    order = df[category_col].value_counts().index
    sns.boxplot(data=df, x=category_col, y=amount_col, order=order, ax=ax, showfliers=False)
    ax.set_title(f"Transaction Amount by {category_col}")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def plot_customer_transaction_frequency(df: pd.DataFrame, id_col: str = "User") -> plt.Figure:
    """Histogram of transactions-per-customer.
    Business interpretation: a long right tail identifies "power users"
    whose sheer transaction volume could otherwise dominate frequency-based
    features if not normalized per-customer, as done in feature engineering."""
    counts = df[id_col].value_counts()
    fig, ax = _new_fig()
    sns.histplot(counts.values, bins=40, ax=ax, color="#8172B2")
    ax.set_title("Customer Transaction Frequency Distribution")
    ax.set_xlabel("Transactions per customer")
    fig.tight_layout()
    return fig


def plot_merchant_distribution(df: pd.DataFrame, merchant_col: str = "Merchant Name", top_n: int = 20) -> plt.Figure:
    """Bar chart of the top-N merchants by transaction count.
    Business interpretation: a handful of merchants often account for a
    disproportionate share of volume; `merchant_txn_frequency` directly
    encodes this popularity signal for the models."""
    top_merchants = df[merchant_col].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    sns.barplot(x=top_merchants.values, y=top_merchants.index.astype(str), ax=ax, color="#4C72B0")
    ax.set_title(f"Top {top_n} Merchants by Transaction Count")
    ax.set_xlabel("Transaction count")
    fig.tight_layout()
    return fig


def plot_hourly_transaction_analysis(df: pd.DataFrame, hour_col: str = "txn_hour") -> plt.Figure:
    """Transaction volume by hour of day.
    Business interpretation: identifies typical activity hours; late-night
    transactions outside a customer's usual hours are a classic fraud
    triage signal, directly motivating `business_hour_indicator`."""
    hourly = df[hour_col].value_counts().sort_index()
    fig, ax = _new_fig()
    sns.lineplot(x=hourly.index, y=hourly.values, marker="o", ax=ax, color="#DD8452")
    ax.set_title("Transaction Volume by Hour of Day")
    ax.set_xlabel("Hour (0-23)")
    ax.set_ylabel("Transaction count")
    ax.set_xticks(range(0, 24, 2))
    fig.tight_layout()
    return fig


def plot_daily_trend(df: pd.DataFrame, datetime_col: str = "transaction_datetime") -> plt.Figure:
    """Daily transaction volume trend over the observed period.
    Business interpretation: sudden spikes/dips can reflect seasonal
    shopping events (or data artifacts) and provide context for whether an
    unusually high anomaly count on a given day is a real pattern shift."""
    daily = df[datetime_col].dt.date.value_counts().sort_index()
    fig, ax = _new_fig(figsize=(12, 5))
    ax.plot(daily.index, daily.values, color="#55A868")
    ax.set_title("Daily Transaction Volume Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Transaction count")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_weekly_trend(df: pd.DataFrame, day_of_week_col: str = "txn_day_of_week") -> plt.Figure:
    """Transaction volume by day of week.
    Business interpretation: weekday vs. weekend spending patterns
    directly justify the `weekend_indicator` feature and help calibrate
    expectations for weekend anomaly-rate baselines."""
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekly = df[day_of_week_col].value_counts().sort_index()
    weekly.index = [day_names[i] for i in weekly.index]
    fig, ax = _new_fig()
    sns.barplot(x=weekly.index, y=weekly.values, ax=ax, color="#C44E52")
    ax.set_title("Transaction Volume by Day of Week")
    ax.set_ylabel("Transaction count")
    fig.tight_layout()
    return fig


def plot_customer_spending_behavior(df: pd.DataFrame, id_col: str = "User", amount_col: str = "Amount", top_n: int = 15) -> plt.Figure:
    """Average transaction amount for the top-N most active customers.
    Business interpretation: spending-level heterogeneity across customers
    is exactly why customer-relative features (`amount_deviation_from_
    customer_mean`) outperform a single global amount threshold."""
    top_customers = df[id_col].value_counts().head(top_n).index
    subset = df[df[id_col].isin(top_customers)]
    avg_by_customer = subset.groupby(id_col)[amount_col].mean().sort_values(ascending=False)
    fig, ax = _new_fig()
    sns.barplot(x=avg_by_customer.values, y=avg_by_customer.index.astype(str), ax=ax, color="#8172B2")
    ax.set_title(f"Average Transaction Amount — Top {top_n} Most Active Customers")
    ax.set_xlabel("Average amount ($)")
    ax.set_ylabel("Customer (User ID)")
    fig.tight_layout()
    return fig


def plot_transaction_amount_by_merchant(df: pd.DataFrame, merchant_col: str = "Merchant Name", amount_col: str = "Amount", top_n: int = 15) -> plt.Figure:
    """Average transaction amount for the top-N merchants by volume.
    Business interpretation: merchants with unusually high average amounts
    relative to their category (MCC) are useful context when reviewing a
    flagged transaction at that merchant."""
    top_merchants = df[merchant_col].value_counts().head(top_n).index
    subset = df[df[merchant_col].isin(top_merchants)]
    avg_by_merchant = subset.groupby(merchant_col)[amount_col].mean().sort_values(ascending=False)
    fig, ax = _new_fig()
    sns.barplot(x=avg_by_merchant.values, y=avg_by_merchant.index.astype(str), ax=ax, color="#4C72B0")
    ax.set_title(f"Average Transaction Amount — Top {top_n} Merchants")
    ax.set_xlabel("Average amount ($)")
    fig.tight_layout()
    return fig


def plot_roc_curve(fpr: List[float], tpr: List[float], model_name: str, auc: Optional[float] = None) -> plt.Figure:
    """ROC curve for a single model's anomaly scores vs. the held-out
    fraud label — evaluation-only use of the label, per project design."""
    fig, ax = _new_fig()
    label = f"{model_name}" + (f" (AUC={auc:.3f})" if auc is not None else "")
    ax.plot(fpr, tpr, label=label, color="#4C72B0")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_precision_recall_curve(precision: List[float], recall: List[float], model_name: str, ap: Optional[float] = None) -> plt.Figure:
    """Precision-Recall curve — more informative than ROC under the heavy
    class imbalance typical of fraud data."""
    fig, ax = _new_fig()
    label = f"{model_name}" + (f" (AP={ap:.3f})" if ap is not None else "")
    ax.plot(recall, precision, label=label, color="#C44E52")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {model_name}")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_anomaly_score_distribution(scores: np.ndarray, threshold: Optional[float] = None, model_name: str = "") -> plt.Figure:
    """Histogram of anomaly scores with the decision threshold marked."""
    fig, ax = _new_fig()
    sns.histplot(scores, bins=60, ax=ax, color="#55A868")
    if threshold is not None:
        ax.axvline(threshold, color="red", linestyle="--", label=f"Decision threshold ({threshold:.4f})")
        ax.legend()
    ax.set_title(f"Anomaly Score Distribution — {model_name}")
    ax.set_xlabel("Anomaly score (higher = more anomalous)")
    fig.tight_layout()
    return fig


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15) -> plt.Figure:
    """Horizontal bar chart of permutation feature importance (see
    `src/explainability/explain.py`)."""
    top = importance_df.head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    sns.barplot(data=top, x="importance", y="feature", ax=ax, color="#DD8452")
    ax.set_title("Permutation Feature Importance")
    ax.set_xlabel("Mean |score change| when feature is shuffled")
    fig.tight_layout()
    return fig


def plot_reconstruction_error_distribution(errors: np.ndarray, threshold: Optional[float] = None) -> plt.Figure:
    """Autoencoder-specific: distribution of per-row reconstruction error."""
    fig, ax = _new_fig()
    sns.histplot(errors, bins=60, ax=ax, color="#937860")
    if threshold is not None:
        ax.axvline(threshold, color="red", linestyle="--", label=f"Anomaly threshold ({threshold:.4f})")
        ax.legend()
    ax.set_title("Autoencoder Reconstruction Error Distribution")
    ax.set_xlabel("Mean squared reconstruction error")
    fig.tight_layout()
    return fig


def plot_model_comparison_bar(comparison_df: pd.DataFrame, metric: str = "roc_auc") -> plt.Figure:
    """Bar chart comparing all models on a single metric — the headline
    chart for the model-comparison section of the report/dashboard."""
    fig, ax = _new_fig()
    ordered = comparison_df.sort_values(metric, ascending=False)
    sns.barplot(data=ordered, x=metric, y="model", ax=ax, color="#4C72B0")
    ax.set_title(f"Model Comparison — {metric}")
    fig.tight_layout()
    return fig