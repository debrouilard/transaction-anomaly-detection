"""
Streamlit dashboard entry point.

`app.py` is Streamlit's main script; every file in `dashboard/pages/` is
auto-discovered as an additional page in the sidebar navigation. This file
sets shared page config and shows a lightweight landing/status view — the
full project narrative lives on the "Home" page (`pages/Home.py`) so the
two don't duplicate content.

Run with: `streamlit run dashboard/app.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import get_config  # noqa: E402
from src.utils.helpers import load_json  # noqa: E402

cfg = get_config()

st.set_page_config(
    page_title=cfg.dashboard.get("app_title", "Transaction Anomaly Detection"),
    page_icon=cfg.dashboard.get("page_icon", "💳"),
    layout="wide",
    initial_sidebar_state="expanded",
)


def _artifact_status() -> dict:
    return {
        "raw_data": cfg.paths.raw_data.exists(),
        "engineered_data": cfg.paths.engineered_data.exists(),
        "scaler": cfg.paths.scaler_path.exists(),
        "isolation_forest": (cfg.paths.models_dir / "isolation_forest.pkl").exists(),
        "local_outlier_factor": (cfg.paths.models_dir / "lof.pkl").exists(),
        "one_class_svm": (cfg.paths.models_dir / "one_class_svm.pkl").exists(),
        "autoencoder": (cfg.paths.models_dir / "autoencoder.pt").exists(),
        "metrics": cfg.paths.metrics_path.exists(),
    }


st.title(f"{cfg.dashboard.get('page_icon', '💳')} {cfg.dashboard.get('app_title')}")
st.caption(
    "Unsupervised anomaly detection on the IBM/Kaggle Credit Card Transactions dataset — "
    "no fraud labels used during model training."
)

status = _artifact_status()
all_ready = all(status.values())

if all_ready:
    st.success("All pipeline artifacts found — every dashboard page is ready to explore.")
else:
    st.warning(
        "Some pipeline artifacts are missing. Run the commands below from the project root, "
        "then refresh this page."
    )
    missing = [k for k, v in status.items() if not v]
    st.code(
        "python scripts/run_pipeline.py   # runs data prep + training + evaluation end-to-end\n"
        "# or step by step:\n"
        "python scripts/train.py\n"
        "python scripts/evaluate.py",
        language="bash",
    )
    st.write("Missing artifacts:", ", ".join(missing))

st.divider()

cols = st.columns(4)
labels_icons = [
    ("raw_data", "Raw data", "📄"),
    ("engineered_data", "Engineered features", "🛠️"),
    ("metrics", "Evaluation metrics", "📊"),
    ("autoencoder", "All 4 models trained", "🤖"),
]
for col, (key, label, icon) in zip(cols, labels_icons):
    with col:
        st.metric(label=f"{icon} {label}", value="Ready" if status[key] else "Missing")

if status["metrics"]:
    st.divider()
    st.subheader("Quick summary")
    try:
        recommendation = load_json(cfg.paths.reports_dir / "recommendation.json")
        st.info(
            f"**Recommended model:** `{recommendation.get('recommended_model')}` — "
            f"{recommendation.get('reason', '')}"
        )
    except FileNotFoundError:
        st.write("Run `python scripts/evaluate.py` to generate a model recommendation.")

st.divider()
st.markdown(
    """
    ### Navigate using the sidebar:
    - **Home** — full project overview, methodology, and dataset description
    - **Data Analysis** — interactive EDA on the engineered transaction data
    - **Model Comparison** — metrics, ROC/PR curves, and runtime comparison across all models
    - **Predict** — upload a CSV of new transactions and score them live
    - **Business Insights** — anomaly patterns by merchant, category, and time
    """
)