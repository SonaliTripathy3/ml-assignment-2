"""
Streamlit app: Breast Cancer Diagnosis - comparing 5 classifiers.

Upload the test CSV (features + 'diagnosis' column), pick a model,
and see its metrics, confusion matrix and classification report.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TARGET = "diagnosis"
CLASS_NAMES = ["Benign (0)", "Malignant (1)"]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "support_vector_machine.pkl",
}

st.set_page_config(page_title="Tumour Classifier Lab", page_icon="🩺", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem;}
    div[data-testid="stMetric"] {
        background: rgba(120, 90, 200, 0.07);
        border: 1px solid rgba(120, 90, 200, 0.25);
        border-radius: 10px; padding: 10px 14px;
    }
    .note {font-size: 0.9rem; opacity: 0.8;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- loading
@st.cache_resource(show_spinner="Loading trained models...")
def load_models() -> dict:
    """Load the saved pipelines. If a pickle can't be read (e.g. a library
    version mismatch on the server), retrain that model from train_data.csv."""
    models = {}
    for name, fname in MODEL_FILES.items():
        try:
            models[name] = joblib.load(MODEL_DIR / fname)
        except Exception:
            models[name] = None
    if any(m is None for m in models.values()):
        import sys
        sys.path.insert(0, str(MODEL_DIR))
        from train_models import build_models  # noqa: E402

        train = pd.read_csv(ROOT / "train_data.csv")
        X, y = train.drop(columns=[TARGET]), train[TARGET]
        fresh = build_models()
        for name, m in models.items():
            if m is None:
                models[name] = fresh[name].fit(X, y)
    return models


@st.cache_data
def load_reference_metrics() -> pd.DataFrame:
    return pd.read_csv(MODEL_DIR / "metrics_comparison.csv")


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    out = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": np.nan,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    if len(np.unique(y_true)) == 2:
        out["AUC"] = roc_auc_score(y_true, y_prob)
    return out


def plot_confusion(cm) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
                annot_kws={"size": 14})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    return fig


def plot_roc(y_true, probs: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        ax.plot(fpr, tpr, label=f"{name} ({roc_auc_score(y_true, p):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    return fig


models = load_models()
feature_cols = list(next(iter(models.values())).feature_names_in_)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("1 · Test data")
    uploaded = st.file_uploader("Upload test CSV", type=["csv"])
    st.caption(f"Must contain the {len(feature_cols)} feature columns; "
               f"include `{TARGET}` (0/1) to get metrics.")
    sample_path = ROOT / "test_data.csv"
    if sample_path.exists():
        st.download_button("Download sample test_data.csv",
                           sample_path.read_bytes(), "test_data.csv", "text/csv")

    st.header("2 · Model")
    model_name = st.selectbox("Choose a classifier", list(MODEL_FILES))

    st.header("3 · Threshold")
    threshold = st.slider("Malignant if P(malignant) ≥", 0.05, 0.95, 0.50, 0.05)

# ---------------------------------------------------------------- header
st.title("🩺 Tumour Classifier Lab")
st.markdown(
    "Five classifiers trained on the **UCI Breast Cancer Wisconsin (Diagnostic)** "
    "dataset — 30 cell-nucleus measurements per biopsy, predicting **benign vs malignant**."
)

if uploaded is None:
    st.info("Upload your test CSV in the sidebar to begin "
            "(or download the sample file there first).")
    st.subheader("Reference results on the held-out test set")
    st.dataframe(load_reference_metrics().style.format(precision=4)
                 .highlight_max(subset=["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
                                color="#d8ccf2"),
                 hide_index=True, use_container_width=True)
    st.stop()

# ---------------------------------------------------------------- data checks
df = pd.read_csv(uploaded)
df.columns = [c.strip().replace(" ", "_") for c in df.columns]
missing = [c for c in feature_cols if c not in df.columns]
if missing:
    st.error(f"The uploaded file is missing {len(missing)} required column(s): "
             f"{', '.join(missing[:8])}{' ...' if len(missing) > 8 else ''}")
    st.stop()

X = df[feature_cols]
has_target = TARGET in df.columns
y = df[TARGET].astype(int) if has_target else None

c1, c2, c3 = st.columns(3)
c1.metric("Rows uploaded", len(df))
c2.metric("Features used", len(feature_cols))
c3.metric("Malignant share", f"{y.mean():.1%}" if has_target else "—")

with st.expander("Preview uploaded data"):
    st.dataframe(df.head(20), use_container_width=True)

model = models[model_name]
prob = model.predict_proba(X)[:, 1]
pred = (prob >= threshold).astype(int)

tab_model, tab_compare, tab_pred = st.tabs(
    [f"📊 {model_name}", "⚖️ Compare all models", "📄 Predictions"]
)

# ---------------------------------------------------------------- tab 1
with tab_model:
    if not has_target:
        st.warning(f"No `{TARGET}` column found — showing predictions only "
                   "(see the Predictions tab).")
    else:
        m = compute_metrics(y, pred, prob)
        cols = st.columns(6)
        for col, (k, v) in zip(cols, m.items()):
            col.metric(k, "n/a" if np.isnan(v) else f"{v:.4f}")

        left, right = st.columns(2)
        with left:
            st.markdown("**Confusion matrix**")
            st.pyplot(plot_confusion(confusion_matrix(y, pred, labels=[0, 1])))
        with right:
            st.markdown("**Classification report**")
            rep = classification_report(y, pred, labels=[0, 1], target_names=CLASS_NAMES,
                                        output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(rep).T.style.format(precision=3),
                         use_container_width=True)
            fn = int(((y == 1) & (pred == 0)).sum())
            st.markdown(f"<p class='note'>Missed malignant cases (false negatives): "
                        f"<b>{fn}</b>. In screening these are the costliest errors — "
                        f"lower the threshold to trade precision for recall.</p>",
                        unsafe_allow_html=True)

# ---------------------------------------------------------------- tab 2
with tab_compare:
    if not has_target:
        st.warning("Comparison needs the target column.")
    else:
        rows, probs = [], {}
        for name, mdl in models.items():
            p = mdl.predict_proba(X)[:, 1]
            probs[name] = p
            rows.append({"ML Model Name": name,
                         **compute_metrics(y, (p >= threshold).astype(int), p)})
        comp = pd.DataFrame(rows)
        st.dataframe(comp.style.format(precision=4)
                     .highlight_max(subset=comp.columns[1:], color="#d8ccf2"),
                     hide_index=True, use_container_width=True)
        a, b = st.columns(2)
        with a:
            metric = st.radio("Metric to plot", comp.columns[1:].tolist(), horizontal=True)
            fig, ax = plt.subplots(figsize=(4.6, 3.6))
            sns.barplot(data=comp, y="ML Model Name", x=metric, color="#7b61c9", ax=ax)
            ax.set_xlim(max(0, comp[metric].min() - 0.1), 1.0)
            ax.set_ylabel("")
            fig.tight_layout()
            st.pyplot(fig)
        with b:
            st.markdown("**ROC curves (AUC in legend)**")
            if len(np.unique(y)) == 2:
                st.pyplot(plot_roc(y, probs))

# ---------------------------------------------------------------- tab 3
with tab_pred:
    out = df.copy()
    out["predicted"] = pred
    out["p_malignant"] = prob.round(4)
    if has_target:
        out["correct"] = out["predicted"] == out[TARGET]
    st.dataframe(out, use_container_width=True, height=380)
    st.download_button("Download predictions CSV", out.to_csv(index=False).encode(),
                       f"predictions_{model_name.lower().replace(' ', '_')}.csv", "text/csv")

st.caption("M.Tech (DSE) · Machine Learning · Assignment 2")
