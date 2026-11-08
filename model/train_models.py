"""
Train and evaluate 5 classification models on the
Breast Cancer Wisconsin (Diagnostic) dataset (UCI).

Run from the project root:
    python model/train_models.py

Outputs
-------
- model/<model_name>.pkl          : fitted model pipelines
- model/metrics_comparison.csv    : evaluation metrics on the held-out test set
- test_data.csv                   : held-out test set (features + 'diagnosis' target)
- train_data.csv                  : training split (for reference)
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 21
TEST_SIZE = 0.25
TARGET = "diagnosis"  # 1 = Malignant, 0 = Benign

try:
    # Running as a script: python model/train_models.py
    ROOT = Path(__file__).resolve().parent.parent
except NameError:
    # Running inside Jupyter (no __file__): use the notebook's folder.
    ROOT = Path.cwd()
    if ROOT.name == "model":
        ROOT = ROOT.parent
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)


def load_dataset() -> pd.DataFrame:
    """Load the UCI Breast Cancer Wisconsin (Diagnostic) data as a DataFrame."""
    raw = load_breast_cancer(as_frame=True)
    df = raw.frame.copy()
    # Clean column names: 'mean radius' -> 'mean_radius'
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    # sklearn encodes 0 = malignant, 1 = benign. Flip so that the
    # positive class (1) is the clinically important one: Malignant.
    df[TARGET] = 1 - df.pop("target")
    return df


def build_models() -> dict:
    """All 5 required classifiers. Distance/margin-based models get scaling."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, C=1.0, random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=4, random_state=RANDOM_STATE
        ),
        "kNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7, weights="distance")),
        ]),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                        random_state=RANDOM_STATE)),
        ]),
    }


def evaluate(model, X, y) -> dict:
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_prob),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }


def file_name(model_name: str) -> str:
    return model_name.lower().replace(" ", "_") + ".pkl"


def main():
    df = load_dataset()
    print(f"Dataset shape: {df.shape}  (features = {df.shape[1] - 1})")
    print("Class balance:\n", df[TARGET].value_counts().rename({1: "Malignant", 0: "Benign"}))

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # Save splits
    pd.concat([X_train, y_train], axis=1).to_csv(ROOT / "train_data.csv", index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(ROOT / "test_data.csv", index=False)

    rows = []
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_DIR / file_name(name))
        scores = evaluate(model, X_test, y_test)
        rows.append({"ML Model Name": name, **scores})
        print(f"{name:<24} " + "  ".join(f"{k}={v:.4f}" for k, v in scores.items()))

    results = pd.DataFrame(rows).round(4)
    results.to_csv(MODEL_DIR / "metrics_comparison.csv", index=False)
    print("\n", results.to_string(index=False))


if __name__ == "__main__":
    np.random.seed(RANDOM_STATE)
    main()
