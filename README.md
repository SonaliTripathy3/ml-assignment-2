# Tumour Classifier Lab — Breast Cancer Diagnosis with 5 ML Classifiers

**M.Tech (DSE) · Machine Learning · Assignment 2**

🔗 **Live app:** `<PASTE STREAMLIT APP LINK>`
🔗 **GitHub repo:** `<PASTE GITHUB REPO LINK>`

---

## a. Problem statement

Given 30 numeric measurements of cell nuclei taken from a digitised image of a
fine-needle aspirate (FNA) of a breast mass, predict whether the tumour is
**malignant (1)** or **benign (0)**. This is a **binary classification** problem.

The goal is to implement five classical classifiers on the same data, evaluate
them on a held-out test set using six metrics (Accuracy, AUC, Precision,
Recall, F1, MCC), compare them, and expose the results through an interactive
Streamlit web app deployed on Streamlit Community Cloud.

Because a missed malignant tumour (false negative) is far costlier than a false
alarm, **Recall on the malignant class** and **MCC** are given extra weight
when choosing the best model.

## b. Dataset description

| Item | Detail |
|---|---|
| Name | Breast Cancer Wisconsin (Diagnostic) |
| Source | UCI Machine Learning Repository — https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic (also on Kaggle as `uciml/breast-cancer-wisconsin-data`) |
| Instances | 569 |
| Features | 30 (all real-valued) |
| Target | `diagnosis` — 1 = Malignant (212, 37.3%), 0 = Benign (357, 62.7%) |
| Missing values | None |

The 30 features are 10 properties of each cell nucleus — radius, texture,
perimeter, area, smoothness, compactness, concavity, concave points, symmetry
and fractal dimension — each summarised three ways: **mean**, **standard
error** and **worst** (mean of the three largest values). Column names follow
that pattern, e.g. `mean_radius`, `radius_error`, `worst_radius`.

Several features are strongly correlated (radius, perimeter and area measure
almost the same thing), which matters for Naive Bayes (see observations).

**Preprocessing / split**
- Target recoded so that Malignant = 1 (the positive class for Precision/Recall/F1).
- Stratified train/test split: 75% train (426 rows) / 25% test (143 rows: 53 malignant, 90 benign), `random_state = 21`.
- `StandardScaler` applied (inside a pipeline, fitted on train only) for Logistic Regression, kNN and SVM, which are scale-sensitive. Decision Tree and Gaussian NB use raw features.
- The test split is saved as `test_data.csv` — this is the file to upload to the app.

## c. GitHub repository link

`<PASTE GITHUB REPO LINK>`

```
ml-assignment-2/
├── app.py                      # Streamlit application
├── requirements.txt
├── README.md
├── test_data.csv               # held-out test set (upload this in the app)
├── train_data.csv              # training split (used only if a model must be refit)
└── model/
    ├── train_models.py         # trains & evaluates all 5 models, saves .pkl files
    ├── metrics_comparison.csv  # metrics table produced by train_models.py
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    └── support_vector_machine.pkl
```

**Run locally**
```bash
pip install -r requirements.txt
python model/train_models.py      # retrains models, regenerates csvs + metrics
streamlit run app.py
```

## d. Models used

| # | Model | Configuration |
|---|---|---|
| 1 | Logistic Regression | StandardScaler + L2 logistic regression, C = 1.0 |
| 2 | Decision Tree | max_depth = 5, min_samples_leaf = 4 |
| 3 | k-Nearest Neighbours | StandardScaler + k = 7, distance-weighted votes |
| 4 | Naive Bayes | Gaussian NB (features are continuous) |
| 5 | Support Vector Machine | StandardScaler + RBF kernel, C = 1.0, gamma = "scale" |

### Comparison table (test set, n = 143)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9720 | 0.9964 | 0.9623 | **0.9623** | **0.9623** | 0.9400 |
| Decision Tree | 0.9441 | 0.9673 | 0.9412 | 0.9057 | 0.9231 | 0.8795 |
| kNN | **0.9720** | 0.9977 | **1.0000** | 0.9245 | 0.9608 | **0.9408** |
| Naive Bayes | 0.9371 | 0.9906 | 0.9400 | 0.8868 | 0.9126 | 0.8644 |
| Support Vector Machine | 0.9650 | **0.9987** | 0.9800 | 0.9245 | 0.9515 | 0.9251 |

Confusion matrices on the test set (TN / FP / FN / TP):
Logistic Regression 88 / 2 / 2 / 51 · Decision Tree 87 / 3 / 5 / 48 · kNN 90 / 0 / 4 / 49 ·
Naive Bayes 87 / 3 / 6 / 47 · SVM 89 / 1 / 4 / 49.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Joint-best accuracy (97.2%) and the **best recall and F1** — it missed only 2 of 53 malignant tumours, fewer than any other model. Precision and recall are perfectly balanced (2 FP, 2 FN). The near-linear separability of the scaled features suits a linear model well, and its AUC of 0.996 shows its probabilities rank cases reliably. |
| Decision Tree | Weakest model overall on AUC (0.967) and second-weakest on MCC (0.880). A single tree split mostly on `worst_concave_points` (≈78% of importance) and `worst_area`, producing coarse, step-like probabilities — hence the lower AUC. It missed 5 malignant cases. Depth was capped at 5 to limit overfitting, but a single tree still has high variance on 426 training rows. |
| kNN | Ties Logistic Regression on accuracy and has the **highest MCC (0.941) and perfect precision** — zero benign cases flagged as malignant. However it missed 4 malignant cases (recall 0.925), so it is conservative about calling malignancy. Works well only because features were standardised; distance-weighted voting with k = 7 gave smooth probabilities (AUC 0.998). |
| Naive Bayes | Lowest accuracy (93.7%), recall (0.887) and MCC (0.864) — it missed 6 malignant tumours. The model assumes features are independent, which is clearly violated here (radius, perimeter and area are almost perfectly correlated), so the same evidence is counted several times. Interestingly its AUC is still high (0.991): the ranking of cases is good, but the probabilities are poorly calibrated around the 0.5 threshold. |
| Support Vector Machine | **Highest AUC (0.9987)** — the best at separating the two classes across all thresholds — with high precision (only 1 false positive). At the default 0.5 threshold it missed 4 malignant cases, same as kNN. Lowering the decision threshold (try the slider in the app) recovers recall with very little loss of precision. |
| **Overall winner for this dataset** | **Logistic Regression.** It ties for best accuracy, is essentially tied with kNN on MCC (0.940 vs 0.941), and has the best recall and F1 — i.e. it misses the fewest cancers, which is what matters most in a diagnostic setting. It is also the simplest and most interpretable model. SVM is a close second with the best AUC. Note that the top three models differ by only one or two test cases, so on a 143-row test set the gap between LR, kNN and SVM is small. |

## Streamlit app features

- **CSV upload** of test data (a sample `test_data.csv` can be downloaded from the sidebar)
- **Model selection dropdown** — any of the 5 trained models
- **Evaluation metrics** — Accuracy, AUC, Precision, Recall, F1, MCC
- **Confusion matrix** heatmap and **classification report**
- **Compare all models** tab with a metrics table, bar chart and overlaid ROC curves
- **Decision-threshold slider** to explore the precision/recall trade-off
- **Predictions** tab with per-row probabilities and CSV download
- Works without the `diagnosis` column too (predictions only)
