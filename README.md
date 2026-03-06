# IEEE-CIS Fraud Detection Pipeline

End-to-end ML pipeline for fraud detection on the [Kaggle IEEE-CIS / Vesta Corporation dataset](https://www.kaggle.com/competitions/ieee-fraud-detection).
LightGBM with behavioral feature engineering, Optuna hyperparameter tuning, and strict no-leakage time-based validation.

---

## Results

| Version | Model | ROC AUC | PR AUC | Δ ROC | Δ PR |
|---------|-------|--------:|-------:|------:|-----:|
| v0 | LightGBM — raw features, default params | 0.9196 | 0.5804 | — | — |
| v1 | LightGBM + aggregations + velocity features | 0.9235 | 0.5813 | +0.0039 | +0.0009 |
| **v2** | **LightGBM + behavioral features + Optuna med_high** | **0.9272** | **0.6039** | **+0.0076** | **+0.0235** |
| v2 | XGBoost + Optuna med | 0.9226 | 0.5641 | +0.0030 | −0.0163 |
| v2 | Ensemble (LightGBM + XGBoost) — built, evaluated, cancelled | 0.9264 | 0.5907 | +0.0068 | +0.0103 |

**Best model: v2 LightGBM** — strongest on both ROC AUC and PR AUC. The ensemble was cancelled: XGBoost PR AUC (0.5641) is below the v0 baseline, and including a weak component degrades the combined result.

> **Why PR AUC matters here:** with a 3.5% fraud rate, PR AUC directly measures how well the model finds fraud without flooding legitimate transactions with false positives. ROC AUC alone is misleading on heavily imbalanced data.

---

## Dataset

- **Source:** [Kaggle IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) — provided by Vesta Corporation
- **Size:** 590,540 transactions | ~6 months
- **Features:** 434 raw (after merge) + 23 engineered = 457 total
- **Fraud rate:** 3.5% — severe class imbalance
- **Files:** `train_transaction.csv` + `train_identity.csv` merged on `TransactionID`

---

## Project Structure

```
ML_Fraud_detection/
├── data/           # Raw CSVs and processed parquet files
├── src/            # Shared modules: preprocessing, training, tuning, evaluation
├── v0/             # Baseline — ROC 0.9196 | PR 0.5804
├── v1/             # Aggregations & velocity — ROC 0.9235 | PR 0.5813
├── v2/             # Behavioral features + Optuna — ROC 0.9272 | PR 0.6039
└── outputs/        # Models, preprocessed splits, Optuna params
```

---

## Pipeline Overview

The pipeline runs across 4 notebooks in `v2/` in order:

**`01_eda_v2.ipynb`** — Exploratory data analysis: fraud patterns, feature correlations, UID validation.

**`02_feature_engineering.ipynb`** — Builds all 23 engineered features. Runs on the full training set before the train/val split to prevent leakage.

**`03_preprocess_train_clean_optuna3.ipynb`** — Time-based train/val split (80/20 by `TransactionDT`), label encoding, Optuna tuning for LightGBM and XGBoost.

**`04_predict_evaluate.ipynb`** — Trains final models, evaluates all models and ensemble, plots ROC/PR curves and feature importance.

> Set `PREPROC_READY_LGBM_XGB = True` and `RUN_OPTUNA_LGBM = False` / `RUN_OPTUNA_XGB = False` in notebook 03 to skip rerunning preprocessing and tuning if artifacts already exist on disk.

---

## Feature Engineering — 23 Features

All features are computed on the full dataset before the train/val split. No-leakage guarantee: all aggregations use `expanding().shift(1)` or `rolling(closed='left')` — the current row is always excluded.

**v1 — Aggregation & Velocity (19 features)**
- Cumulative transaction statistics per user (UID = `card1 + addr1`): count, mean, std, min, max of `TransactionAmt`
- Multi-period velocity: transaction counts over last 3d / 7d / 30d windows
- Delta to previous transaction: amount change, time since last transaction
- Email domain instability: new payer/recipient email flags, nunique per card (fraud lift up to 1.85×)
- Device instability: new device flag, nunique per card (fraud lift 1.66×)

**v2 — Behavioral Fingerprint + Product Profile (4 + 1 features)**
- Deviation from personal median amount (`amt_vs_personal_median`, `amt_z_score`)
- Unusual transaction hour for this specific card (`hour_vs_typical`)
- Behavioral entropy: `uid_time_entropy` — unpredictability of transaction timing
- Product profile: `is_new_product` — new product category for this card

---

## Hyperparameter Tuning

Optuna TPE (Tree-structured Parzen Estimator) — independent tuning per model:

| Model | Quality | Trials | Sample | Approx. Time (CPU) |
|-------|---------|-------:|-------:|-------------------:|
| LightGBM | med_high | 75 | 75% of train | ~10h |
| XGBoost | med | 50 | 50% of train | ~3h |

Running med_high for LightGBM yielded +0.0017 ROC AUC over med (0.9255 → 0.9272). Given the modest gain, XGBoost was not re-tuned at med_high level.

> **Note on CatBoost:** CatBoost was evaluated as a third ensemble component. Optuna MED tuning was killed after ~10 hours with only ~50% of trials completed — its ordered target statistics encoding is extremely compute-intensive on CPU. The partially trained model (ROC 0.9177) was below the v0 baseline (0.9196) and was removed from the pipeline. CatBoost with GPU remains a future candidate.

---

## Validation Strategy

- **Time-based split:** 80% train / 20% val, split by `TransactionDT` — no random shuffling
- **Why time-based:** fraud patterns evolve over time; random splits leak future information into training, producing inflated and unreliable metrics
- **Metrics:** ROC AUC (official Kaggle metric) + PR AUC tracked at every version

---

## Comparison with Kaggle Leaderboard

| Level | ROC AUC |
|-------|--------:|
| 1st place (Chris Deotte) — GPU, multi-model ensemble | 0.9459 |
| Top 1% — gold medal threshold | ~0.940+ |
| Top 5% — community-reported threshold | ~0.9230 |
| **This project — v2 LightGBM (CPU only)** | **0.9272** |
| v0 baseline | 0.9196 |

Results are on a strict time-based local validation set. The community-reported top 5% threshold (~0.9230) is on Kaggle's private test set — direct comparison is approximate, but the result is competitive with top-tier submissions.

The gap to top solutions reflects three factors: scale of feature engineering (top solutions use hundreds of features vs. 23 here), fully tuned multi-model ensembles, and GPU acceleration enabling deeper Optuna search.

---

## Tech Stack

| Tool | Role |
|------|------|
| Python 3.10+ | Core language |
| LightGBM | Primary model |
| XGBoost | Evaluated ensemble component |
| Optuna | Hyperparameter tuning (TPE) |
| pandas / numpy | Feature engineering |
| scikit-learn | Preprocessing, metrics |
| matplotlib | ROC/PR curves, feature importance |
| Jupyter / Cursor IDE | Development environment |

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place Kaggle data files in data/raw/
# train_transaction.csv, train_identity.csv, test_transaction.csv, test_identity.csv

# 3. Run notebooks in order
jupyter notebook v2/01_eda_v2.ipynb
jupyter notebook v2/02_feature_engineering.ipynb
jupyter notebook v2/03_preprocess_train_clean_optuna3.ipynb
jupyter notebook v2/04_predict_evaluate.ipynb
```

---

## Future Improvements

- **Feature importance analysis** — deep SHAP-based analysis of V-columns (339 anonymous Vesta features); understanding their real meaning is the single most impactful improvement available
- **CatBoost with GPU** — architecturally different from LightGBM; strongest ensemble candidate once GPU is available
- **MLP (Neural Network) with GPU** — non-tree model adds genuine ensemble diversity; requires GPU for practical Optuna tuning
- **Isolation Forest as meta-feature** — unsupervised anomaly score as an additional LightGBM input feature
- **Target encoding** — replace label encoding with leave-one-out target encoding for `card1`, `addr1`, `P_emaildomain`
- **UID construction** — extend with D1 drift correction for a more stable user identity key

---

*Final project — Machine Learning course at HIT (Holon Institute of Technology).*
