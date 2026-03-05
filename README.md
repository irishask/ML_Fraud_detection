# IEEE-CIS Fraud Detection Pipeline

End-to-end ML pipeline for fraud detection on the [Kaggle IEEE-CIS / Vesta Corporation dataset](https://www.kaggle.com/competitions/ieee-fraud-detection).
LightGBM + XGBoost with behavioral feature engineering, Optuna hyperparameter tuning, and strict no-leakage time-based validation.

---

## Results

| Version | Model | ROC AUC | PR AUC | Δ ROC | Δ PR |
|---------|-------|--------:|-------:|------:|-----:|
| v0 | LightGBM — raw features, default params | 0.9196 | 0.5804 | — | — |
| v1 | LightGBM + aggregations + velocity features | 0.9235 | 0.5813 | +0.0039 | +0.0009 |
| **v2** | **LightGBM + behavioral features + Optuna** | **0.9272** | **0.6039** | **+0.0076** | **+0.0235** |
| v2 | XGBoost + Optuna | 0.9226 | 0.5641 | +0.0030 | −0.0163 |
| v2 | Ensemble (LightGBM + XGBoost) | 0.9264 | 0.5907 | +0.0068 | +0.0103 |

**Best model: v2 LightGBM** — strongest on both ROC AUC and PR AUC.

> **Why PR AUC matters here:** with a 3.5% fraud rate, PR AUC directly measures how well the model finds fraud without flooding legitimate transactions with false positives. ROC AUC alone is misleading on heavily imbalanced data.

---

## Dataset

- **Source:** [Kaggle IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) — provided by Vesta Corporation
- **Size:** ~590K transactions, ~431 raw features
- **Fraud rate:** 3.5% (heavily imbalanced)
- **Files:** `train_transaction.csv` + `train_identity.csv` merged on `TransactionID`

---

## Project Structure

```
ML_Fraud_detection/
│
├── data/
│   ├── raw/                        # Original Kaggle CSVs — never modified
│   ├── processed/                  # Merged + memory-reduced parquet files
│   ├── train.parquet
│   └── test.parquet
│
├── src/                            # Shared modules across all versions
│   ├── config.py                   # Paths, constants, seeds
│   ├── data_loader.py              # Load, merge, save raw and processed data
│   ├── feature_init_utils.py       # Initial features from EDA (time, device)
│   ├── pipeline_preprocess.py      # Load splits, preprocess, run Optuna
│   ├── pipeline_evaluate.py        # Train LightGBM / XGBoost, save models
│   ├── preproc_lgbm_xgboost.py     # Label encoding + NaN fill
│   ├── preproc_agg.py              # Aggregation features
│   ├── preproc_behavioral.py       # Behavioral fingerprint features
│   ├── preproc_product.py          # Product/service profile features
│   ├── train_lightgbm.py           # LightGBM training module
│   ├── train_xgboost.py            # XGBoost training module
│   ├── train_ensemble.py           # Weighted average ensemble
│   ├── tune_optuna.py              # Optuna tuning: LightGBM, XGBoost
│   └── evaluate_ml.py              # Metrics, ROC/PR plots, feature importance
│
├── v0/                             # Baseline
│   ├── baseline_v0.ipynb           # ROC 0.9196 | PR 0.5804
│   ├── preproc_v0.py
│   ├── split_v0.py
│   └── train_v0.py
│
├── v1/                             # Sprint 1 — aggregations & velocity
│   └── v1_aggr_ensemble_with_optuna.ipynb   # ROC 0.9235 | PR 0.5813
│
├── v2/                             # Sprint 2 — behavioral features + ensemble
│   ├── 01_eda_v2.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_preprocess_train_clean_optuna3.ipynb
│   └── 04_predict_evaluate.ipynb   # ROC 0.9272 | PR 0.6039
│
└── outputs/
    ├── enriched/                   # Engineered features (parquet)
    ├── models/                     # Saved models (pkl)
    ├── preproc/                    # Preprocessed splits (parquet)
    ├── best_params_lgbm.json       # Optuna best params — LightGBM
    └── best_params_xgb.json        # Optuna best params — XGBoost
```

---

## Pipeline Overview

The pipeline runs across 4 notebooks in order:

**`01_eda_v2.ipynb`** — Exploratory data analysis: distributions, fraud patterns, missing values, feature correlations.

**`02_feature_engineering.ipynb`** — Builds `train_enriched.parquet` with all engineered features. Runs on the full training set before the train/val split to avoid leakage.

**`03_preprocess_train_clean_optuna3.ipynb`** — Time-based train/val split (80/20 by `TransactionDT`), label encoding, Optuna hyperparameter tuning for LightGBM and XGBoost.

**`04_predict_evaluate.ipynb`** — Trains final models, computes ensemble, evaluates all models, plots ROC/PR curves and feature importance.

---

## Feature Engineering

Features are built in three versioned layers:

**v1 — Aggregation & Velocity**
- Cumulative transaction statistics per user: mean, std, min, max, count of `TransactionAmt`
- Multi-period velocity: transaction counts over last 3d / 7d / 30d windows
- Delta to previous transaction: amount change, time since last transaction
- Email domain instability: entropy of email domain changes per card
- Device instability: entropy of device changes per card

**v2 — Behavioral Fingerprint**
- Typical transaction hour per user (percentile-based)
- Deviation from personal median amount (`amt_vs_personal_median`, `amt_z_score`)
- Amount ratio relative to user history (`tx_amt_ratio`)
- Behavioral entropy: `uid_time_entropy` — unpredictability of transaction timing

**No-leakage guarantee:** all aggregations use `expanding().shift(1)` or `closed='left'` rolling windows — the current row is always excluded. The train/val split is strictly time-based.

---

## Hyperparameter Tuning

Optuna TPE (Tree-structured Parzen Estimator) with **MED quality profile** (~50 trials, 50% data sample):

| Model | Trials | Sample | Approx. Time (CPU) |
|-------|-------:|-------:|-------------------:|
| LightGBM | 50 | 50% of train | ~2–3h |
| XGBoost | 50 | 50% of train | ~2–3h |

> **Note on CatBoost:** CatBoost was evaluated but removed from the final pipeline. Without a GPU, Optuna tuning at MED level ran for 7+ hours and was far from completion — CatBoost's ordered target statistics encoding is extremely memory- and compute-intensive on 472K rows × 497 features. The partially trained CatBoost model (ROC 0.9177) performed worse than the v0 baseline (ROC 0.9196), confirming it was undertrained. Re-tuning with GPU access is listed as a future improvement.

---

## Validation Strategy

- **Time-based split:** 80% train / 20% val, split by `TransactionDT` — no random shuffling
- **Why time-based:** fraud patterns evolve over time; random splits leak future information into training, producing inflated and unreliable metrics
- **Metrics:** both ROC AUC and PR AUC tracked at every version — PR AUC is the primary metric given the 3.5% class imbalance

---

## Comparison with Kaggle Leaderboard

| Level | ROC AUC |
|-------|--------:|
| 1st place (Chris Deotte) — GPU, thousands of features, 3-model ensemble | 0.9459 |
| Top 1% (silver) — multi-model blend | ~0.936 |
| Top 5% | ~0.930–0.934 |
| **This project — v2 LightGBM (CPU only)** | **0.9272** |
| v0 baseline | 0.9196 |

The gap to top solutions is explained by three factors: scale of feature engineering (top solutions engineer hundreds of features vs. 23 here), multi-model ensembles with all models fully tuned on final features, and GPU acceleration enabling much deeper Optuna search.

---

## Tech Stack

| Tool | Role |
|------|------|
| Python 3.10+ | Core language |
| LightGBM | Primary model |
| XGBoost | Ensemble model |
| Optuna | Hyperparameter tuning (TPE) |
| pandas / numpy | Feature engineering |
| scikit-learn | Preprocessing, metrics |
| matplotlib | ROC/PR curves, feature importance |
| Jupyter / Cursor IDE | Development environment |

---

## How to Run

```bash
# 1. Install dependencies
pip install lightgbm xgboost optuna pandas numpy scikit-learn matplotlib

# 2. Place Kaggle data files in data/raw/
# train_transaction.csv, train_identity.csv, test_transaction.csv, test_identity.csv

# 3. Run notebooks in order
jupyter notebook v2/01_eda_v2.ipynb
jupyter notebook v2/02_feature_engineering.ipynb
jupyter notebook v2/03_preprocess_train_clean_optuna3.ipynb
jupyter notebook v2/04_predict_evaluate.ipynb
```

> Set `PREPROC_READY_LGBM_XGB = True` and `RUN_OPTUNA_LGBM = False` / `RUN_OPTUNA_XGB = False` in notebook 03 to skip rerunning preprocessing and tuning if artifacts already exist on disk.

---

## Future Improvements

- **Re-tune XGBoost with updated features** — XGBoost PR AUC (0.5641) is below baseline; re-running Optuna specifically optimizing for PR AUC instead of ROC AUC may fix this
- **Target encoding** — replace label encoding with leave-one-out target encoding for high-cardinality categoricals (`card1`, `addr1`, `P_emaildomain`)
- **UID construction** — build a stable user identity from `card1 + addr1 + D1` following Chris Deotte's approach; aggregate all features by UID for richer behavioral signals
- **CatBoost with GPU** — CatBoost's ordered boosting is well-suited for this dataset but requires GPU for practical Optuna tuning at scale
- **Stacking ensemble** — replace weighted average with a meta-learner (logistic regression on out-of-fold predictions) for a stronger, less correlated ensemble

---

*Built as a final project for the Machine Learning course at HIT (Holon Institute of Technology).*
