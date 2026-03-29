# IEEE-CIS Fraud Detection — ML Pipeline

**HIT Final ML Course | Portfolio Project**  
Irena Shtelman Kravitz — MSc. in Data Science, Holon Institute of Technology

---

## Project Overview

End-to-end fraud detection pipeline on the Kaggle IEEE-CIS / Vesta Corporation dataset (~590K real e-commerce transactions, 3.5% fraud rate).

**Goal:** Build a production-quality, leakage-free ML pipeline that maximizes fraud detection quality on a strictly frozen test set.

---

## Final Results

| Model | Test ROC AUC | Test PR AUC |
|---|---|---|
| Statistical baseline (predict all legitimate) | — | 0.035 |
| V2 LightGBM (ROC AUC objective) | 0.8974 | 0.5087 |
| V3 LightGBM (PR AUC + instance weighting) | 0.8904 | 0.5142 |
| **★ Ensemble V2+V3 (Final)** | **0.8990** | **0.5178** |

**~14.8× improvement in PR AUC over statistical baseline.**  
Comparable to top 20% of 6,381 Kaggle teams — evaluated on a strictly frozen test set, touched exactly once.

---

## Architecture

### 3-Way Temporal Split (60/20/20)
- **Train:** days 1–101 (354,324 rows) — model training
- **Val:** days 101–141 (118,108 rows) — early stopping + Optuna objective
- **Test:** days 141–183 (118,108 rows) — **frozen, touched exactly once**

### Feature Engineering — 24 New Features
All features computed on full dataset (train+test concatenated), using strictly prior transactions only (`expanding().shift(1)`, `rolling(closed='left')`). Zero leakage by design.

- **Aggregation (18):** transaction history per card — velocity, amounts, email/device diversity
- **Behavioral Fingerprint (4):** deviation from card's personal spending pattern
- **Product Profile (2):** purchase pattern per ProductCD

**UID = card1 + addr1** — reconstructed user identity, confirmed by 6 independent analyses (incl. anchor fix test).

### Models
- **LightGBM V2** — Optuna med_high (75 trials, 75% data), ROC AUC objective
- **LightGBM V3** — Optuna med_high (75 trials, 75% data), PR AUC objective + temporal instance weighting (4 chunks, w=1→4)
- **Final: Ensemble V2+V3** — simple average of predicted probabilities

### Hyperparameter Tuning
Optuna TPE — Bayesian optimization, learns from every trial.
- V2: maximize ROC AUC on val
- V3: maximize PR AUC on val + stratified sampling per temporal chunk

---

## Experiments

| # | Experiment | Result |
|---|---|---|
| 1 | V2: Feature Engineering + Optuna (ROC AUC) | LightGBM ✅ best single model. XGBoost and CatBoost below LightGBM on PR AUC. |
| 2 | OOF Stacking (LightGBM → XGBoost) | ❌ 3.5% fraud rate → corrupted signal in small temporal folds |
| 3 | XGBoost top-100 features from LightGBM | ❌ XGBoost shifts to C-columns (depth-wise architecture) — Test PR 0.4666, below all other models |
| 4 | amt_vs_product_median (24th feature) | ✅ Ranked #4 in LightGBM importance, Test PR 0.5033→0.5087 |
| 5 | UID anchor fix (floor(TransactionDT/86400 − D1)) | ❌ As UID: groups ×5.3, fraud purity drops. 1st place used it for aggregation features, not as UID. |
| 6 | V3: PR AUC objective + instance weighting | ✅ PR improved, ROC tradeoff — expected at 3.5% fraud rate |
| 7 | Ensemble V2+V3 | ✅ Best across both metrics simultaneously — final model |

---

## Why PR AUC as Primary Metric

With 3.5% fraud rate, ROC AUC has a ceiling effect — nearly any model achieves 0.89+. PR AUC directly measures fraud detection quality on the minority class and is more discriminating between models.

---

## Key Engineering Principles

- **No leakage** — strict fit/transform; feature windows always `closed='left'`
- **No hardcode** — every parameter is a named variable with documented rationale
- **Frozen test set** — touched exactly once; no metric shopping
- **Production thinking** — feature engineering on concat(train+test) mirrors real deployment
- **Modular codebase** — all logic in `.py` modules; notebooks are pure orchestrators
- **Evidence-based** — every experiment compared to statistical baseline on both ROC AUC and PR AUC; failures documented

---

## Project Structure
```
ML_Fraud_detection/
├── data/
│   ├── raw/                            # Original Kaggle CSVs — never modified
│   │   ├── train_transaction.csv
│   │   └── train_identity.csv
│   ├── orig_full_train.parquet         # Full merged dataset
│   ├── train.parquet                   # 60% train split
│   ├── val.parquet                     # 20% val split
│   └── test.parquet                    # 20% frozen test split
├── src/
│   ├── config.py                       # Shared paths, constants, seeds
│   ├── data_loader.py                  # Load/merge/save raw and processed data
│   ├── eda.py                          # EDA analysis functions
│   ├── evaluate_ml.py                  # ROC/PR plots, feature importance
│   ├── feature_init_utils.py           # Initial features from EDA (time, device)
│   ├── pipeline_preprocess.py          # LightGBM/XGBoost preprocessing + Optuna
│   ├── pipeline_evaluate.py            # Model evaluation utilities
│   ├── pipeline_feature_selection.py   # XGBoost top-N experiment
│   ├── preproc_agg.py                  # Aggregation features (18)
│   ├── preproc_behavioral.py           # Behavioral fingerprint features (4)
│   ├── preproc_product.py              # Product profile features (2)
│   ├── preproc_lgbm_xgboost.py         # Label encoding for LightGBM/XGBoost
│   ├── preproc_weights.py              # V3 instance weighting
│   ├── project_utils.py                # Project utilities
│   ├── train_lightgbm.py              # LightGBM training
│   ├── train_xgboost.py               # XGBoost training
│   ├── train_lgbm_v3.py               # V3 LightGBM training
│   ├── train_ensemble.py              # Ensemble utilities
│   ├── train_stacking.py              # OOF stacking (experiment, not used)
│   ├── tune_optuna_with_early_stop.py  # V2 Optuna (ROC AUC objective)
│   └── tune_optuna_v3.py              # V3 Optuna (PR AUC objective)
├── v2/
│   ├── 02_feature_engineering.ipynb    # 24 engineered features
│   ├── 03_preprocess_train_clean_optuna.ipynb  # Preprocessing + Optuna
│   ├── 04_predict_evaluate.ipynb       # Train → Predict → Evaluate (V2)
│   └── 05_feature_selection_xgb.ipynb  # XGBoost top-N experiment
├── v3/
│   └── 04_train_evaluate_v3.ipynb      # V3: PR AUC + instance weighting
├── outputs/
│   ├── enriched/                       # Feature-enriched splits
│   ├── preproc/                        # Preprocessed splits + encoding map
│   ├── models/                         # Saved models (.pkl)
│   ├── best_params_lgbm.json           # V2 Optuna params
│   ├── best_params_lgbm_v3.json        # V3 Optuna params
│   └── best_params_xgb.json            # XGBoost Optuna params
├── 01_eda.ipynb                        # EDA — data exploration + UID validation
├── requirements.txt
└── .gitignore
```

---

## Requirements

- Python 3.9+
- lightgbm, xgboost, optuna
- scikit-learn, pandas, numpy
- See `requirements.txt`

---

## Dataset

Kaggle IEEE-CIS Fraud Detection — Vesta Corporation  
https://www.kaggle.com/competitions/ieee-fraud-detection

*Note: Kaggle test CSVs deleted — project uses own frozen 3-way temporal split.*