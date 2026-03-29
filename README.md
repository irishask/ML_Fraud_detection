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

**~14.8x improvement in PR AUC over statistical baseline.**  
Comparable to top 20% of 6,381 Kaggle teams — evaluated on a strictly frozen test set, touched exactly once.

---

## Architecture

### 3-Way Temporal Split (60/20/20)
- **Train:** days 1–101 (354,324 rows) — model training
- **Val:** days 101–141 (118,108 rows) — early stopping + Optuna objective
- **Test:** days 141–183 (118,108 rows) — **frozen, touched exactly once**

### Feature Engineering — 24 New Features
All features computed on full dataset (train+test concatenated), using strictly prior transactions only (`expanding().shift(1)`, `rolling(closed='left')`). Zero leakage by design.

- **Aggregation (16):** transaction history per card — velocity, amounts, email/device diversity
- **Behavioral Fingerprint (4):** deviation from card's personal spending pattern
- **Product Profile (2):** purchase pattern per ProductCD

**UID = card1 + addr1** — reconstructed user identity, confirmed by 5 independent analyses.

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
| 1 | V2: Feature Engineering + Optuna (ROC AUC) | LightGBM ✅ improved vs statistical baseline. XGBoost + CatBoost below statistical baseline. |
| 2 | OOF Stacking (LightGBM → XGBoost) | ❌ 3.5% fraud rate → corrupted signal in small temporal folds |
| 3 | XGBoost top-100 features from LightGBM | ❌ Still selected V-columns — all numerical, no enforcement possible |
| 4 | V3: PR AUC objective + instance weighting | ✅ PR improved, ROC tradeoff — expected at 3.5% fraud rate |
| 5 | Ensemble V2+V3 | ✅ Best across both metrics simultaneously — final model |

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
├── src/
│   ├── config.py
│   ├── pipeline_preprocess.py      # Feature engineering
│   ├── pipeline_evaluate.py        # Evaluation utilities
│   ├── preproc_weights.py          # V3 instance weighting
│   ├── tune_optuna_v3.py           # V3 Optuna (PR AUC objective)
│   └── train_lgbm_v3.py            # V3 LightGBM training
├── v0/                             # First model version notebooks
├── v3/                             # V3 notebooks
│   └── 04_train_evaluate_v3.ipynb
├── outputs/
│   ├── preproc/                    # Preprocessed splits
│   ├── models/                     # Saved models
│   └── best_params_lgbm_v3.json    # V3 Optuna params
└── README.md
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
