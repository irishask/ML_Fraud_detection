"""
train_lightgbm.py — LightGBM Training Module
═════════════════════════════════════════════
Trains LightGBM on preprocessed v1 features with early stopping.
Loads best hyperparameters from Optuna JSON if available,
falls back to sensible defaults otherwise.

WHY a new file instead of reusing train_v0.py:
    train_v0.py is frozen — it belongs to the v0 baseline and must not change.
    This module follows the same interface as train_xgboost.py and
    train_catboost.py: load_params → merge → train → return (model, y_pred_val).
    Consistent interface across all three models is required by train_ensemble.py.

WHY LightGBM in the ensemble:
    LightGBM uses leaf-wise tree growth — different from XGBoost (level-wise)
    and CatBoost (ordered boosting). Each model makes different errors on the
    same data → diversity → stronger ensemble.

Functions:
    train_lgbm() — train LightGBM, return (model, y_pred_val)
"""

import os
import sys

import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

# Ensure v1 modules are importable regardless of how this module is loaded.
_V1_PATH = os.path.dirname(os.path.abspath(__file__))
if _V1_PATH not in sys.path:
    sys.path.append(_V1_PATH)

from tune_optuna import load_params


# ── Default Parameters ────────────────────────────────────────────────────────

# Sensible defaults used when no Optuna JSON is found.
# Carried over from train_v0.py with WHY documented for each value.
# WHY these values:
#   num_leaves=128       : controls tree complexity — deeper than default (31)
#                          to handle 400+ features and ~500k rows
#   learning_rate=0.05   : moderate — balances speed and generalization
#   feature_fraction=0.4 : aggressive column subsampling — with 400+ features,
#                          using all columns leads to correlated trees
#   bagging_fraction=0.8 : row subsampling — reduces overfitting
#   bagging_freq=1       : apply bagging every iteration
#   min_child_samples=50 : higher than default (20) — reduces overfitting on
#                          rare fraud patterns in a 3.5% imbalanced dataset
#   reg_alpha=0.1        : L1 regularization — sparse feature usage
#   reg_lambda=1.0       : L2 regularization — prevents large weights on noisy features
#   is_unbalance=True    : automatically adjusts weights for 3.5% fraud rate —
#                          more robust than manual scale_pos_weight
#   n_estimators=3000    : high ceiling — early stopping controls actual rounds
DEFAULT_PARAMS = {
    "objective":        "binary",
    "metric":           "auc",
    "boosting_type":    "gbdt",
    "is_unbalance":     True,
    "num_leaves":       128,
    "learning_rate":    0.05,
    "feature_fraction": 0.4,
    "bagging_fraction": 0.8,
    "bagging_freq":     1,
    "min_child_samples":50,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "n_estimators":     3000,
    "random_state":     42,
    "n_jobs":           -1,
    "verbose":          -1,
}

# Fixed params — never overridden by JSON or caller.
# WHY separate: Optuna tunes model complexity, not infrastructure settings.
FIXED_PARAMS = {
    "objective":     "binary",
    "metric":        "auc",
    "boosting_type": "gbdt",
    "verbose":       -1,
    "n_jobs":        -1,
}

# Early stopping and logging defaults.
# WHY 200 early stopping: LightGBM plateaus often appear around rounds 300-500;
#   200 rounds of patience is enough to escape local plateaus.
# WHY 100 log period: sufficient visibility without excessive output.
DEFAULT_EARLY_STOPPING = 200
DEFAULT_LOG_PERIOD     = 100


# ── Train ─────────────────────────────────────────────────────────────────────

def train_lgbm(X_train, y_train, X_val, y_val,
               params=None,
               early_stopping_rounds=DEFAULT_EARLY_STOPPING,
               log_period=DEFAULT_LOG_PERIOD,
               params_path="best_params_lgbm.json",
               verbose=True):
    """
    Train LightGBM with early stopping on validation set.

    Parameter priority (highest to lowest):
        1. params argument  — explicitly passed by caller
        2. params_path JSON — best params from Optuna tuning
        3. DEFAULT_PARAMS   — sensible fallback if no JSON found

    WHY consistent interface with train_xgboost.py and train_catboost.py:
        train_ensemble.py calls all three models the same way and collects
        (model, y_pred_val) from each. Consistent interface = no special
        cases in ensemble code.

    Parameters
    ----------
    X_train              : pd.DataFrame — training features
                           (output of preproc_lgbm_xgboost.preprocess_fit)
    y_train              : pd.Series    — training target
    X_val                : pd.DataFrame — validation features
                           (output of preproc_lgbm_xgboost.preprocess_transform)
    y_val                : pd.Series    — validation target
    params               : dict | None  — model params to override defaults/JSON
                           WHY None default: allows JSON and defaults to apply
                           without requiring explicit params on every call
    early_stopping_rounds: int          — stop if val AUC does not improve
                           for this many rounds (default: 200)
                           WHY 200: LightGBM needs more patience than XGBoost/CatBoost
                           due to leaf-wise growth — plateaus appear and resolve later
    log_period           : int          — print val AUC every N rounds (default: 100)
    params_path          : str          — path to Optuna best params JSON
                           WHY parameter not hardcode: path may differ per environment
    verbose              : bool         — print progress and metrics (default: True)

    Returns
    -------
    tuple (model, y_pred_val)
        model        — trained LGBMClassifier
        y_pred_val   — predicted probabilities on val set (np.ndarray, float32)
    """
    if verbose:
        print("=" * 60)
        print("TRAINING — LightGBM")
        print("=" * 60)

    # ── Build final params (priority: explicit > JSON > defaults) ─────────────
    final_params = DEFAULT_PARAMS.copy()

    json_params = load_params(params_path)
    if json_params:
        final_params.update(json_params)
        if verbose:
            print(f"   Loaded Optuna params from: {params_path}")
    else:
        if verbose:
            print("   No Optuna params found — using defaults")

    if params:
        final_params.update(params)
        if verbose:
            print(f"   Overrides applied: {list(params.keys())}")

    # FIXED_PARAMS always win — infrastructure settings must not be overridden
    final_params.update(FIXED_PARAMS)

    if verbose:
        print(f"\n   Final params: {final_params}")
        print(f"   early_stopping_rounds: {early_stopping_rounds}")

    # ── Train ─────────────────────────────────────────────────────────────────
    if verbose:
        print(f"\n   Train: {X_train.shape[0]:,} rows × {X_train.shape[1]} features")
        print(f"   Val:   {X_val.shape[0]:,} rows × {X_val.shape[1]} features")

    model = lgb.LGBMClassifier(**final_params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=verbose),
            lgb.log_evaluation(period=log_period if verbose else -1),
        ],
    )

    if verbose:
        print(f"\n   Best iteration: {model.best_iteration_}")
        print(f"   Best val AUC:   {model.best_score_['valid_0']['auc']:.6f}")

    # ── Predict ───────────────────────────────────────────────────────────────
    # predict_proba returns [prob_class_0, prob_class_1] — take column 1 (fraud)
    y_pred_val = model.predict_proba(X_val)[:, 1].astype(np.float32)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    roc_auc = roc_auc_score(y_val, y_pred_val)
    pr_auc  = average_precision_score(y_val, y_pred_val)

    if verbose:
        print(f"\n{'─' * 60}")
        print(f"   LightGBM Results")
        print(f"{'─' * 60}")
        print(f"   ROC AUC : {roc_auc:.4f}")
        print(f"   PR AUC  : {pr_auc:.4f}")
        print(f"{'─' * 60}")

    return model, y_pred_val