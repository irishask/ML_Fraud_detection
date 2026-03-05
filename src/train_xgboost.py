"""
train_xgboost.py — XGBoost Training Module
═══════════════════════════════════════════
Trains XGBoost on preprocessed v1 features with early stopping.
Loads best hyperparameters from Optuna JSON if available,
falls back to sensible defaults otherwise.

WHY XGBoost in the ensemble:
    XGBoost uses level-wise tree growth and colsample_bylevel — both
    different from LightGBM's leaf-wise growth. This means XGBoost
    makes different errors on the same data → diversity → stronger ensemble.

WHY DMatrix:
    XGBoost's native data format. Pre-computes histograms for all features
    (tree_method='hist'), stores features and labels together for fast memory
    access, and eliminates redundant conversion on each training call.
    Especially important during Optuna tuning — DMatrix is created once
    and reused across all 100 trials.

Functions:
    train_xgb() — train XGBoost, return (model, y_pred_val)
"""

import os
import sys

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score

# Ensure v1 modules are importable regardless of how this module is loaded.
_V1_PATH = os.path.dirname(os.path.abspath(__file__))
if _V1_PATH not in sys.path:
    sys.path.append(_V1_PATH)

from tune_optuna import load_params


# ── Default Parameters ────────────────────────────────────────────────────────

# Sensible defaults used when no Optuna JSON is found.
# WHY these values:
#   max_depth=8        : deeper than LGBM default — captures complex fraud patterns
#   eta=0.05           : moderate learning rate — balance speed vs generalization
#   subsample=0.8      : row subsampling — reduces overfitting on imbalanced data
#   colsample_bytree=0.5  : aggressive column subsampling — increases diversity vs LGBM
#   colsample_bylevel=0.5 : XGBoost-specific — additional subsampling per tree level,
#                           no LightGBM equivalent → key source of ensemble diversity
#   min_child_weight=100  : high value — fraud class is rare (3.5%), prevents splits
#                           on tiny fraud subgroups that would overfit
#   scale_pos_weight=28   : neg/pos ratio ≈ 96.5/3.5 ≈ 28 — compensates class imbalance
#   seed=42            : reproducibility
DEFAULT_PARAMS = {
    "objective":         "binary:logistic",
    "eval_metric":       "auc",
    "tree_method":       "hist",
    "scale_pos_weight":  28,
    "max_depth":         8,
    "eta":               0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.5,
    "colsample_bylevel": 0.5,
    "min_child_weight":  100,
    "seed":              42,
    "verbosity":         0,
}

# Fixed params that must always be set regardless of Optuna results.
# WHY separate: Optuna tunes model complexity params, not infrastructure params.
# These are never overridden by JSON.
FIXED_PARAMS = {
    "objective":    "binary:logistic",
    "eval_metric":  "auc",
    "tree_method":  "hist",
    "verbosity":    0,
}

# Default training rounds and early stopping.
# WHY 2000 rounds: with eta=0.05 and early stopping, actual rounds ~300-800.
# WHY 100 early stopping: slower learning rate needs more patience —
#   50 rounds would stop too early before the model finds the optimum.
DEFAULT_NUM_ROUNDS          = 2000
DEFAULT_EARLY_STOPPING      = 100


# ── Train ─────────────────────────────────────────────────────────────────────

def train_xgb(X_train, y_train, X_val, y_val,
              params=None,
              num_rounds=DEFAULT_NUM_ROUNDS,
              early_stopping_rounds=DEFAULT_EARLY_STOPPING,
              params_path="best_params_xgb.json",
              verbose=True):
    """
    Train XGBoost with early stopping on validation set.

    Parameter priority (highest to lowest):
        1. params argument  — explicitly passed by caller
        2. params_path JSON — best params from Optuna tuning
        3. DEFAULT_PARAMS   — sensible fallback if no JSON found

    WHY this priority order: caller always has final control; Optuna params
    are preferred over defaults when available; defaults ensure the module
    works even before tuning has been run.

    WHY DMatrix over DataFrame:
        DMatrix pre-computes histograms for all features (tree_method='hist'),
        stores features and labels together for fast memory access, and
        eliminates redundant conversion on each call. Created once, reused
        for training and prediction.

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
    num_rounds           : int          — max boosting rounds (default: 2000)
                           WHY 2000: with early stopping actual rounds ~300-800;
                           high ceiling ensures early stopping drives termination
    early_stopping_rounds: int          — stop if val AUC does not improve
                           for this many rounds (default: 100)
                           WHY 100: slower eta needs more patience than LGBM's 50
    params_path          : str          — path to Optuna best params JSON
                           WHY parameter not hardcode: path may differ per environment
    verbose              : bool         — print progress and metrics (default: True)

    Returns
    -------
    tuple (model, y_pred_val)
        model        — trained xgb.Booster
        y_pred_val   — predicted probabilities on val set (np.ndarray, float32)
    """
    if verbose:
        print("=" * 60)
        print("TRAINING — XGBoost")
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
        print(f"   num_rounds: {num_rounds} (with early stopping={early_stopping_rounds})")

    # ── Build DMatrix ─────────────────────────────────────────────────────────
    # DMatrix: XGBoost's native optimized data format.
    # Pre-computes histograms, stores features + labels together for fast access.
    # Created once here — reused for training, evaluation, and prediction.
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val,   label=y_val)

    if verbose:
        print(f"\n   Train: {dtrain.num_row():,} rows × {dtrain.num_col()} features")
        print(f"   Val:   {dval.num_row():,} rows × {dval.num_col()} features")

    # ── Train ─────────────────────────────────────────────────────────────────
    evals_result = {}

    model = xgb.train(
        final_params,
        dtrain,
        num_boost_round=num_rounds,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=100 if verbose else False,
    )

    if verbose:
        print(f"\n   Best iteration: {model.best_iteration}")
        print(f"   Best val AUC:   {model.best_score:.6f}")

    # ── Predict ───────────────────────────────────────────────────────────────
    # iteration_range: predict using only rounds up to best_iteration —
    # avoids using rounds that overfit beyond the early stopping point
    y_pred_val = model.predict(
        dval,
        iteration_range=(0, model.best_iteration + 1)
    ).astype(np.float32)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    roc_auc = roc_auc_score(y_val, y_pred_val)
    pr_auc  = average_precision_score(y_val, y_pred_val)

    if verbose:
        print(f"\n{'─' * 60}")
        print(f"   XGBoost Results")
        print(f"{'─' * 60}")
        print(f"   ROC AUC : {roc_auc:.4f}")
        print(f"   PR AUC  : {pr_auc:.4f}")
        print(f"{'─' * 60}")

    return model, y_pred_val