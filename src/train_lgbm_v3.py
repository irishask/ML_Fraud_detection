"""
train_lgbm_v3.py — LightGBM Training with Instance Weights (V3)
════════════════════════════════════════════════════════════════
V3 changes vs train_lightgbm.py:

    1. SAMPLE WEIGHTS: accepts sample_weight parameter → passed to fit()
       WHY: temporal chunk weighting gives recent transactions stronger
       gradient influence — recent fraud patterns dominate training.

    2. PARAMS PATH: loads from best_params_lgbm_v3.json (not best_params_lgbm.json)
       WHY: V3 Optuna optimized for PR AUC with weighted data — different
       optimal params than V2 ROC AUC optimization.

    3. METRIC MONITORING: LightGBM monitors 'average_precision' during training
       WHY: consistent with Optuna objective — early stopping tracks the same
       metric that was optimized. Using 'auc' for early stopping while
       optimizing PR AUC in Optuna would be inconsistent.

All other logic (parameter priority, fixed params, early stopping, logging)
is identical to train_lightgbm.py.

Functions:
    train_lgbm_v3() — train LightGBM with weights, return (model, y_pred_val)
"""

import os
import sys

import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

_SRC_PATH = os.path.dirname(os.path.abspath(__file__))
if _SRC_PATH not in sys.path:
    sys.path.append(_SRC_PATH)

from tune_optuna_v3 import load_params


# ── Default Parameters ────────────────────────────────────────────────────────
# Same as V2 defaults — used only when no JSON is found.

DEFAULT_PARAMS = {
    "objective":        "binary",
    "metric":           "average_precision",  # V3: monitor PR AUC during training
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
FIXED_PARAMS = {
    "objective":     "binary",
    "metric":        "average_precision",  # V3: always monitor PR AUC
    "boosting_type": "gbdt",
    "verbose":       -1,
    "n_jobs":        -1,
}

DEFAULT_EARLY_STOPPING = 200
DEFAULT_LOG_PERIOD     = 100


# ── Train ─────────────────────────────────────────────────────────────────────

def train_lgbm_v3(X_train, y_train, X_val, y_val,
                  sample_weight,
                  params=None,
                  early_stopping_rounds=DEFAULT_EARLY_STOPPING,
                  log_period=DEFAULT_LOG_PERIOD,
                  params_path="best_params_lgbm_v3.json",
                  verbose=True):
    """
    Train LightGBM with instance weights and early stopping on validation set.

    V3 changes vs train_lgbm():
        - sample_weight required → passed to fit() for weighted gradient updates
        - Loads from best_params_lgbm_v3.json (PR AUC optimized params)
        - Monitors average_precision (PR AUC) during training for early stopping

    WHY sample_weight in training (not just Optuna):
        Optuna found params optimal for weighted training. Final training
        must also use weights — otherwise params are mismatched.

    WHY early stopping monitors PR AUC (not ROC AUC):
        Consistent with Optuna objective. Stopping on ROC AUC while
        optimizing PR AUC could stop too early or too late for PR AUC.

    WHY val without weights for early stopping:
        Val evaluation must be unbiased — applying weights to val would
        distort when early stopping fires.

    Parameter priority (highest to lowest):
        1. params argument  — explicitly passed by caller
        2. params_path JSON — best params from V3 Optuna (PR AUC objective)
        3. DEFAULT_PARAMS   — fallback if no JSON found

    Parameters
    ----------
    X_train              : pd.DataFrame — training features (460 cols)
    y_train              : pd.Series    — training target
    X_val                : pd.DataFrame — validation features
    y_val                : pd.Series    — validation target
    sample_weight        : pd.Series    — per-row weights from compute_sample_weights()
                           WHY required: V3 is specifically designed for weighted
                           training — None would defeat the purpose
    params               : dict | None  — override defaults/JSON
    early_stopping_rounds: int          — patience for val PR AUC (default: 200)
    log_period           : int          — print val score every N rounds (default: 100)
    params_path          : str          — path to V3 Optuna params JSON
    verbose              : bool         — print progress and metrics (default: True)

    Returns
    -------
    tuple (model, y_pred_val)
        model      — trained LGBMClassifier
        y_pred_val — predicted probabilities on val set (np.ndarray, float32)
    """
    if verbose:
        print("=" * 60)
        print("TRAINING V3 — LightGBM (PR AUC + Instance Weights)")
        print("=" * 60)

    # ── Build final params ────────────────────────────────────────────────────
    final_params = DEFAULT_PARAMS.copy()

    json_params = load_params(params_path)
    if json_params:
        final_params.update(json_params)
        if verbose:
            print(f"   Loaded V3 Optuna params from: {params_path}")
    else:
        if verbose:
            print("   No V3 Optuna params found — using defaults")

    if params:
        final_params.update(params)
        if verbose:
            print(f"   Overrides applied: {list(params.keys())}")

    # FIXED_PARAMS always win
    final_params.update(FIXED_PARAMS)

    if verbose:
        print(f"\n   Final params: {final_params}")
        print(f"   early_stopping_rounds: {early_stopping_rounds}")
        print(f"\n   Train: {X_train.shape[0]:,} rows × {X_train.shape[1]} features")
        print(f"   Val:   {X_val.shape[0]:,} rows × {X_val.shape[1]} features")
        print(f"\n   Weight distribution in train:")
        vc = sample_weight.value_counts().sort_index()
        for w_val, count in vc.items():
            print(f"     weight={w_val:.1f}: {count:,} rows")

    # ── Train ─────────────────────────────────────────────────────────────────
    model = lgb.LGBMClassifier(**final_params)

    model.fit(
        X_train, y_train,
        sample_weight=sample_weight,   # WHY: weighted gradient updates
        eval_set=[(X_val, y_val)],     # WHY no weight on val: unbiased eval
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=verbose),
            lgb.log_evaluation(period=log_period if verbose else -1),
        ],
    )

    if verbose:
        print(f"\n   Best iteration: {model.best_iteration_}")

    # ── Predict ───────────────────────────────────────────────────────────────
    y_pred_val = model.predict_proba(X_val)[:, 1].astype(np.float32)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    roc_auc = roc_auc_score(y_val, y_pred_val)
    pr_auc  = average_precision_score(y_val, y_pred_val)

    if verbose:
        print(f"\n{'─' * 60}")
        print(f"   LightGBM V3 Results")
        print(f"{'─' * 60}")
        print(f"   ROC AUC : {roc_auc:.4f}")
        print(f"   PR AUC  : {pr_auc:.4f}  ← primary metric")
        print(f"{'─' * 60}")

    return model, y_pred_val
