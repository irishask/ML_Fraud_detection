"""
tune_optuna.py — Hyperparameter Optimization via Optuna (Bayesian TPE)
═══════════════════════════════════════════════════════════════════════
Tunes LightGBM and XGBoost independently using Optuna's
Tree-structured Parzen Estimator (TPE) — a Bayesian optimization algorithm.

WHY Optuna over Grid/Random Search:
    - Grid Search:   impractical — 8-10 parameters × 3 values = 59,000+ combinations
    - Random Search: no learning between trials — inefficient for large search spaces
    - Optuna TPE:    learns from previous trials → finds optimum in 50-100 trials
    - Pruning:       Optuna stops unpromising trials early → faster search

WHY each model is tuned independently:
    Each model has different hyperparameters and different sensitivities.
    Independent tuning maximizes each model's individual strength.
    A stronger individual model → stronger ensemble.

WHY subsampling for tuning (tune_frac < 1.0):
    Hyperparameters like num_leaves, learning_rate, regularization are stable
    across dataset sizes — optimal values on 50% data are very close to optimal
    on 100% data. Subsampling cuts trial time proportionally with minimal
    quality loss. Final training in train_*.py always uses the full dataset.

WHY reduced rounds for tuning:
    Tuning finds the best hyperparameter region, not the exact tree count.
    Early stopping handles termination. Final training uses 2000 rounds
    for full convergence.

Quality/latency balance is controlled by the `quality` parameter:
    "min"  — fast run,    ~0.7h,  ~0.004 AUC loss vs high
    "med"  — balanced,    ~3h,    ~0.001 AUC loss vs high
    "high" — full search, ~18h,   baseline (no loss)

Overfitting risk: minimal — each trial evaluates on time-based val split
with early stopping. Val is never used for training.

Best parameters are saved to JSON and loaded automatically by train_*.py.

Functions:
    tune_lgbm()   — tune LightGBM, save best_params_lgbm.json
    tune_xgb()    — tune XGBoost,  save best_params_xgb.json
    load_params() — load params from JSON file
"""

import os
import sys
import json

import numpy as np
import optuna
from sklearn.metrics import roc_auc_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Ensure v0 modules are importable regardless of how this module is loaded.
_V0_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v0")
if _V0_PATH not in sys.path:
    sys.path.append(_V0_PATH)


# ── Quality / Latency Profiles ────────────────────────────────────────────────

# Controls the trade-off between tuning quality and runtime.
# Pass quality="min" | "med" | "high" to each tune_*() function.
#
# Parameters explained:
#   n_trials        : Optuna trials per model
#                     WHY matters: TPE first ~15 trials are random exploration;
#                     remaining trials are guided. More trials = better coverage.
#   tune_frac       : fraction of X_train used per trial
#                     WHY matters: hyperparameters are stable across sizes —
#                     optimal values on 50% ≈ optimal on 100%. Cuts time ~2x.
#                     Final training always uses full dataset.
#   num_boost_round : max boosting iterations per trial
#                     WHY matters: tuning finds the best region, not exact count.
#                     Early stopping handles termination. Lower = faster trials.
#   early_stopping  : stop trial if val AUC does not improve for N rounds
#                     WHY matters: aggressive = faster but may miss slow-converging
#                     configs (low learning_rate). Conservative = slower but safer.
#   expected_time_h : approximate wall-clock time for all 3 models combined
#   expected_auc_loss: estimated ROC AUC loss vs "high" profile on final model

QUALITY_PROFILES = {
    "min": {
        "n_trials":          30,
        "tune_frac":         0.3,    # 141k rows
        "num_boost_round":   500,
        "early_stopping":    30,
        "expected_time_h":   "~0.7h",
        "expected_auc_loss": "~0.004",
    },
    "med": {
        "n_trials":          50,
        "tune_frac":         0.5,    # 236k rows
        "num_boost_round":   1000,
        "early_stopping":    50,
        "expected_time_h":   "~3h",
        "expected_auc_loss": "~0.001",
    },
    "high": {
        "n_trials":          100,
        "tune_frac":         1.0,    # 472k rows — full dataset
        "num_boost_round":   2000,
        "early_stopping":    100,
        "expected_time_h":   "~18h",
        "expected_auc_loss": "0 (reference)",
    },
}

# Default quality level used when not specified explicitly.
DEFAULT_QUALITY = "med"

# Fixed random seed for reproducibility across trials.
RANDOM_SEED = 42

# Class imbalance ratio for scale_pos_weight (XGBoost).
# WHY 28: neg/pos ≈ (1 - 0.035) / 0.035 ≈ 27.6 → rounded to 28.
SCALE_POS_WEIGHT = 28


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_params(path):
    """
    Load best hyperparameters from a JSON file saved by a previous tuning run.

    Parameters
    ----------
    path : str — path to JSON file (e.g. 'outputs/best_params_xgb.json')

    Returns
    -------
    dict — hyperparameters, or empty dict if file does not exist
    """
    if not os.path.exists(path):
        print(f"   [tune_optuna] No params file found at {path} — using defaults.")
        return {}
    with open(path) as f:
        params = json.load(f)
    print(f"   [tune_optuna] Loaded params from {path}")
    return params


def _save_params(params, path):
    """
    Save best hyperparameters to a JSON file.

    Parameters
    ----------
    params : dict — best hyperparameters from Optuna study
    path   : str  — output path
    """
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"   [tune_optuna] Best params saved to {path}")


def _get_profile(quality):
    """
    Resolve quality string to profile dict with validation.

    Parameters
    ----------
    quality : str — "min" | "med" | "high"

    Returns
    -------
    dict — profile from QUALITY_PROFILES
    """
    if quality not in QUALITY_PROFILES:
        raise ValueError(
            f"Unknown quality level: '{quality}'. "
            f"Choose from: {list(QUALITY_PROFILES.keys())}"
        )
    return QUALITY_PROFILES[quality]


def _subsample(X_train, y_train, frac, random_state):
    """
    Subsample training data for faster tuning trials.

    WHY not stratified: fraud rate (3.5%) on even 30% of data = ~5k fraud
    cases — sufficient for stable gradient estimates. Simple random sample
    is faster and avoids sklearn dependency here.

    Parameters
    ----------
    X_train      : pd.DataFrame
    y_train      : pd.Series
    frac         : float — fraction to keep (e.g. 0.5); 1.0 = no subsampling
    random_state : int

    Returns
    -------
    tuple (X_tune, y_tune)
    """
    if frac >= 1.0:
        return X_train, y_train
    idx = X_train.sample(frac=frac, random_state=random_state).index
    return X_train.loc[idx], y_train.loc[idx]


# ── Tune: LightGBM ────────────────────────────────────────────────────────────

def tune_lgbm(X_train, y_train, X_val, y_val,
              quality=DEFAULT_QUALITY,
              save_path="best_params_lgbm.json",
              verbose=True):
    """
    Find best LightGBM hyperparameters using Optuna TPE.

    WHY Dataset recreated inside each trial (not shared outside):
        LightGBM Dataset caches feature splits based on the first trial's
        min_child_samples. When Optuna suggests a smaller value in a later
        trial, LightGBM raises LightGBMError: "Reducing min_data_in_leaf
        with feature_pre_filter=true may cause unexpected behaviour."
        Recreating Dataset per trial gives each trial a clean state.
        XGBoost DMatrix and CatBoost Pool do NOT have this issue —
        they are static containers that don't cache training params.

    Search space:
        num_leaves        : [20, 300]   — model complexity
        learning_rate     : [0.01, 0.1] — generalization
        feature_fraction  : [0.4, 1.0]  — column subsampling per tree
        bagging_fraction  : [0.4, 1.0]  — row subsampling per iteration
        min_child_samples : [20, 300]   — min samples per leaf
        lambda_l1         : [0, 5]      — L1 regularization
        lambda_l2         : [0, 5]      — L2 regularization

    Parameters
    ----------
    X_train   : pd.DataFrame — full training features (after preprocess_fit)
    y_train   : pd.Series    — full training target
    X_val     : pd.DataFrame — validation features
    y_val     : pd.Series    — validation target
    quality   : str          — "min" | "med" | "high" (default: "med")
                               controls n_trials, tune_frac, num_boost_round,
                               early_stopping — see QUALITY_PROFILES
    save_path : str          — path to save best params JSON
    verbose   : bool         — print progress (default: True)

    Returns
    -------
    dict — best hyperparameters found by Optuna
    """
    import lightgbm as lgb

    profile       = _get_profile(quality)
    n_trials      = profile["n_trials"]
    tune_frac     = profile["tune_frac"]
    num_rounds    = profile["num_boost_round"]
    early_stop    = profile["early_stopping"]

    if verbose:
        print("=" * 60)
        print(f"OPTUNA TUNING — LightGBM | quality='{quality}'")
        print(f"   trials={n_trials}, frac={tune_frac}, "
              f"rounds={num_rounds}, early_stop={early_stop}")
        print(f"   Expected time: {profile['expected_time_h']} (this model only)")
        print("=" * 60)

    X_tune, y_tune = _subsample(X_train, y_train, tune_frac, RANDOM_SEED)

    if verbose:
        print(f"   Tuning on {len(X_tune):,} rows "
              f"({int(tune_frac*100)}% of {len(X_train):,})")

    def objective(trial):
        params = {
            "objective":         "binary",
            "metric":            "auc",
            "verbosity":         -1,
            "boosting_type":     "gbdt",
            "is_unbalance":      True,
            "seed":              RANDOM_SEED,
            "num_leaves":        trial.suggest_int("num_leaves", 20, 300),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "feature_fraction":  trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction":  trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq":      1,
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 300),
            "lambda_l1":         trial.suggest_float("lambda_l1", 0.0, 5.0),
            "lambda_l2":         trial.suggest_float("lambda_l2", 0.0, 5.0),
        }

        # WHY recreated per trial: see docstring above.
        dtrain = lgb.Dataset(X_tune, label=y_tune)
        dval   = lgb.Dataset(X_val,  label=y_val, reference=dtrain)

        callbacks = [lgb.early_stopping(early_stop, verbose=False),
                     lgb.log_evaluation(-1)]

        model = lgb.train(
            params, dtrain,
            num_boost_round=num_rounds,
            valid_sets=[dval],
            callbacks=callbacks,
        )

        y_pred = model.predict(X_val)
        return roc_auc_score(y_val, y_pred)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(objective, n_trials=n_trials,
                   show_progress_bar=verbose)

    best_params = study.best_params
    best_score  = study.best_value

    if verbose:
        print(f"\n   Best ROC AUC : {best_score:.6f}")
        print(f"   Best params  : {best_params}")

    _save_params(best_params, save_path)
    return best_params


# ── Tune: XGBoost ─────────────────────────────────────────────────────────────

def tune_xgb(X_train, y_train, X_val, y_val,
             quality=DEFAULT_QUALITY,
             save_path="best_params_xgb.json",
             verbose=True):
    """
    Find best XGBoost hyperparameters using Optuna TPE.

    WHY DMatrix created once outside objective:
        XGBoost DMatrix is a static data container — it does not cache
        training params. Safe to reuse across all trials, unlike LightGBM Dataset.

    Search space:
        max_depth         : [4, 12]
        eta               : [0.01, 0.1]
        subsample         : [0.6, 1.0]
        colsample_bytree  : [0.4, 1.0]
        colsample_bylevel : [0.4, 1.0]  — XGBoost-specific, increases diversity
        min_child_weight  : [50, 300]
        lambda            : [0, 5]
        alpha             : [0, 5]

    Parameters
    ----------
    X_train   : pd.DataFrame — full training features (after preprocess_fit)
    y_train   : pd.Series    — full training target
    X_val     : pd.DataFrame — validation features
    y_val     : pd.Series    — validation target
    quality   : str          — "min" | "med" | "high" (default: "med")
    save_path : str          — path to save best params JSON
    verbose   : bool         — print progress (default: True)

    Returns
    -------
    dict — best hyperparameters found by Optuna
    """
    import xgboost as xgb

    profile       = _get_profile(quality)
    n_trials      = profile["n_trials"]
    tune_frac     = profile["tune_frac"]
    num_rounds    = profile["num_boost_round"]
    early_stop    = profile["early_stopping"]

    if verbose:
        print("=" * 60)
        print(f"OPTUNA TUNING — XGBoost | quality='{quality}'")
        print(f"   trials={n_trials}, frac={tune_frac}, "
              f"rounds={num_rounds}, early_stop={early_stop}")
        print(f"   Expected time: {profile['expected_time_h']} (this model only)")
        print("=" * 60)

    X_tune, y_tune = _subsample(X_train, y_train, tune_frac, RANDOM_SEED)

    if verbose:
        print(f"   Tuning on {len(X_tune):,} rows "
              f"({int(tune_frac*100)}% of {len(X_train):,})")

    # WHY outside objective: DMatrix is a static container —
    # does not cache training params. Safe to reuse across all trials.
    dtrain = xgb.DMatrix(X_tune, label=y_tune)
    dval   = xgb.DMatrix(X_val,  label=y_val)

    def objective(trial):
        params = {
            "objective":         "binary:logistic",
            "eval_metric":       "auc",
            "tree_method":       "hist",
            "scale_pos_weight":  SCALE_POS_WEIGHT,
            "seed":              RANDOM_SEED,
            "verbosity":         0,
            "max_depth":         trial.suggest_int("max_depth", 4, 12),
            "eta":               trial.suggest_float("eta", 0.01, 0.1, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 50, 300),
            "lambda":            trial.suggest_float("lambda", 0.0, 5.0),
            "alpha":             trial.suggest_float("alpha", 0.0, 5.0),
        }

        model = xgb.train(
            params, dtrain,
            num_boost_round=num_rounds,
            evals=[(dval, "val")],
            early_stopping_rounds=early_stop,
            verbose_eval=False,
        )

        y_pred = model.predict(dval)
        return roc_auc_score(y_val, y_pred)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(objective, n_trials=n_trials,
                   show_progress_bar=verbose)

    best_params = study.best_params
    best_score  = study.best_value

    if verbose:
        print(f"\n   Best ROC AUC : {best_score:.6f}")
        print(f"   Best params  : {best_params}")

    _save_params(best_params, save_path)
    return best_params

