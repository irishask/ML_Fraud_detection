"""
tune_optuna_v3.py — Hyperparameter Optimization via Optuna (V3)
═══════════════════════════════════════════════════════════════
V3 changes vs tune_optuna_with_early_stop.py:

    1. OBJECTIVE: PR AUC (average_precision_score) instead of ROC AUC
       WHY: PR AUC is the primary metric for fraud detection with 3.5%
       class imbalance. ROC AUC exhibits ceiling effects — near any
       non-degenerate classifier achieves 0.90+ regardless of fraud
       detection quality. PR AUC directly measures precision-recall
       tradeoff on the minority class.

    2. SAMPLE WEIGHTS: instance weighting via stratified temporal sampling
       WHY: recent transactions are more representative of current fraud
       patterns. Optuna trials sample proportionally from each temporal
       chunk — guaranteeing all time periods are represented while
       preserving the weight signal for model training.

    3. LightGBM ONLY: XGBoost tuning removed
       WHY: XGBoost showed Test PR 0.4977 (below baseline 0.5033) due
       to architectural limitations — level-wise growth ignores engineered
       features regardless of hyperparameters. Tuning XGBoost for PR AUC
       would not address this root cause.

All other logic (TPE sampler, study-level early stopping, quality profiles,
progress bar) is identical to tune_optuna_with_early_stop.py.

Functions:
    tune_lgbm_v3() — tune LightGBM for PR AUC with instance weights
    load_params()  — load params from JSON (same as V2)
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import average_precision_score
from tqdm.auto import tqdm


# ── Study-level early stopping ────────────────────────────────────────────────

class EarlyStoppingCallback:
    """Stop Optuna study if best PR AUC does not improve for `patience` trials."""
    def __init__(self, patience, min_delta=0.0):
        self.patience   = patience
        self.min_delta  = min_delta
        self._best      = None
        self._no_improv = 0

    def __call__(self, study, trial):
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        if self._best is None or trial.value > self._best + self.min_delta:
            self._best      = trial.value
            self._no_improv = 0
        else:
            self._no_improv += 1
        if self._no_improv >= self.patience:
            study.stop()


optuna.logging.set_verbosity(optuna.logging.WARNING)

_SRC_PATH = os.path.dirname(os.path.abspath(__file__))
if _SRC_PATH not in sys.path:
    sys.path.append(_SRC_PATH)


# ── Quality profiles ──────────────────────────────────────────────────────────
# Identical to V2 — same quality/latency tradeoff logic applies.

QUALITY_PROFILES = {
    "min":      {"n_trials": 30,  "tune_frac": 0.30, "num_boost_round": 500,  "early_stopping": 50,  "expected_time_h": 0.7},
    "med":      {"n_trials": 50,  "tune_frac": 0.50, "num_boost_round": 1000, "early_stopping": 100, "expected_time_h": 3.0},
    "med_high": {"n_trials": 75,  "tune_frac": 0.75, "num_boost_round": 1500, "early_stopping": 150, "expected_time_h": 9.0},
    "high":     {"n_trials": 100, "tune_frac": 1.00, "num_boost_round": 2000, "early_stopping": 200, "expected_time_h": 18.0},
}

DEFAULT_QUALITY  = "med_high"
RANDOM_SEED      = 42
STUDY_PATIENCE   = 15
STUDY_MIN_DELTA  = 0.0001


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_params(path):
    """
    Load best hyperparameters from a JSON file.

    Parameters
    ----------
    path : str — path to JSON file

    Returns
    -------
    dict — hyperparameters, or empty dict if file does not exist
    """
    if not os.path.exists(path):
        print(f"   [tune_optuna_v3] No params file at {path} — using defaults.")
        return {}
    with open(path) as f:
        params = json.load(f)
    print(f"   [tune_optuna_v3] Loaded params from {path}")
    return params


def _save_params(params, path):
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"   [tune_optuna_v3] Best params saved to {path}")


def _get_profile(quality):
    if quality not in QUALITY_PROFILES:
        raise ValueError(
            f"Unknown quality level: '{quality}'. "
            f"Choose from: {list(QUALITY_PROFILES.keys())}"
        )
    return QUALITY_PROFILES[quality]


def _make_progress_callback(pbar):
    def _callback(study, trial):
        pbar.set_postfix({"best_pr_auc": f"{study.best_value:.4f}"})
        pbar.update(1)
    return _callback


def _stratified_time_sample(X_train, y_train, sample_weight,
                             tune_frac, time_col="tx_day",
                             chunk_days=None, random_state=42):
    """
    Sample tune_frac% from each temporal chunk independently.

    WHY stratified by chunk (not random):
        Random sampling of 75% is statistically ~equal across time,
        but explicit stratification GUARANTEES each temporal period
        is represented in every Optuna trial. This is especially
        important for the most recent chunk (days 76–101) which has
        the highest weights — ensuring it is never accidentally
        undersampled in a trial.

    WHY preserve weights in sampled data:
        Each sampled row retains its original weight. LightGBM receives
        the same weight signal regardless of which rows were selected —
        the relative weight ratios between chunks are preserved.

    Parameters
    ----------
    X_train       : pd.DataFrame
    y_train       : pd.Series
    sample_weight : pd.Series — per-row weights from compute_sample_weights()
    tune_frac     : float     — fraction to sample from each chunk
    time_col      : str       — day column name
    chunk_days    : list of (int,int) | None — chunk boundaries
    random_state  : int

    Returns
    -------
    tuple (X_tune, y_tune, w_tune)
    """
    from preproc_weights import DEFAULT_CHUNK_DAYS
    chunk_days = chunk_days or DEFAULT_CHUNK_DAYS

    rng = np.random.RandomState(random_state)
    sampled_indices = []

    for (d_start, d_end) in chunk_days:
        mask = (X_train[time_col] >= d_start) & (X_train[time_col] <= d_end)
        chunk_idx = X_train.index[mask]
        n_sample = max(1, int(len(chunk_idx) * tune_frac))
        chosen = rng.choice(chunk_idx, size=min(n_sample, len(chunk_idx)),
                            replace=False)
        sampled_indices.append(chosen)

    all_idx = np.concatenate(sampled_indices)

    return (
        X_train.loc[all_idx],
        y_train.loc[all_idx],
        sample_weight.loc[all_idx],
    )


# ── Tune: LightGBM V3 ─────────────────────────────────────────────────────────

def tune_lgbm_v3(X_train, y_train, X_val, y_val,
                 sample_weight,
                 quality=DEFAULT_QUALITY,
                 save_path="best_params_lgbm_v3.json",
                 verbose=True):
    """
    Find best LightGBM hyperparameters using Optuna TPE.
    Optimizes PR AUC with instance weighting.

    V3 changes vs tune_lgbm():
        - Objective: average_precision_score (PR AUC) instead of roc_auc_score
        - sample_weight passed to lgb.Dataset → fit() uses weighted gradients
        - Stratified temporal sampling per chunk (not random subsample)
        - Saves to best_params_lgbm_v3.json (not best_params_lgbm.json)

    WHY PR AUC as Optuna objective:
        ROC AUC has ceiling effects at 3.5% fraud rate — nearly any model
        achieves 0.90+. PR AUC directly measures fraud detection quality
        on the minority class and is more discriminating between models.
        See: preprints.org/manuscript/202510.0958

    WHY sample_weight in Optuna trials:
        Each trial must see the same weight signal as final training.
        Without weights, Optuna finds params optimal for unweighted data —
        those params may not be optimal for weighted training.

    WHY val without weights:
        Val is used for unbiased early stopping and PR AUC evaluation.
        Applying weights to val would distort the stopping criterion.

    Parameters
    ----------
    X_train       : pd.DataFrame — training features (460 cols)
    y_train       : pd.Series    — training target
    X_val         : pd.DataFrame — validation features
    y_val         : pd.Series    — validation target
    sample_weight : pd.Series    — per-row weights from compute_sample_weights()
                    WHY required (not optional): V3 is specifically designed
                    for weighted training — calling without weights defeats the purpose
    quality       : str          — "min"|"med"|"med_high"|"high" (default: "med_high")
    save_path     : str          — path to save best params JSON
    verbose       : bool         — print progress (default: True)

    Returns
    -------
    dict — best hyperparameters found by Optuna
    """
    import lightgbm as lgb

    profile    = _get_profile(quality)
    n_trials   = profile["n_trials"]
    tune_frac  = profile["tune_frac"]
    num_rounds = profile["num_boost_round"]
    early_stop = profile["early_stopping"]

    if verbose:
        print("=" * 60)
        print(f"OPTUNA TUNING V3 — LightGBM | quality='{quality}'")
        print(f"   Objective    : PR AUC (average_precision_score)")
        print(f"   Weights      : 4 temporal chunks [1.0, 2.0, 3.0, 4.0]")
        print(f"   trials={n_trials}, frac={tune_frac}, "
              f"rounds={num_rounds}, early_stop={early_stop}")
        print(f"   Expected time: {profile['expected_time_h']}h (this model only)")
        print("=" * 60)

    # Stratified sample from each temporal chunk
    X_tune, y_tune, w_tune = _stratified_time_sample(
        X_train, y_train, sample_weight,
        tune_frac=tune_frac,
        random_state=RANDOM_SEED,
    )

    if verbose:
        print(f"\n   Tuning on {len(X_tune):,} rows "
              f"({int(tune_frac*100)}% stratified from {len(X_train):,})")
        print(f"   Weight distribution in tune set:")
        vc = w_tune.value_counts().sort_index()
        for w_val, count in vc.items():
            print(f"     weight={w_val:.1f}: {count:,} rows")

    def objective(trial):
        params = {
            "objective":         "binary",
            "metric":            "average_precision",  # WHY: monitor PR AUC during training
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

        # WHY recreated per trial: LightGBM Dataset caches feature splits
        # based on min_child_samples — must recreate to avoid LightGBMError.
        # WHY weight=w_tune: each trial uses weighted gradients so Optuna
        # finds params optimal for weighted training, not unweighted.
        dtrain = lgb.Dataset(X_tune, label=y_tune, weight=w_tune)
        dval   = lgb.Dataset(X_val,  label=y_val,  reference=dtrain)

        callbacks = [
            lgb.early_stopping(early_stop, verbose=False),
            lgb.log_evaluation(-1),
        ]

        model = lgb.train(
            params, dtrain,
            num_boost_round=num_rounds,
            valid_sets=[dval],
            callbacks=callbacks,
        )

        # WHY predict on full val (no weights): unbiased PR AUC evaluation
        y_pred = model.predict(X_val)
        return average_precision_score(y_val, y_pred)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )

    early_stop_study = EarlyStoppingCallback(
        patience=STUDY_PATIENCE,
        min_delta=STUDY_MIN_DELTA,
    )

    if verbose:
        with tqdm(total=n_trials, desc="Optuna LightGBM V3", unit="trial") as pbar:
            study.optimize(
                objective, n_trials=n_trials,
                callbacks=[_make_progress_callback(pbar), early_stop_study],
            )
    else:
        study.optimize(
            objective, n_trials=n_trials,
            callbacks=[early_stop_study],
        )

    best_params = study.best_params
    best_score  = study.best_value

    if verbose:
        print(f"\n   Best PR AUC  : {best_score:.6f}")
        print(f"   Best params  : {best_params}")

    _save_params(best_params, save_path)
    return best_params
