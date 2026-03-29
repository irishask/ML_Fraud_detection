"""
train_stacking.py
─────────────────────────────────────────────────────────────────────────────
Feature Augmentation Stacking — LightGBM → XGBoost.

Approach:
    1. LightGBM OOF (9 temporal folds, expanding window, 10 days each)
       → logit(pred_lgbm_oof) added as new feature to X_train
    2. LightGBM trained on full train
       → logit(pred_lgbm_val), logit(pred_lgbm_test) added to X_val, X_test
    3. XGBoost trained on X_train_aug → predicts X_val_aug, X_test_aug

No-leakage guarantee:
    - Each OOF fold predicts only on days it has NOT seen during training.
    - Val and Test are predicted by LightGBM trained on full train only.
    - XGBoost trains on train data only.
    - Test evaluated once at the end.

Best practices applied:
    - Expanding window OOF (mandatory for temporal data, no shuffle)
    - Simplified LightGBM params for OOF (stability on small folds)
    - Logit transformation of predictions (better scale for tree models)
    - colsample_bytree=0.20 for XGBoost (engineered features exposure)
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb
import xgboost as xgb_lib

# ── OOF fold boundaries ───────────────────────────────────────────────────────
# 9 folds x 10 days each, expanding window.
FOLD_BOUNDARIES = [
    {"train_end": i * 10, "oof_start": i * 10 + 1, "oof_end": i * 10 + 10}
    for i in range(1, 10)
]

# Simplified LightGBM params for OOF — stability over small folds.
# NOT Optuna params: those are tuned for 354K rows, unstable on 38K (Fold 1).
# min_child_samples=50: stronger regularisation to avoid overfitting on small folds.
OOF_LGBM_PARAMS = {
    "n_estimators":      500,
    "num_leaves":        31,
    "learning_rate":     0.05,
    "min_child_samples": 50,
    "is_unbalance":      True,
    "random_state":      42,
    "n_jobs":            -1,
    "verbose":           -1,
}

# XGBoost colsample_bytree — chosen by val PR AUC comparison (Step 4, nb05).
# 0.20: ~92 columns/tree; engineered features (23) get fair exposure vs
# V-columns (336) which dominated at higher colsample values.
COLSAMPLE_BYTREE = 0.20


# ── Helpers ───────────────────────────────────────────────────────────────────

def _logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """
    Convert probability to logit: log(p / (1-p)).
    Logit is more symmetric and linear — better input scale for tree models.
    eps: clip boundary to avoid log(0).
    """
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p)).astype("float32")


def _load_xgb_params(params_path: str) -> dict:
    """Load Optuna XGBoost params from JSON, override colsample_bytree."""
    path = Path(params_path)
    if path.exists():
        with open(path) as f:
            params = json.load(f)
        print(f"   XGBoost params loaded <- {path}")
    else:
        params = {}
        print("   XGBoost params: JSON not found — using defaults")
    params["colsample_bytree"] = COLSAMPLE_BYTREE
    params.setdefault("n_estimators",     3000)
    params.setdefault("learning_rate",    0.05)
    params.setdefault("max_depth",        6)
    params.setdefault("random_state",     42)
    params.setdefault("n_jobs",          -1)
    params.setdefault("eval_metric",     "aucpr")
    params.setdefault("scale_pos_weight", 28)
    print(f"   colsample_bytree overridden -> {COLSAMPLE_BYTREE}")
    return params


def _load_lgbm_params(params_path: str) -> dict:
    """Load Optuna LightGBM params from JSON (used for final full-train model)."""
    path = Path(params_path)
    if path.exists():
        with open(path) as f:
            params = json.load(f)
        print(f"   LightGBM params loaded <- {path}")
    else:
        params = {}
        print("   LightGBM params: JSON not found — using defaults")
    params.setdefault("n_estimators",  3000)
    params.setdefault("learning_rate", 0.05)
    params.setdefault("num_leaves",    31)
    params.setdefault("is_unbalance",  True)
    params.setdefault("random_state",  42)
    params.setdefault("n_jobs",       -1)
    return params


# ── Main public functions ─────────────────────────────────────────────────────

def build_oof_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tx_day_train: pd.Series,
    oof_params: dict = OOF_LGBM_PARAMS,
    fold_boundaries: list = FOLD_BOUNDARIES,
) -> np.ndarray:
    """
    Build LightGBM OOF predictions on train using 9 temporal folds.

    Parameters
    ----------
    X_train, y_train : full train features and target
    tx_day_train     : tx_day column aligned with X_train (for fold masks)
    oof_params       : simplified LightGBM params for OOF stability
    fold_boundaries  : list of {train_end, oof_start, oof_end} dicts

    Returns
    -------
    oof_logit : np.ndarray (float32), shape (len(X_train),)
        logit(pred_lgbm_oof) for each train row.
        Rows not covered by any fold (day 101 boundary) filled with 0.0
        (neutral logit = 0.5 probability).
    """
    oof_proba = np.full(len(X_train), np.nan, dtype="float32")

    for fold_idx, bounds in enumerate(fold_boundaries, 1):
        tr_mask  = tx_day_train <= bounds["train_end"]
        val_mask = (tx_day_train >= bounds["oof_start"]) & \
                   (tx_day_train <= bounds["oof_end"])

        X_tr, y_tr = X_train[tr_mask],  y_train[tr_mask]
        X_vl, y_vl = X_train[val_mask], y_train[val_mask]

        model = lgb.LGBMClassifier(**oof_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_vl, y_vl)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        pred = model.predict_proba(X_vl)[:, 1].astype("float32")
        oof_proba[val_mask] = pred

        roc = roc_auc_score(y_vl, pred)
        pr  = average_precision_score(y_vl, pred)
        print(f"   Fold {fold_idx} "
              f"(train days 1-{bounds['train_end']:3d}: {tr_mask.sum():,} rows | "
              f"oof days {bounds['oof_start']}-{bounds['oof_end']}: {val_mask.sum():,} rows) "
              f"-> ROC={roc:.4f}  PR={pr:.4f}")

    # Rows not covered (e.g. day 101 boundary): fill with neutral logit
    n_missing = np.isnan(oof_proba).sum()
    if n_missing > 0:
        print(f"   {n_missing} rows not covered by OOF -> filled with logit=0.0")
        oof_proba = np.where(np.isnan(oof_proba), 0.5, oof_proba)

    oof_logit = _logit(oof_proba)
    print(f"\n   OOF complete: {len(oof_logit):,} rows | "
          f"logit range [{oof_logit.min():.2f}, {oof_logit.max():.2f}]")
    return oof_logit


def augment_features(
    X: pd.DataFrame,
    lgbm_logit: np.ndarray,
    col_name: str = "lgbm_logit",
) -> pd.DataFrame:
    """
    Add logit(pred_lgbm) as a new column to feature matrix.

    Parameters
    ----------
    X          : original feature DataFrame
    lgbm_logit : logit-transformed LightGBM predictions aligned with X
    col_name   : name of the new augmented column

    Returns
    -------
    X_aug : DataFrame with one extra column appended (copy, original unchanged)
    """
    X_aug = X.copy()
    X_aug[col_name] = lgbm_logit
    return X_aug


def train_lgbm_full(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params_path: str,
    model_save_path: Path,
    early_stopping_rounds: int = 100,
) -> tuple:
    """
    Train LightGBM on full train with Optuna params.
    Used to generate val and test predictions for augmentation.

    Returns
    -------
    (model, y_pred_val) : fitted model + val probabilities (float32)
    """
    params = _load_lgbm_params(params_path)
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    with open(model_save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"   LightGBM saved -> {model_save_path}")

    y_pred_val = model.predict_proba(X_val)[:, 1].astype("float32")
    return model, y_pred_val


def train_xgb_augmented(
    X_train_aug: pd.DataFrame,
    y_train: pd.Series,
    X_val_aug: pd.DataFrame,
    y_val: pd.Series,
    params_path: str,
    model_save_path: Path,
    early_stopping_rounds: int = 100,
) -> tuple:
    """
    Train XGBoost on augmented features (X + lgbm_logit column).

    Returns
    -------
    (model, y_pred_val) : fitted XGBoost booster + val probabilities (float32)
    """
    params = _load_xgb_params(params_path)

    dtrain = xgb_lib.DMatrix(X_train_aug, label=y_train)
    dval   = xgb_lib.DMatrix(X_val_aug,   label=y_val)

    xgb_p = {k: v for k, v in params.items()
              if k not in ("n_estimators", "random_state", "n_jobs")}
    xgb_p["seed"]    = params.get("random_state", 42)
    xgb_p["nthread"] = params.get("n_jobs", -1)

    model = xgb_lib.train(
        xgb_p, dtrain,
        num_boost_round=params.get("n_estimators", 3000),
        evals=[(dval, "val")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=100,
    )
    with open(model_save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"   XGBoost saved -> {model_save_path}")

    y_pred_val = model.predict(
        dval, iteration_range=(0, model.best_iteration + 1)
    ).astype("float32")
    return model, y_pred_val


def print_results_table(
    y_true: np.ndarray,
    predictions: dict,
    v0_roc: float,
    v0_pr: float,
    label: str = "VAL",
) -> pd.DataFrame:
    """Print formatted results table vs V0 baseline."""
    rows = []
    for name, y_pred in predictions.items():
        rows.append({
            "Model":   name,
            "ROC AUC": roc_auc_score(y_true, y_pred),
            "PR AUC":  average_precision_score(y_true, y_pred),
        })
    df = pd.DataFrame(rows)
    df["Delta ROC"] = df["ROC AUC"].apply(lambda x: f"{x - v0_roc:+.4f}")
    df["Delta PR"]  = df["PR AUC"].apply(lambda x: f"{x - v0_pr:+.4f}")
    print(f"\n{'='*60}")
    print(f"  {label} RESULTS vs V0 Baseline")
    print(f"{'='*60}")
    print(df.to_string(index=False))
    print(f"{'='*60}")
    return df
