"""
pipeline_evaluate.py — Training and Evaluation Pipeline for V2
════════════════════════════════════════════════════════════════
Contains all logic for 04_predict_evaluate.ipynb.
The notebook is a pure orchestrator — no logic lives in notebook cells.

Pipeline steps (called in order by the notebook):
    1. load_splits()       — load preprocessed splits from outputs/preproc/
    2. train_lgbm_model()  — train LightGBM, save model to outputs/models/
       train_xgb_model()   — train XGBoost,  save model to outputs/models/

Model save formats:
    LightGBM → outputs/models/model_lgbm.pkl  (pickle — sklearn API)
    XGBoost  → outputs/models/model_xgb.pkl   (pickle — Booster via pickle)

Functions:
    load_splits()       — load all preprocessed splits and encoding artifacts
    train_lgbm_model()  — train LightGBM, save, return (model, y_pred_val)
    train_xgb_model()   — train XGBoost,  save, return (model, y_pred_val)
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure src/ and v0/ are importable from any working directory.
# WHY: pipeline_evaluate.py lives in src/; it imports from src/ and v0/.
# ---------------------------------------------------------------------------
_SRC_PATH = os.path.dirname(os.path.abspath(__file__))
_V0_PATH  = os.path.join(_SRC_PATH, "..", "v0")
for _p in [_SRC_PATH, _V0_PATH]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from train_lightgbm import train_lgbm
from train_xgboost  import train_xgb
from config         import NON_FEATURE_COLS


# ── Step 1: Load preprocessed splits ─────────────────────────────────────────

def load_splits(preproc_dir, verbose=True):
    """
    Load all preprocessed splits saved by 03_preprocess_train_clean.ipynb.

    Parameters
    ----------
    preproc_dir : Path — directory containing preprocessed artifacts
                  (outputs/preproc/)
    verbose     : bool — print shapes and fraud rates (default: True)

    Returns
    -------
    tuple (X_train_lgbm, X_val_lgbm, X_train_raw, X_val_raw,
           encoding_map, y_train, y_val)
        X_train_lgbm : pd.DataFrame — label-encoded train features (LightGBM / XGBoost)
        X_val_lgbm   : pd.DataFrame — label-encoded val features   (LightGBM / XGBoost)
        X_train_raw  : pd.DataFrame — raw train features (kept for compatibility)
        X_val_raw    : pd.DataFrame — raw val features   (kept for compatibility)
        encoding_map : dict         — fitted label encoders
        y_train      : pd.Series    — training target
        y_val        : pd.Series    — validation target
    """
    if verbose:
        print("=" * 60)
        print("STEP 1 — Load preprocessed splits")
        print("=" * 60)

    X_train_lgbm = pd.read_parquet(preproc_dir / "X_train_lgbm.parquet")
    X_val_lgbm   = pd.read_parquet(preproc_dir / "X_val_lgbm.parquet")
    X_train_raw  = pd.read_parquet(preproc_dir / "X_train_raw.parquet")
    X_val_raw    = pd.read_parquet(preproc_dir / "X_val_raw.parquet")
    y_train      = pd.read_parquet(preproc_dir / "y_train.parquet")["isFraud"]
    y_val        = pd.read_parquet(preproc_dir / "y_val.parquet")["isFraud"]

    with open(preproc_dir / "encoding_map.pkl", "rb") as f:
        encoding_map = pickle.load(f)

    if verbose:
        print(f"   X_train_lgbm : {X_train_lgbm.shape}  (label-encoded, LightGBM / XGBoost)")
        print(f"   X_val_lgbm   : {X_val_lgbm.shape}    (label-encoded, LightGBM / XGBoost)")
        print(f"   X_train_raw  : {X_train_raw.shape}   (raw, for CatBoost)")
        print(f"   X_val_raw    : {X_val_raw.shape}     (raw, for CatBoost)")
        print(f"   encoding_map : {len(encoding_map)} encoders")
        print(f"   y_train fraud rate : {y_train.mean():.4%}")
        print(f"   y_val   fraud rate : {y_val.mean():.4%}")

    return (X_train_lgbm, X_val_lgbm, X_train_raw, X_val_raw,
            encoding_map, y_train, y_val)


# ── Step 2a: Train LightGBM ───────────────────────────────────────────────────

def train_lgbm_model(X_train_lgbm, y_train, X_val_lgbm, y_val,
                     models_dir,
                     params_path="best_params_lgbm.json",
                     verbose=True):
    """
    Train LightGBM on preprocessed splits and save the model to disk.

    WHY save as pickle: LightGBMClassifier uses sklearn API — pickle is the
    standard serialization format. Loading is straightforward:
        with open(path, 'rb') as f: model = pickle.load(f)

    Parameters
    ----------
    X_train_lgbm : pd.DataFrame — label-encoded training features
    y_train      : pd.Series    — training target
    X_val_lgbm   : pd.DataFrame — label-encoded validation features
    y_val        : pd.Series    — validation target
    models_dir   : Path         — directory to save model_lgbm.pkl
    params_path  : str          — path to best_params_lgbm.json
    verbose      : bool         — print progress (default: True)

    Returns
    -------
    tuple (model, y_pred_val)
        model      — trained LGBMClassifier
        y_pred_val — predicted probabilities on val set (np.ndarray, float32)
    """
    if verbose:
        print("=" * 60)
        print("STEP 2 — Train LightGBM")
        print("=" * 60)

    model, y_pred_val = train_lgbm(
        X_train_lgbm, y_train,
        X_val_lgbm,   y_val,
        params_path=params_path,
        verbose=verbose,
    )

    models_dir.mkdir(parents=True, exist_ok=True)
    save_path = models_dir / "model_lgbm.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(model, f)

    if verbose:
        print(f"\n   Model saved → {save_path}")

    return model, y_pred_val


# ── Step 2b: Train XGBoost ────────────────────────────────────────────────────

def train_xgb_model(X_train_lgbm, y_train, X_val_lgbm, y_val,
                    models_dir,
                    params_path="best_params_xgb.json",
                    verbose=True):
    """
    Train XGBoost on preprocessed splits and save the model to disk.

    WHY same X_train_lgbm as LightGBM:
        XGBoost and LightGBM both require label-encoded numeric features.
        preproc_lgbm_xgboost.py was designed for both models — no separate
        preprocessing needed.

    WHY save as pickle: xgb.Booster does not have a sklearn-compatible
    save_model() that preserves best_iteration context. Pickle preserves
    the full Booster object including best_iteration for correct prediction.

    Parameters
    ----------
    X_train_lgbm : pd.DataFrame — label-encoded training features (same as LightGBM)
    y_train      : pd.Series    — training target
    X_val_lgbm   : pd.DataFrame — label-encoded validation features
    y_val        : pd.Series    — validation target
    models_dir   : Path         — directory to save model_xgb.pkl
    params_path  : str          — path to best_params_xgb.json
    verbose      : bool         — print progress (default: True)

    Returns
    -------
    tuple (model, y_pred_val)
        model      — trained xgb.Booster
        y_pred_val — predicted probabilities on val set (np.ndarray, float32)
    """
    if verbose:
        print("=" * 60)
        print("STEP 2 — Train XGBoost")
        print("=" * 60)

    model, y_pred_val = train_xgb(
        X_train_lgbm, y_train,
        X_val_lgbm,   y_val,
        params_path=params_path,
        verbose=verbose,
    )

    models_dir.mkdir(parents=True, exist_ok=True)
    save_path = models_dir / "model_xgb.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(model, f)

    if verbose:
        print(f"\n   Model saved → {save_path}")

    return model, y_pred_val