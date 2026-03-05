"""
pipeline_preprocess.py — Preprocessing Pipeline for V2
════════════════════════════════════════════════════════
Contains all logic for 03_preprocess_train.ipynb.
The notebook is a pure orchestrator — no logic lives in notebook cells.

Pipeline steps (called in order by the notebook):
    1. load_enriched()         — load train_enriched + y_train, verify alignment
    2. split_train_val()       — time-based 80/20 split, verify temporal ordering
    3. preprocess_and_save()   — fit encoding on train, transform val, save artifacts
       OR load_preprocessed()  — load saved artifacts (when PREPROC_READY=True)
    4. run_optuna_lgbm()       — Optuna HPO for LightGBM (when RUN_OPTUNA=True)
    5. All validation and summary logic lives in the notebook (Step 6).

No-leakage guarantee:
    Label encoding is fit ONLY on X_train (after time split).
    X_val is transformed using the encoding map from X_train — never refitted.
    isFraud is never used in any feature transformation.

Functions:
    load_enriched()       — load enriched features + target, verify index alignment
    split_train_val()     — time-based train/val split, verify temporal ordering
    preprocess_and_save() — fit + transform + save preprocessed splits and artifacts
    load_preprocessed()   — load saved preprocessed splits and artifacts
    run_optuna_lgbm()     — run Optuna HPO for LightGBM, save best params JSON

Note on raw splits:
    preprocess_and_save() saves X_train_raw / X_val_raw BEFORE label encoding.
    04_predict_evaluate.ipynb. LightGBM and XGBoost use X_train_lgbm / X_val_lgbm.
"""

import os
import sys
import pickle

import pandas as pd

# ---------------------------------------------------------------------------
# Ensure v0/ and src/ are importable when this module is loaded from any location.
# WHY: pipeline_preprocess.py lives in src/; it imports from v0/ and other src/ modules.
# os.path.dirname(__file__) gives src/ → join("..", "v0") gives project_root/v0/
# ---------------------------------------------------------------------------
_SRC_PATH = os.path.dirname(os.path.abspath(__file__))
_V0_PATH  = os.path.join(_SRC_PATH, "..", "v0")
for _p in [_SRC_PATH, _V0_PATH]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from split_v0 import time_split
from preproc_lgbm_xgboost import preprocess_fit, preprocess_transform
from tune_optuna import tune_lgbm
from config import TIME_COL, TRAIN_RATIO, NON_FEATURE_COLS


# ── Step 1: Load enriched data ────────────────────────────────────────────────

def load_enriched(enriched_dir, verbose=True):
    """
    Load train_enriched.parquet and y_train.parquet from enriched_dir.
    Verifies that their indexes are aligned before returning.

    WHY index verification: train_enriched and y_train are saved separately
    in 02_feature_engineering.ipynb with index=True. If files are regenerated
    independently or partially, index mismatch would silently corrupt labels.
    Fail fast here rather than getting wrong metrics later.

    Parameters
    ----------
    enriched_dir : Path — directory containing train_enriched.parquet and y_train.parquet
                   WHY Path not str: Path objects are OS-agnostic and composable
                   with / operator — no string concatenation bugs on Windows
    verbose      : bool — print progress (default: True)

    Returns
    -------
    tuple (train_enriched, y_train_full)
        train_enriched : pd.DataFrame — enriched features, no target column
        y_train_full   : pd.Series    — isFraud target aligned by index
    """
    if verbose:
        print("=" * 60)
        print("STEP 1 — Load enriched data")
        print("=" * 60)

    train_enriched = pd.read_parquet(enriched_dir / "train_enriched.parquet")
    y_train_full   = pd.read_parquet(enriched_dir / "y_train.parquet")["isFraud"]

    # Index alignment: both files must have identical index to guarantee
    # that row i in train_enriched corresponds to row i in y_train_full
    if not train_enriched.index.equals(y_train_full.index):
        raise ValueError(
            "Index mismatch between train_enriched and y_train_full. "
            "Re-run 02_feature_engineering.ipynb to regenerate aligned files."
        )

    if verbose:
        print(f"   train_enriched : {train_enriched.shape}")
        print(f"   y_train_full   : {y_train_full.shape}")
        print(f"   Fraud rate     : {y_train_full.mean():.4%}")
        print(f"   Index alignment: OK ✓")

    return train_enriched, y_train_full


# ── Step 2: Time-based train/val split ───────────────────────────────────────

def split_train_val(train_enriched, y_train_full,
                    time_col=TIME_COL,
                    train_ratio=TRAIN_RATIO,
                    verbose=True):
    """
    Split enriched train data into train and validation sets by time.

    WHY time-based split: fraud patterns evolve over time. A random split
    would leak future behavioral patterns into the training set and
    overestimate validation performance — misrepresenting real deployment.

    WHY 80/20: standard split for this dataset size (~590k rows).
    Val set (~118k rows) is large enough for stable AUC estimates.
    Defined in config.py as TRAIN_RATIO — not hardcoded here.

    Parameters
    ----------
    train_enriched : pd.DataFrame — enriched features from load_enriched()
    y_train_full   : pd.Series    — full target from load_enriched()
    time_col       : str          — timestamp column for ordering (default: TIME_COL)
    train_ratio    : float        — fraction of data used for training (default: 0.80)
                     WHY named param: mirrors config.py TRAIN_RATIO —
                     single source of truth, no magic 0.8 in code
    verbose        : bool         — print progress (default: True)

    Returns
    -------
    tuple (X_train, X_val, y_train, y_val)
        X_train : pd.DataFrame — training features
        X_val   : pd.DataFrame — validation features
        y_train : pd.Series    — training target
        y_val   : pd.Series    — validation target
    """
    if verbose:
        print("=" * 60)
        print("STEP 2 — Time-based train/val split")
        print("=" * 60)

    X_train, X_val, y_train, y_val = time_split(
        train_enriched, y_train_full,
        time_col=time_col,
        train_ratio=train_ratio,
    )

    # Temporal ordering check: every train timestamp must be strictly before
    # every val timestamp. Failure here means time_split() has a bug.
    if X_train[time_col].max() >= X_val[time_col].min():
        raise ValueError(
            f"Temporal leak detected: train DT max ({X_train[time_col].max()}) "
            f">= val DT min ({X_val[time_col].min()}). "
            "Check time_split() implementation."
        )

    if verbose:
        print(f"   X_train : {X_train.shape}  | fraud rate: {y_train.mean():.4%}")
        print(f"   X_val   : {X_val.shape}    | fraud rate: {y_val.mean():.4%}")
        print(f"   Train DT max : {X_train[time_col].max()}")
        print(f"   Val   DT min : {X_val[time_col].min()}")
        print(f"   Temporal ordering: OK ✓")

    return X_train, X_val, y_train, y_val


# ── Step 3a: Preprocess and save ─────────────────────────────────────────────

def preprocess_and_save(X_train, X_val, y_train, y_val,
                         preproc_dir,
                         cols_to_drop=None,
                         fill_value=-1,
                         verbose=True):
    """
    Fit label encoding on X_train, transform X_val, save all artifacts.

    WHY fit on X_train only: fitting on X_val would leak val category
    distribution into the encoding — a form of data leakage. The encoding
    map learned here is also saved for use on test in 04_predict_evaluate.ipynb.

    WHY save encoding_map.pkl: test set preprocessing in the next notebook
    must use the exact same encoders as training. Saving guarantees consistency
    even if the notebook is restarted between sessions.

    Parameters
    ----------
    X_train      : pd.DataFrame — training features (output of split_train_val)
    X_val        : pd.DataFrame — validation features
    y_train      : pd.Series    — training target
    y_val        : pd.Series    — validation target
    preproc_dir  : Path         — directory to save all preprocessed artifacts
    cols_to_drop : list[str]    — non-feature columns to remove
                   WHY default None → NON_FEATURE_COLS: defined once in config.py,
                   passed explicitly so the caller controls what is dropped
    fill_value   : numeric      — NaN fill value (default: -1)
                   WHY -1: consistent with all preprocessing modules in v0/v1;
                   LightGBM learns -1 as a "missing" signal
    verbose      : bool         — print progress (default: True)

    Returns
    -------
    tuple (X_train_lgbm, X_val_lgbm, encoding_map)
        X_train_lgbm : pd.DataFrame — label-encoded features for LightGBM / XGBoost
        X_val_lgbm   : pd.DataFrame — label-encoded features for LightGBM / XGBoost
        encoding_map : dict         — fitted label encoders (column → encoder)

    Also saves to preproc_dir:
        X_train_raw.parquet — features BEFORE label encoding
        X_val_raw.parquet   — features BEFORE label encoding
    """
    cols_to_drop = cols_to_drop or NON_FEATURE_COLS

    if verbose:
        print("=" * 60)
        print("STEP 3 — Preprocess and save (LightGBM)")
        print("=" * 60)

    # Save raw splits BEFORE encoding.
    # its internal ordered target statistics. Label-encoded integers cause
    preproc_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_parquet(preproc_dir / "X_train_raw.parquet", index=True)
    X_val.to_parquet(  preproc_dir / "X_val_raw.parquet",   index=True)

    if verbose:
        print(f"   Raw splits saved:")
        print(f"     X_train_raw : {X_train.shape}")
        print(f"     X_val_raw   : {X_val.shape}")
        print(f"     → {preproc_dir / 'X_train_raw.parquet'}")
        print(f"     → {preproc_dir / 'X_val_raw.parquet'}")
        print()

    # Fit encoding on train only — no val information used
    X_train_lgbm, encoding_map = preprocess_fit(
        X_train,
        cols_to_drop=cols_to_drop,
        fill_value=fill_value,
        verbose=verbose,
    )

    # Transform val using encoding map from train
    X_val_lgbm = preprocess_transform(
        X_val,
        encoding_map=encoding_map,
        cols_to_drop=cols_to_drop,
        fill_value=fill_value,
        verbose=verbose,
    )

    # Save label-encoded feature splits — index=True preserves row alignment
    X_train_lgbm.to_parquet(preproc_dir / "X_train_lgbm.parquet", index=True)
    X_val_lgbm.to_parquet(  preproc_dir / "X_val_lgbm.parquet",   index=True)

    # Save targets — index=True keeps them aligned with X splits
    y_train.to_frame().to_parquet(preproc_dir / "y_train.parquet", index=True)
    y_val.to_frame().to_parquet(  preproc_dir / "y_val.parquet",   index=True)

    # Save encoding map as pickle — needed for test set transformation
    with open(preproc_dir / "encoding_map.pkl", "wb") as f:
        pickle.dump(encoding_map, f)

    if verbose:
        print(f"\n   Label-encoded splits saved (for LightGBM / XGBoost):")
        print(f"     X_train_lgbm : {X_train_lgbm.shape}")
        print(f"     X_val_lgbm   : {X_val_lgbm.shape}")
        print(f"     → {preproc_dir / 'X_train_lgbm.parquet'}")
        print(f"     → {preproc_dir / 'X_val_lgbm.parquet'}")
        print(f"     encoding_map : {len(encoding_map)} encoders")
        print(f"     y_train / y_val saved to {preproc_dir}")

    return X_train_lgbm, X_val_lgbm, encoding_map


# ── Step 3b: Load preprocessed (PREPROC_READY=True) ──────────────────────────

def load_preprocessed(preproc_dir, verbose=True):
    """
    Load previously saved preprocessed splits and encoding artifacts.

    Called instead of preprocess_and_save() when PREPROC_READY=True.
    Allows re-running the notebook without repeating expensive preprocessing.

    WHY load y_train/y_val here (not from split_train_val): when PREPROC_READY=True,
    split_train_val() is also skipped. The saved y files are the ground truth —
    they are aligned with the saved X files by construction (saved together).

    Parameters
    ----------
    preproc_dir : Path — directory containing saved preprocessed artifacts
    verbose     : bool — print progress (default: True)

    Returns
    -------
    tuple (X_train_lgbm, X_val_lgbm, X_train_raw, X_val_raw, encoding_map, y_train, y_val)
        X_train_lgbm : pd.DataFrame — label-encoded features for LightGBM / XGBoost
        X_val_lgbm   : pd.DataFrame — label-encoded features for LightGBM / XGBoost
        X_train_raw  : pd.DataFrame — raw features BEFORE encoding
        X_val_raw    : pd.DataFrame — raw features BEFORE encoding
        encoding_map : dict         — fitted label encoders
        y_train      : pd.Series    — training target
        y_val        : pd.Series    — validation target
    """
    if verbose:
        print("=" * 60)
        print("STEP 3 — Load preprocessed (PREPROC_READY=True)")
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
        print(f"   X_train_raw  : {X_train_raw.shape}")
        print(f"   X_val_raw    : {X_val_raw.shape}")
        print(f"   encoding_map : {len(encoding_map)} encoders")
        print(f"   y_train fraud rate: {y_train.mean():.4%}")
        print(f"   y_val   fraud rate: {y_val.mean():.4%}")

    return X_train_lgbm, X_val_lgbm, X_train_raw, X_val_raw, encoding_map, y_train, y_val


# ── Step 4: Optuna HPO ────────────────────────────────────────────────────────

def run_optuna_lgbm(X_train_lgbm, y_train, X_val_lgbm, y_val,
                    outputs_dir,
                    quality="med",
                    verbose=True):
    """
    Run Optuna hyperparameter optimization for LightGBM.

    Saves best params to outputs_dir/best_params_lgbm.json.
    train_lightgbm.py loads this file automatically.

    WHY quality='med' as default: 50 trials on 50% data (~3h) gives a good
    balance between search quality and runtime. Switch to 'high' for final
    training before submission.

    Quality profiles:
        'min'  — 30 trials, 30% data, ~0.7h  | ~0.004 AUC loss vs high
        'med'  — 50 trials, 50% data, ~3h    | ~0.001 AUC loss vs high
        'high' — 100 trials, 100% data, ~18h | reference (no loss)

    Parameters
    ----------
    X_train_lgbm : pd.DataFrame — preprocessed training features
    y_train      : pd.Series    — training target
    X_val_lgbm   : pd.DataFrame — preprocessed validation features
    y_val        : pd.Series    — validation target
    outputs_dir  : Path         — directory to save best_params_lgbm.json
                   WHY separate from preproc_dir: JSON params are model-level
                   artifacts, not preprocessing artifacts — different lifecycle
    quality      : str          — 'min' | 'med' | 'high' (default: 'med')
                   WHY named param: caller controls the quality/time tradeoff
                   without modifying this function
    verbose      : bool         — print progress (default: True)

    Returns
    -------
    dict — best hyperparameters found by Optuna
    """
    if verbose:
        print("=" * 60)
        print(f"STEP 4 — Optuna HPO | quality='{quality}'")
        print("=" * 60)

    best_params = tune_lgbm(
        X_train_lgbm, y_train,
        X_val_lgbm,   y_val,
        quality=quality,
        save_path=str(outputs_dir / "best_params_lgbm.json"),
        verbose=verbose,
    )

    if verbose:
        print(f"\n   Best params: {best_params}")
        print(f"   Saved to: {outputs_dir / 'best_params_lgbm.json'}")

    return best_params