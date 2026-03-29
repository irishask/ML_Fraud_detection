"""
pipeline_preprocess.py — Preprocessing Pipeline for V2
════════════════════════════════════════════════════════
Contains all logic for 03_preprocess_train.ipynb.
The notebook is a pure orchestrator — no logic lives in notebook cells.

Pipeline steps (called in order by the notebook):
    1. load_enriched()         — load train/val/test enriched splits, verify alignment
    2. preprocess_and_save()   — fit encoding on train, transform val + frozen TEST, save artifacts
       OR load_preprocessed()  — load saved artifacts (when PREPROC_READY=True)
    3. run_optuna_lgbm()       — Optuna HPO for LightGBM (when RUN_OPTUNA=True)
    4. All validation and summary logic lives in the notebook.

No-leakage guarantee:
    Label encoding is fit ONLY on X_train (60% split).
    X_val and X_test are transformed using the encoding map from X_train — never refitted.
    isFraud is never used in any feature transformation.

Functions:
    load_enriched()       — load all three enriched splits + targets, verify index alignment
    preprocess_and_save() — fit + transform + save preprocessed splits and artifacts
    load_preprocessed()   — load saved preprocessed splits and artifacts
    run_optuna_lgbm()     — run Optuna HPO for LightGBM, save best params JSON
    print_preprocessing_summary()  — print shapes, dtypes, NaN checks, saved files check
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

from preproc_lgbm_xgboost import preprocess_fit, preprocess_transform
from config import TIME_COL, NON_FEATURE_COLS

from tune_optuna_with_early_stop import tune_lgbm


# ── Step 1: Load enriched data ────────────────────────────────────────────────

def load_enriched(enriched_dir, verbose=True):
    """
    Load all three enriched splits (train / val / test) and their targets.

    WHY load all three here: in the 3-way split architecture (60/20/20),
    train_enriched, val_enriched, and test_enriched are produced together by
    02_feature_engineering.ipynb. preprocess_and_save() needs all three to
    fit encoding on train, transform val and frozen TEST in one consistent pass.

    WHY index verification: all files are saved separately with index=True.
    If any file is regenerated independently, index mismatch would silently
    corrupt label alignment. Fail fast here rather than getting wrong metrics later.

    Parameters
    ----------
    enriched_dir : Path — directory containing enriched parquet files
                   WHY Path not str: Path objects are OS-agnostic and composable
                   with / operator — no string concatenation bugs on Windows
    verbose      : bool — print progress (default: True)

    Returns
    -------
    tuple (train_enriched, y_train, val_enriched, y_val, test_enriched, y_test)
        train_enriched : pd.DataFrame — 60% train features
        y_train        : pd.Series    — 60% train labels
        val_enriched   : pd.DataFrame — 20% val features (early stopping + Optuna)
        y_val          : pd.Series    — 20% val labels
        test_enriched  : pd.DataFrame — 20% frozen TEST features (touched once)
        y_test         : pd.Series    — 20% frozen TEST labels
    """
    if verbose:
        print("=" * 60)
        print("STEP 1 — Load enriched data (train / val / frozen TEST)")
        print("=" * 60)

    train_enriched = pd.read_parquet(enriched_dir / "train_enriched.parquet")
    y_train        = pd.read_parquet(enriched_dir / "y_train.parquet")["isFraud"]
    val_enriched   = pd.read_parquet(enriched_dir / "val_enriched.parquet")
    y_val          = pd.read_parquet(enriched_dir / "y_val.parquet")["isFraud"]
    test_enriched  = pd.read_parquet(enriched_dir / "test_enriched.parquet")
    y_test         = pd.read_parquet(enriched_dir / "y_test.parquet")["isFraud"]

    # Index alignment checks — all feature/label pairs must be perfectly aligned
    for name, X, y in [
        ("train", train_enriched, y_train),
        ("val",   val_enriched,   y_val),
        ("test",  test_enriched,  y_test),
    ]:
        if not X.index.equals(y.index):
            raise ValueError(
                f"Index mismatch between {name}_enriched and y_{name}. "
                "Re-run 02_feature_engineering.ipynb to regenerate aligned files."
            )

    if verbose:
        print(f"   train_enriched : {train_enriched.shape}  | fraud rate: {y_train.mean():.4%}")
        print(f"   val_enriched   : {val_enriched.shape}    | fraud rate: {y_val.mean():.4%}")
        print(f"   test_enriched  : {test_enriched.shape}   | fraud rate: {y_test.mean():.4%}  (frozen TEST)")
        print(f"   Index alignment: OK ✓")

    return train_enriched, y_train, val_enriched, y_val, test_enriched, y_test


# ── Step 2: Preprocess and save ──────────────────────────────────────────────

def preprocess_and_save(X_train, X_val, X_test,
                         y_train, y_val, y_test,
                         preproc_dir,
                         cols_to_drop=None,
                         fill_value=-1,
                         verbose=True):
    """
    Fit label encoding on X_train, transform X_val and X_test, save all artifacts.

    WHY fit on X_train only: fitting on val or test would leak their category
    distribution into the encoding — a form of data leakage. The encoding map
    learned here is saved for reproducibility and future inference.

    WHY transform X_test here: frozen TEST uses the same encoder as train/val.
    This mirrors production — new transactions are transformed with the encoder
    fitted on historical data, never refitted on incoming data.

    WHY save encoding_map.pkl: guarantees that any future re-run of notebook 04
    uses the exact same encoders as training, even after a kernel restart.

    Parameters
    ----------
    X_train      : pd.DataFrame — 60% train features (from load_enriched)
    X_val        : pd.DataFrame — 20% val features
    X_test       : pd.DataFrame — 20% frozen TEST features (touched once at final eval)
    y_train      : pd.Series    — 60% train labels
    y_val        : pd.Series    — 20% val labels
    y_test       : pd.Series    — 20% frozen TEST labels
    preproc_dir  : Path         — directory to save all preprocessed artifacts
    cols_to_drop : list[str]    — non-feature columns to remove
                   WHY default None → NON_FEATURE_COLS: defined once in config.py,
                   passed explicitly so the caller controls what is dropped
    fill_value   : numeric      — NaN fill value (default: -1)
                   WHY -1: consistent with all preprocessing modules in v0/v1;
                   LightGBM learns -1 as a missing signal
    verbose      : bool         — print progress (default: True)

    Returns
    -------
    tuple (X_train_lgbm, X_val_lgbm, X_test_lgbm, encoding_map)
        X_train_lgbm : pd.DataFrame — label-encoded train features
        X_val_lgbm   : pd.DataFrame — label-encoded val features
        X_test_lgbm  : pd.DataFrame — label-encoded frozen TEST features
        encoding_map : dict         — fitted label encoders (column → encoder)

    Also saves to preproc_dir:
        X_train_lgbm.parquet, X_val_lgbm.parquet, X_test_lgbm.parquet
        y_train.parquet, y_val.parquet, y_test.parquet
        encoding_map.pkl
    """
    cols_to_drop = cols_to_drop or NON_FEATURE_COLS

    if verbose:
        print("=" * 60)
        print("STEP 2 — Preprocess and save (train / val / frozen TEST)")
        print("=" * 60)

    preproc_dir.mkdir(parents=True, exist_ok=True)

    # Fit encoding on train only — no val or test information used
    X_train_lgbm, encoding_map = preprocess_fit(
        X_train,
        cols_to_drop=cols_to_drop,
        fill_value=fill_value,
        verbose=verbose,
    )

    # Transform val and frozen TEST using encoding map from train
    X_val_lgbm = preprocess_transform(
        X_val,
        encoding_map=encoding_map,
        cols_to_drop=cols_to_drop,
        fill_value=fill_value,
        verbose=verbose,
    )

    X_test_lgbm = preprocess_transform(
        X_test,
        encoding_map=encoding_map,
        cols_to_drop=cols_to_drop,
        fill_value=fill_value,
        verbose=verbose,
    )

    # Save label-encoded feature splits — index=True preserves row alignment
    X_train_lgbm.to_parquet(preproc_dir / "X_train_lgbm.parquet", index=True)
    X_val_lgbm.to_parquet(  preproc_dir / "X_val_lgbm.parquet",   index=True)
    X_test_lgbm.to_parquet( preproc_dir / "X_test_lgbm.parquet",  index=True)

    # Save targets — index=True keeps them aligned with X splits
    y_train.to_frame().to_parquet(preproc_dir / "y_train.parquet", index=True)
    y_val.to_frame().to_parquet(  preproc_dir / "y_val.parquet",   index=True)
    y_test.to_frame().to_parquet( preproc_dir / "y_test.parquet",  index=True)

    # Save encoding map as pickle — needed for consistent transformation
    with open(preproc_dir / "encoding_map.pkl", "wb") as f:
        pickle.dump(encoding_map, f)

    if verbose:
        print(f"\n   Label-encoded splits saved (for LightGBM / XGBoost):")
        print(f"     X_train_lgbm : {X_train_lgbm.shape}")
        print(f"     X_val_lgbm   : {X_val_lgbm.shape}")
        print(f"     X_test_lgbm  : {X_test_lgbm.shape}  (frozen TEST)")
        print(f"     encoding_map : {len(encoding_map)} encoders")
        print(f"     All artifacts saved to: {preproc_dir}")

    return X_train_lgbm, X_val_lgbm, X_test_lgbm, encoding_map


# ── Step 3b: Load preprocessed (PREPROC_READY=True) ──────────────────────────

def load_preprocessed(preproc_dir, verbose=True):
    """
    Load previously saved preprocessed splits and encoding artifacts.

    Called instead of preprocess_and_save() when PREPROC_READY=True.
    Allows re-running the notebook without repeating expensive preprocessing.

    WHY load y_train/y_val/y_test here: when PREPROC_READY=True, load_enriched()
    is also skipped. The saved y files are the ground truth — they are aligned
    with the saved X files by construction (saved together in preprocess_and_save).

    Parameters
    ----------
    preproc_dir : Path — directory containing saved preprocessed artifacts
    verbose     : bool — print progress (default: True)

    Returns
    -------
    tuple (X_train_lgbm, X_val_lgbm, X_test_lgbm, encoding_map, y_train, y_val, y_test)
        X_train_lgbm : pd.DataFrame — label-encoded 60% train features
        X_val_lgbm   : pd.DataFrame — label-encoded 20% val features
        X_test_lgbm  : pd.DataFrame — label-encoded 20% frozen TEST features
        encoding_map : dict         — fitted label encoders
        y_train      : pd.Series    — 60% train labels
        y_val        : pd.Series    — 20% val labels
        y_test       : pd.Series    — 20% frozen TEST labels
    """
    if verbose:
        print("=" * 60)
        print("STEP 2 — Load preprocessed (PREPROC_READY=True)")
        print("=" * 60)

    X_train_lgbm = pd.read_parquet(preproc_dir / "X_train_lgbm.parquet")
    X_val_lgbm   = pd.read_parquet(preproc_dir / "X_val_lgbm.parquet")
    X_test_lgbm  = pd.read_parquet(preproc_dir / "X_test_lgbm.parquet")
    y_train      = pd.read_parquet(preproc_dir / "y_train.parquet")["isFraud"]
    y_val        = pd.read_parquet(preproc_dir / "y_val.parquet")["isFraud"]
    y_test       = pd.read_parquet(preproc_dir / "y_test.parquet")["isFraud"]

    with open(preproc_dir / "encoding_map.pkl", "rb") as f:
        encoding_map = pickle.load(f)

    if verbose:
        print(f"   X_train_lgbm : {X_train_lgbm.shape}  | fraud rate: {y_train.mean():.4%}")
        print(f"   X_val_lgbm   : {X_val_lgbm.shape}    | fraud rate: {y_val.mean():.4%}")
        print(f"   X_test_lgbm  : {X_test_lgbm.shape}   | fraud rate: {y_test.mean():.4%}  (frozen TEST)")
        print(f"   encoding_map : {len(encoding_map)} encoders")

    return X_train_lgbm, X_val_lgbm, X_test_lgbm, encoding_map, y_train, y_val, y_test



# ── Step 3: CatBoost preprocessing ───────────────────────────────────────────

def preprocess_catboost_and_save(train_enriched, val_enriched, test_enriched,
                                  y_train, y_val, y_test,
                                  preproc_dir,
                                  cols_to_drop=None,
                                  verbose=True):
    """
    Prepare and save CatBoost-specific preprocessed splits.

    WHY a separate step from preprocess_and_save(): CatBoost requires raw string
    values in categorical columns — label encoding (done for LightGBM/XGBoost)
    discards the native categorical handling advantage of CatBoost.
    This step runs after preprocess_and_save() and saves additional artifacts.

    Parameters
    ----------
    train_enriched : pd.DataFrame — 60% train features (from load_enriched)
    val_enriched   : pd.DataFrame — 20% val features
    test_enriched  : pd.DataFrame — 20% frozen TEST features
    y_train        : pd.Series    — 60% train labels
    y_val          : pd.Series    — 20% val labels
    y_test         : pd.Series    — 20% frozen TEST labels
    preproc_dir    : Path         — directory to save CatBoost artifacts
                     WHY same preproc_dir as LightGBM/XGBoost: all preprocessed
                     artifacts live together — single source of truth for nb04
    cols_to_drop   : list[str]    — non-feature columns to remove
    verbose        : bool         — print progress (default: True)

    Returns
    -------
    tuple (X_train_cat, X_val_cat, X_test_cat, cat_features)

    Also saves to preproc_dir:
        X_train_cat.parquet, X_val_cat.parquet, X_test_cat.parquet
        cat_features.pkl
    """
    cols_to_drop = cols_to_drop or NON_FEATURE_COLS

    if verbose:
        print("=" * 60)
        print("STEP 3 — CatBoost preprocessing (train / val / frozen TEST)")
        print("=" * 60)

    return preprocess_catboost(
        train_enriched, val_enriched, test_enriched,
        y_train, y_val, y_test,
        preproc_dir=preproc_dir,
        cols_to_drop=cols_to_drop,
        verbose=verbose,
    )


def load_catboost_preprocessed(preproc_dir, verbose=True):
    """
    Load previously saved CatBoost preprocessed splits.

    Called in nb03/nb04 when CAT_PREPROC_READY=True.

    Parameters
    ----------
    preproc_dir : Path — directory containing saved CatBoost artifacts
    verbose     : bool — print progress (default: True)

    Returns
    -------
    tuple (X_train_cat, X_val_cat, X_test_cat, cat_features)
    """
    return load_catboost_splits(preproc_dir, verbose=verbose)


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


# ── Step 6: Preprocessing summary ────────────────────────────────────────────
 
def print_preprocessing_summary(
    X_train_lgbm, X_val_lgbm, X_test_lgbm,
    y_train, y_val, y_test,
    encoding_map,
    preproc_dir,
    outputs_dir,
    verbose=True,
):
    """
    Print a full preprocessing summary and validate all artifacts.
 
    Covers:
        - Split shapes and fraud rates
        - Dtype distribution in X_train_lgbm
        - NaN checks with assert (fail fast if preprocessing is broken)
        - Existence check for all expected saved files
 
    WHY assert instead of warning: NaN in model input silently degrades
    performance. A hard assert here forces the user to fix the issue
    before wasting time on training.
 
    WHY check both params JSONs: best_params_xgb.json may be missing if
    Optuna has not yet run — shown as '✗ MISSING' without crashing,
    so the summary can be called before XGBoost tuning completes.
 
    Parameters
    ----------
    X_train_lgbm : pd.DataFrame — label-encoded train features
    X_val_lgbm   : pd.DataFrame — label-encoded val features
    X_test_lgbm  : pd.DataFrame — label-encoded frozen TEST features
    y_train      : pd.Series    — train labels
    y_val        : pd.Series    — val labels
    y_test       : pd.Series    — frozen TEST labels
    encoding_map : dict         — fitted label encoders
    preproc_dir  : Path         — directory containing saved preprocessed artifacts
    outputs_dir  : Path         — directory containing best_params JSON files
    verbose      : bool         — print progress (default: True)
    """
    if not verbose:
        return
 
    print("=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
 
    # ── Shapes and fraud rates ────────────────────────────────────────────────
    print(f"  X_train_lgbm : {X_train_lgbm.shape}  | fraud rate: {y_train.mean():.4%}")
    print(f"  X_val_lgbm   : {X_val_lgbm.shape}    | fraud rate: {y_val.mean():.4%}")
    print(f"  X_test_lgbm  : {X_test_lgbm.shape}   | fraud rate: {y_test.mean():.4%}  (frozen TEST)")
    print(f"  encoding_map : {len(encoding_map)} label encoders fitted on train")
 
    # ── Dtype distribution ────────────────────────────────────────────────────
    print()
    print("  Dtypes in X_train_lgbm:")
    for dtype, count in X_train_lgbm.dtypes.value_counts().items():
        print(f"    {str(dtype):<12}: {count} columns")
 
    # ── NaN checks ────────────────────────────────────────────────────────────
    # WHY assert: NaN in model input silently degrades performance.
    # Hard assert forces fix before wasting time on training.
    print()
    nan_train = X_train_lgbm.isna().sum().sum()
    nan_val   = X_val_lgbm.isna().sum().sum()
    nan_test  = X_test_lgbm.isna().sum().sum()
    print(f"  NaN in X_train_lgbm : {nan_train} (expected: 0)")
    print(f"  NaN in X_val_lgbm   : {nan_val}   (expected: 0)")
    print(f"  NaN in X_test_lgbm  : {nan_test}  (expected: 0)")
 
    assert nan_train == 0, "NaN found in X_train_lgbm after preprocessing!"
    assert nan_val   == 0, "NaN found in X_val_lgbm after preprocessing!"
    assert nan_test  == 0, "NaN found in X_test_lgbm after preprocessing!"
    print("  NaN check: OK ✓")
 
    # ── Saved files check ─────────────────────────────────────────────────────
    # WHY show ✗ MISSING without crashing for params JSONs: best_params_xgb.json
    # may not exist yet if Optuna has not run — this is expected, not an error.
    print()
    saved_files = [
        preproc_dir  / "X_train_lgbm.parquet",
        preproc_dir  / "X_val_lgbm.parquet",
        preproc_dir  / "X_test_lgbm.parquet",
        preproc_dir  / "y_train.parquet",
        preproc_dir  / "y_val.parquet",
        preproc_dir  / "y_test.parquet",
        preproc_dir  / "encoding_map.pkl",
        outputs_dir  / "best_params_lgbm.json",
        outputs_dir  / "best_params_xgb.json",
    ]
    print("  Saved files:")
    for path in saved_files:
        status = "✓" if path.exists() else "✗ MISSING"
        print(f"    {status}  {path}")
 
    print("=" * 60)