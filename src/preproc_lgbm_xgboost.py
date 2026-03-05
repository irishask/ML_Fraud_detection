"""
preproc_lgbm_xgboost.py — Preprocessing for LightGBM and XGBoost
═════════════════════════════════════════════════════════════════
Applies encoding, fill, and drop steps after compute_user_aggregations()
has already been run on the full dataset before time_split().

Used by: LightGBM, XGBoost
NOT used by: CatBoost (see preproc_catboost.py — CatBoost handles
             categorical encoding internally via ordered target statistics)

Pipeline (this module covers steps 2–4 only):
    Step 1 [preproc_agg.py]  : compute_user_aggregations(full_train)
    Step 2 [this module, fit] : encode_categoricals_fit  — learn label encoding from train only
    Step 3 [this module]      : fill_missing             — fill NaN with fill_value
    Step 4 [this module]      : drop_non_features        — remove non-model columns

Functions:
    preprocess_fit()       — fit on train slice, return (df, encoding_map)
    preprocess_transform() — apply encoding_map to val/test slice, return df
"""

import sys
import os

# Ensure v0 modules are importable regardless of how this module is loaded.
# WHY: this module lives in v1/ and reuses utility functions from v0/.
# sys.path fix here guarantees the import works even after kernel restart.
_V0_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v0")
if _V0_PATH not in sys.path:
    sys.path.append(_V0_PATH)

from preproc_v0 import (
    encode_categoricals_fit,
    encode_categoricals_transform,
    fill_missing,
    drop_non_features,
)


# ── Preprocess: Fit ───────────────────────────────────────────────────────────

def preprocess_fit(df, cols_to_drop,
                   fill_value=-1,
                   verbose=True):
    """
    Fit encoding on the training slice and return learned state.

    IMPORTANT: compute_user_aggregations() (preproc_agg.py) must be called
    on the FULL dataset BEFORE time_split() and BEFORE this function.
    Aggregation features must already be present in df when this is called.

    Pipeline:
        1. encode_categoricals_fit — learn label encoding from train only
                                     (fitting on val/test would be leakage)
        2. fill_missing            — fill all remaining NaN with fill_value
        3. drop_non_features       — remove non-model columns (target, ID, time)

    Parameters
    ----------
    df           : pd.DataFrame — training slice AFTER time_split(),
                   already containing aggregation features
    cols_to_drop : list[str]    — non-feature columns to remove
                   WHY passed as parameter: defined once in config.py (NON_FEATURE_COLS)
                   and shared across all preprocessing modules — no duplication
    fill_value   : numeric      — NaN replacement value (default: -1)
                   WHY -1: LightGBM and XGBoost treat -1 as a valid numeric value
                   and learn to split on it; consistent with preproc_v0 baseline
    verbose      : bool         — print progress (default: True)

    Returns
    -------
    tuple (df, encoding_map)
        df           — processed training DataFrame ready for model input
        encoding_map — dict mapping column names to learned label encoders;
                       must be passed to preprocess_transform() for val/test
    """
    if verbose:
        print("=" * 60)
        print("PREPROCESSING — FIT (LightGBM / XGBoost)")
        print("=" * 60)

    # Step 1: label encode categoricals — fit on train only
    df, encoding_map = encode_categoricals_fit(df, verbose=verbose)

    # Step 2: fill remaining NaN
    df = fill_missing(df, fill_value=fill_value, verbose=verbose)

    # Step 3: drop non-feature columns
    df = drop_non_features(df, cols_to_drop=cols_to_drop, verbose=verbose)

    if verbose:
        print(f"\n   Final shape: {df.shape}")
        print(f"   Dtypes: {dict(df.dtypes.value_counts())}")

    return df, encoding_map


# ── Preprocess: Transform ─────────────────────────────────────────────────────

def preprocess_transform(df, encoding_map, cols_to_drop,
                          fill_value=-1,
                          verbose=True):
    """
    Apply fitted preprocessing to validation or test data.
    Uses encoding_map learned from train — no data leakage.

    IMPORTANT: same requirement as preprocess_fit() — aggregation features
    must already be present in df (computed before time_split() on full train).

    Pipeline:
        1. encode_categoricals_transform — apply learned label encoding
                                           (transform only — never refit on val/test)
        2. fill_missing                  — fill all remaining NaN with fill_value
        3. drop_non_features             — remove non-model columns

    Parameters
    ----------
    df           : pd.DataFrame — val/test slice AFTER time_split(),
                   already containing aggregation features
    encoding_map : dict         — learned encoders from preprocess_fit()
    cols_to_drop : list[str]    — non-feature columns to remove
                   WHY passed as parameter: same cols_to_drop used in fit —
                   must be consistent to avoid shape mismatch at model input
    fill_value   : numeric      — NaN replacement value (default: -1)
                   WHY -1: must match fill_value used in preprocess_fit()
                   so train and val/test have identical feature semantics
    verbose      : bool         — print progress (default: True)

    Returns
    -------
    pd.DataFrame — processed DataFrame ready for model input
    """
    if verbose:
        print("=" * 60)
        print("PREPROCESSING — TRANSFORM (LightGBM / XGBoost)")
        print("=" * 60)

    # Step 1: apply learned label encoding — transform only, never refit
    df = encode_categoricals_transform(df, encoding_map, verbose=verbose)

    # Step 2: fill remaining NaN
    df = fill_missing(df, fill_value=fill_value, verbose=verbose)

    # Step 3: drop non-feature columns
    df = drop_non_features(df, cols_to_drop=cols_to_drop, verbose=verbose)

    if verbose:
        print(f"\n   Final shape: {df.shape}")
        print(f"   Dtypes: {dict(df.dtypes.value_counts())}")

    return df