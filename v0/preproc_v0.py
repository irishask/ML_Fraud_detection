"""
preproc_v0.py — Preprocessing for baseline v0
══════════════════════════════════════════════════
Simple preprocessing: encode categoricals, fill NaN, drop non-feature columns.
No feature engineering — raw features only.

KEY PRINCIPLE: fit on train, transform on val/test (no data leakage).

Functions:
    encode_categoricals_fit()       — learn encoding from train
    encode_categoricals_transform() — apply learned encoding to any set
    fill_missing()                  — fill NaN with a value
    drop_non_features()             — remove columns not used as model input
    preprocess_fit()                — fit all steps on train
    preprocess_transform()          — apply fitted steps to val/test
"""

import pandas as pd
import numpy as np


def encode_categoricals_fit(df, verbose=True):
    """
    Learn label encoding from training data.
    Returns encoding mapping — use with encode_categoricals_transform().

    Parameters
    ----------
    df : pd.DataFrame — training data only
    verbose : bool — print progress

    Returns
    -------
    tuple (df, encoding_map)
        df           — DataFrame with encoded columns
        encoding_map — dict {col_name: {value: code}} for applying to val/test
    """
    if verbose:
        print(">> Encoding categorical columns (fit on train)...")
        print(f"   Shape before: {df.shape}")

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if verbose:
        print(f"   Found {len(cat_cols)} categorical columns: "
              f"{cat_cols[:10]}{'...' if len(cat_cols) > 10 else ''}")

    # Learn mapping from train and apply
    encoding_map = {}
    for col in cat_cols:
        codes, uniques = pd.factorize(df[col])
        # Build mapping: value → code
        mapping = {val: idx for idx, val in enumerate(uniques)}
        encoding_map[col] = mapping
        df[col] = codes.astype(np.int32)

    if verbose:
        print(f"   Encoded {len(cat_cols)} columns")
        print(f"   Shape after: {df.shape}")

    return df, encoding_map


def encode_categoricals_transform(df, encoding_map, unseen_value=-2, verbose=True):
    """
    Apply learned encoding to validation/test data.
    Values not seen during training get unseen_value code.

    Parameters
    ----------
    df : pd.DataFrame — validation or test data
    encoding_map : dict — from encode_categoricals_fit()
    unseen_value : int — code for values not seen in training (default=-2)
    verbose : bool — print progress

    Returns
    -------
    pd.DataFrame with encoded columns
    """
    if verbose:
        print(">> Encoding categorical columns (transform)...")
        print(f"   Shape before: {df.shape}")

    unseen_count = 0

    for col, mapping in encoding_map.items():
        if col not in df.columns:
            continue

        # Map known values, replace unseen with unseen_value
        original_values = df[col]
        df[col] = original_values.map(mapping)

        # Count and fill unseen values (NaN after mapping = unseen)
        unseen_mask = df[col].isna() & original_values.notna()
        unseen_in_col = unseen_mask.sum()
        unseen_count += unseen_in_col

        df[col] = df[col].fillna(unseen_value).astype(np.int32)

    if verbose:
        print(f"   Transformed {len(encoding_map)} columns")
        print(f"   Unseen values (mapped to {unseen_value}): {unseen_count:,}")
        print(f"   Shape after: {df.shape}")

    return df


def fill_missing(df, fill_value=-1, verbose=True):
    """
    Fill all remaining NaN values with fill_value.
    Default -1 is compatible with LightGBM/XGBoost/CatBoost.

    Parameters
    ----------
    df : pd.DataFrame
    fill_value : numeric — value to replace NaN (default=-1)
    verbose : bool — print NaN count before/after

    Returns
    -------
    pd.DataFrame with no NaN values
    """
    if verbose:
        nan_before = df.isnull().sum().sum()
        nan_cols_before = df.isnull().any().sum()
        print(f"\n>> Filling missing values with {fill_value}...")
        print(f"   NaN before: {nan_before:,} values in {nan_cols_before} columns")

    df = df.fillna(fill_value)

    if verbose:
        nan_after = df.isnull().sum().sum()
        print(f"   NaN after:  {nan_after}")

    return df


def drop_non_features(df, cols_to_drop, verbose=True):
    """
    Remove columns that should not be used as model input features.
    Drops only columns that exist in the DataFrame (safe for test set
    which may not have target column).

    Parameters
    ----------
    df : pd.DataFrame
    cols_to_drop : list of str — column names to remove
    verbose : bool — print which columns were dropped

    Returns
    -------
    pd.DataFrame without specified columns
    """
    if verbose:
        print(f"\n>> Dropping non-feature columns...")
        print(f"   Shape before: {df.shape}")

    existing = [col for col in cols_to_drop if col in df.columns]
    missing = [col for col in cols_to_drop if col not in df.columns]

    df = df.drop(columns=existing)

    if verbose:
        print(f"   Dropped: {existing}")
        if missing:
            print(f"   Not found (skipped): {missing}")
        print(f"   Shape after: {df.shape}")

    return df


def preprocess_fit(df, cols_to_drop, fill_value=-1, verbose=True):
    """
    Fit preprocessing on training data: encode → fill NaN → drop non-features.
    Returns processed DataFrame and encoding_map for applying to val/test.

    Parameters
    ----------
    df : pd.DataFrame — training data only
    cols_to_drop : list of str — columns to remove (e.g. NON_FEATURE_COLS)
    fill_value : numeric — NaN replacement value (default=-1)
    verbose : bool — print progress

    Returns
    -------
    tuple (df, encoding_map)
        df           — processed training DataFrame
        encoding_map — dict to pass to preprocess_transform()
    """
    if verbose:
        print("=" * 60)
        print("PREPROCESSING v0 — FIT (train)")
        print("=" * 60)

    df, encoding_map = encode_categoricals_fit(df, verbose=verbose)
    df = fill_missing(df, fill_value=fill_value, verbose=verbose)
    df = drop_non_features(df, cols_to_drop=cols_to_drop, verbose=verbose)

    if verbose:
        print(f"\n   Final shape: {df.shape}")
        print(f"   Dtypes: {dict(df.dtypes.value_counts())}")

    return df, encoding_map


def preprocess_transform(df, encoding_map, cols_to_drop, fill_value=-1, verbose=True):
    """
    Apply fitted preprocessing to validation/test data.
    Uses encoding_map learned from training data — no data leakage.

    Parameters
    ----------
    df : pd.DataFrame — validation or test data
    encoding_map : dict — from preprocess_fit()
    cols_to_drop : list of str — columns to remove
    fill_value : numeric — NaN replacement value (default=-1)
    verbose : bool — print progress

    Returns
    -------
    pd.DataFrame — processed DataFrame
    """
    if verbose:
        print("=" * 60)
        print("PREPROCESSING v0 — TRANSFORM (val/test)")
        print("=" * 60)

    df = encode_categoricals_transform(df, encoding_map, verbose=verbose)
    df = fill_missing(df, fill_value=fill_value, verbose=verbose)
    df = drop_non_features(df, cols_to_drop=cols_to_drop, verbose=verbose)

    if verbose:
        print(f"\n   Final shape: {df.shape}")
        print(f"   Dtypes: {dict(df.dtypes.value_counts())}")

    return df