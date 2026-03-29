"""
split_v0.py — Time-based train/val/test split loader
══════════════════════════════════════════════════════
Loads pre-split parquet files produced by data_loader.save_processed().

WHY replaced original time_split() logic:
    The original function used train_ratio=0.80 (80/20) with no frozen TEST.
    This produced an optimistic val estimate — no unbiased final metric existed.

    The correct evaluation requires a 3-way temporal split (60/20/20):
        train : days 1–101   (354,324 rows) — model training only
        val   : days 101–141 (118,108 rows) — early stopping only
        test  : days 141–183 (118,108 rows) — frozen TEST, final unbiased metric

    These files are already produced by data_loader.save_processed() and
    stored in data/. time_split() now simply loads them — keeping the same
    function name so baseline_v0.ipynb requires no changes.

Functions:
    time_split() — load pre-split train/val/test parquet files
"""

import pandas as pd
from pathlib import Path


# Target column — extracted from splits after loading.
# WHY named constant: avoids hardcoding 'isFraud' in multiple places.
TARGET_COL = "isFraud"


def time_split(data_path, time_col="TransactionDT", verbose=True):
    """
    Load pre-split train / val / test parquet files.

    WHY load instead of split:
        data_loader.save_processed() already performed the correct 60/20/20
        temporal split. Loading saved files guarantees baseline uses exactly
        the same split boundaries as V2 models — results are directly comparable.

    WHY 60/20/20 instead of original 80/20:
        The old 80/20 split had no frozen TEST set — val was used as both
        early stopping target and final metric, producing an optimistic estimate.
        With 60/20/20: val = early stopping only; TEST = final unbiased metric
        touched exactly once. This is the correct evaluation protocol.

    Parameters
    ----------
    data_path : str or Path — folder containing train.parquet, val.parquet,
                              test.parquet (output of data_loader.save_processed)
                              WHY replaces original (df, y) params: the notebook
                              passes the data path — splits are loaded here
    time_col  : str         — timestamp column (default: 'TransactionDT')
                              kept for API compatibility and diagnostic logging
    verbose   : bool        — print split diagnostics (default: True)

    Returns
    -------
    tuple (X_train, X_val, X_test, y_train, y_val, y_test)
        X_train : pd.DataFrame — 60% train features   (days 1–101)
        X_val   : pd.DataFrame — 20% val features     (days 101–141)
        X_test  : pd.DataFrame — 20% frozen TEST features (days 141–183)
        y_train : pd.Series    — train target
        y_val   : pd.Series    — val target
        y_test  : pd.Series    — frozen TEST target (touched once at final eval)
    """
    data_path = Path(data_path)

    if verbose:
        print(">> Loading pre-split train / val / test parquet files...")
        print(f"   Path: {data_path}")

    train = pd.read_parquet(data_path / "train.parquet")
    val   = pd.read_parquet(data_path / "val.parquet")
    test  = pd.read_parquet(data_path / "test.parquet")

    # Extract targets
    y_train = train[TARGET_COL].copy()
    y_val   = val[TARGET_COL].copy()
    y_test  = test[TARGET_COL].copy()

    # Drop target from features
    X_train = train.drop(columns=[TARGET_COL])
    X_val   = val.drop(columns=[TARGET_COL])
    X_test  = test.drop(columns=[TARGET_COL])

    if verbose:
        print(f"\n   Train : {len(X_train):,} rows | "
              f"days {int(train[time_col].min()//86400)}–{int(train[time_col].max()//86400)} | "
              f"fraud rate: {y_train.mean():.4%}")
        print(f"   Val   : {len(X_val):,} rows | "
              f"days {int(val[time_col].min()//86400)}–{int(val[time_col].max()//86400)} | "
              f"fraud rate: {y_val.mean():.4%}  ← early stopping")
        print(f"   Test  : {len(X_test):,} rows | "
              f"days {int(test[time_col].min()//86400)}–{int(test[time_col].max()//86400)} | "
              f"fraud rate: {y_test.mean():.4%}  ← frozen TEST")

        # Temporal ordering check
        assert train[time_col].max() <= val[time_col].min(), \
            "Temporal order broken: train/val!"
        assert val[time_col].max() <= test[time_col].min(), \
            "Temporal order broken: val/test!"
        print("   Temporal ordering : OK ✓")

    return X_train, X_val, X_test, y_train, y_val, y_test
