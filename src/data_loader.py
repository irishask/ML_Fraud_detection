"""
data_loader.py — Shared data loading for all versions
══════════════════════════════════════════════════════
Functions:
    load_raw_data()    — load 2 train CSVs, merge transaction + identity
    reduce_memory()    — downcast numeric dtypes to save RAM
    save_processed()   — split orig_full_train into train/val/test and save
    load_processed()   — load ready parquet files (for all versions)
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_raw_data(raw_path, top=7, verbose=True):
    """
    Load 2 raw CSV files and merge transaction + identity tables.
    Only train data is loaded — Kaggle test CSVs are not part of this project.

    Parameters
    ----------
    raw_path : str or Path — folder with raw CSV files
    top      : int         — number of rows to preview (default=7)
    verbose  : bool        — print progress and diagnostics

    Returns
    -------
    pd.DataFrame — merged train DataFrame (590,540 rows, isFraud known)
    """
    raw_path = Path(raw_path)

    if verbose:
        print(">> Loading raw CSV files...")

    train_tx = pd.read_csv(raw_path / "train_transaction.csv")
    train_id = pd.read_csv(raw_path / "train_identity.csv")

    if verbose:
        print(f"  train_transaction: {train_tx.shape}")
        print(f"  train_identity:    {train_id.shape}")
        print("\n>> Merging transaction + identity (left join)...")

    train = train_tx.merge(train_id, on="TransactionID", how="left")
    del train_tx, train_id

    if verbose:
        print(f"  Train after merge: {train.shape}")
        print(f"\n  Train preview (top {top} rows):")
        display(train.head(top))

    return train


def reduce_memory(df, name="DataFrame", verbose=True):
    """
    Downcast numeric columns to smallest possible dtype.
    No impact on model performance — tree models split on value > threshold,
    precision beyond float32 (7 decimal places) is irrelevant.

    Parameters
    ----------
    df      : pd.DataFrame
    name    : str  — name for logging
    verbose : bool — print before/after memory usage

    Returns
    -------
    pd.DataFrame with downcasted dtypes
    """
    if verbose:
        print(f"\n>> Reducing memory for {name}...")
        print(f"   Shape: {df.shape}")

    start_mem = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and col_type.name != "category":
            c_min, c_max = df[col].min(), df[col].max()

            if str(col_type).startswith("int"):
                for dtype in [np.int8, np.int16, np.int32, np.int64]:
                    if c_min >= np.iinfo(dtype).min and c_max <= np.iinfo(dtype).max:
                        df[col] = df[col].astype(dtype)
                        break
            elif str(col_type).startswith("float"):
                for dtype in [np.float32, np.float64]:
                    if c_min >= np.finfo(dtype).min and c_max <= np.finfo(dtype).max:
                        df[col] = df[col].astype(dtype)
                        break

    end_mem = df.memory_usage(deep=True).sum() / 1024**2

    if verbose:
        pct = 100 * (start_mem - end_mem) / start_mem
        print(f"   Memory: {start_mem:.0f} MB → {end_mem:.0f} MB (↓ {pct:.1f}%)")
        print(f"   Dtypes: {dict(df.dtypes.value_counts())}")

    return df


def save_processed(orig_full_train, save_path,
                   train_ratio=0.60, val_ratio=0.20,
                   time_col="TransactionDT",
                   verbose=True):
    """
    Save orig_full_train as parquet and split into train / val / test by time.

    Split strategy (time-based, sort + index — no quantile):
        train : first 60% by TransactionDT  ← model training
        val   : next  20%                   ← early stopping + Optuna
        test  : last  20%                   ← frozen TEST, final evaluation

    WHY 60/20/20:
        - test (last 20%) is the frozen TEST set — never used during training.
        - val  (middle 20%) is used for early stopping and Optuna — model sees
          it indirectly, so it cannot be an unbiased test set.
        - train (first 60%) is used for fitting the model.
        - All three sets have isFraud known — no Kaggle submission needed.

    WHY sort + index instead of quantile:
        quantile(ratio) finds a time VALUE — duplicate TransactionDT at the
        boundary causes unpredictable actual ratios. Index-based split always
        produces exactly the specified fraction of rows.

    Parameters
    ----------
    orig_full_train : pd.DataFrame — full merged + memory-reduced train DataFrame
    save_path       : str or Path  — folder to save parquet files
    train_ratio     : float        — fraction for training set (default: 0.60)
    val_ratio       : float        — fraction for validation set (default: 0.20)
                      test_ratio is implied: 1 - train_ratio - val_ratio = 0.20
    time_col        : str          — column for chronological ordering
                      (default: 'TransactionDT')
    verbose         : bool         — print split stats (default: True)
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    test_ratio = round(1.0 - train_ratio - val_ratio, 10)
    assert test_ratio > 0, "train_ratio + val_ratio must be < 1.0"

    if verbose:
        print(">> Splitting orig_full_train (60/20/20 by time)...")
        print(f"   Total rows : {len(orig_full_train):,}")
        print(f"   train_ratio: {train_ratio:.0%} | val_ratio: {val_ratio:.0%} | test_ratio: {test_ratio:.0%}")

    # Sort by time — guarantees chronological order before index split
    df_sorted = orig_full_train.sort_values(time_col).reset_index(drop=True)

    # Exact index split — no boundary ambiguity
    n = len(df_sorted)
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))

    train = df_sorted.iloc[:train_end]
    val   = df_sorted.iloc[train_end:val_end]
    test  = df_sorted.iloc[val_end:]

    if verbose:
        print(f"\n   train : {len(train):,} rows | fraud: {train['isFraud'].mean():.4%}")
        print(f"   val   : {len(val):,} rows | fraud: {val['isFraud'].mean():.4%}  <- early stopping + Optuna")
        print(f"   test  : {len(test):,} rows | fraud: {test['isFraud'].mean():.4%}  <- frozen TEST")

        # Temporal ordering check
        assert train[time_col].max() <= val[time_col].min(), "Temporal order broken: train/val!"
        assert val[time_col].max()   <= test[time_col].min(), "Temporal order broken: val/test!"
        print("   Temporal ordering : OK ✓")

    # Save all 4 files
    if verbose:
        print("\n>> Saving parquet files...")

    orig_full_train.to_parquet(save_path / "orig_full_train.parquet", index=False)
    train.to_parquet(save_path / "train.parquet", index=False)
    val.to_parquet(  save_path / "val.parquet",   index=False)
    test.to_parquet( save_path / "test.parquet",  index=False)

    if verbose:
        for fname in ["orig_full_train.parquet", "train.parquet", "val.parquet", "test.parquet"]:
            size = (save_path / fname).stat().st_size / 1024**2
            print(f"   {fname}: {size:.0f} MB")
        print("\n   Done ✓")


def load_processed(data_path, top=7, verbose=True):
    """
    Load pre-processed parquet files. Fast — preserves dtypes from reduce_memory.

    Parameters
    ----------
    data_path : str or Path — folder containing train.parquet, val.parquet, test.parquet
    top       : int         — number of rows to preview (default=7)
    verbose   : bool        — print shapes and memory

    Returns
    -------
    tuple (train, val, test) — DataFrames ready for feature engineering
        train : pd.DataFrame — 60% of orig_full_train (isFraud known)
        val   : pd.DataFrame — 20% of orig_full_train (isFraud known) <- early stopping
        test  : pd.DataFrame — 20% of orig_full_train (isFraud known) <- frozen TEST
    """
    data_path = Path(data_path)

    if verbose:
        print(">> Loading processed parquet files (train + val + test)...")

    train = pd.read_parquet(data_path / "train.parquet")
    val   = pd.read_parquet(data_path / "val.parquet")
    test  = pd.read_parquet(data_path / "test.parquet")

    if verbose:
        train_mem = train.memory_usage(deep=True).sum() / 1024**2
        val_mem   = val.memory_usage(deep=True).sum()   / 1024**2
        test_mem  = test.memory_usage(deep=True).sum()  / 1024**2
        print(f"   train : {train.shape[0]:,} rows × {train.shape[1]} cols ({train_mem:.0f} MB)")
        print(f"   val   : {val.shape[0]:,} rows × {val.shape[1]} cols ({val_mem:.0f} MB)  <- early stopping")
        print(f"   test  : {test.shape[0]:,} rows × {test.shape[1]} cols ({test_mem:.0f} MB)  <- frozen TEST")
        print(f"   Dtypes: {dict(train.dtypes.value_counts())}")
        print(f"\n   Train preview (top {top} rows):")
        display(train.head(top))

    return train, val, test