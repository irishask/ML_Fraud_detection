"""
data_loader.py — Shared data loading for all versions
══════════════════════════════════════════════════════
Functions:
    load_and_merge()   — load 4 CSVs, merge transaction + identity
    reduce_memory()    — downcast numeric dtypes to save RAM
    save_processed()   — save DataFrames as parquet (call manually)
    load_processed()   — load ready parquet files (for all versions)
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_and_merge(raw_path, top=7, verbose=True):
    """
    Load 4 raw CSV files and merge transaction + identity tables.

    Parameters
    ----------
    raw_path : str or Path — folder with raw CSV files
    top : int — number of rows to preview (default=7)
    verbose : bool — print progress and diagnostics

    Returns
    -------
    tuple (train, test) — merged DataFrames
    """
    raw_path = Path(raw_path)

    if verbose:
        print(">> Loading raw CSV files...")

    train_tx = pd.read_csv(raw_path / "train_transaction.csv")
    train_id = pd.read_csv(raw_path / "train_identity.csv")
    test_tx = pd.read_csv(raw_path / "test_transaction.csv")
    test_id = pd.read_csv(raw_path / "test_identity.csv")

    if verbose:
        print(f"  train_transaction: {train_tx.shape}")
        print(f"  train_identity:    {train_id.shape}")
        print(f"  test_transaction:  {test_tx.shape}")
        print(f"  test_identity:     {test_id.shape}")

        missing_in_test = set(train_tx.columns) - set(test_tx.columns)
        print(f"  In train but not in test: {missing_in_test}")

    if verbose:
        print("\n>> Merging transaction + identity (left join)...")

    train = train_tx.merge(train_id, on="TransactionID", how="left")
    test = test_tx.merge(test_id, on="TransactionID", how="left")

    del train_tx, train_id, test_tx, test_id

    if verbose:
        print(f"  Train after merge: {train.shape}")
        print(f"  Test after merge:  {test.shape}")
        print(f"\n  Train preview (top {top} rows):")
        display(train.head(top))

    return train, test


def reduce_memory(df, name="DataFrame", verbose=True):
    """
    Downcast numeric columns to smallest possible dtype.
    No impact on model performance — tree models split on value > threshold,
    precision beyond float32 (7 decimal places) is irrelevant.

    Parameters
    ----------
    df : pd.DataFrame
    name : str — name for logging
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


def save_processed(train, test, save_path, verbose=True):
    """
    Save DataFrames as parquet files. Call manually when data is ready.

    Parameters
    ----------
    train : pd.DataFrame
    test : pd.DataFrame
    save_path : str or Path — folder to save parquet files
    verbose : bool — print file sizes
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(">> Saving as parquet...")

    train.to_parquet(save_path / "train.parquet", index=False)
    test.to_parquet(save_path / "test.parquet", index=False)

    if verbose:
        train_size = (save_path / "train.parquet").stat().st_size / 1024**2
        test_size = (save_path / "test.parquet").stat().st_size / 1024**2
        print(f"   Saved to {save_path}:")
        print(f"     train.parquet: {train_size:.0f} MB ({train.shape[0]:,} rows × {train.shape[1]} cols)")
        print(f"     test.parquet:  {test_size:.0f} MB ({test.shape[0]:,} rows × {test.shape[1]} cols)")


def load_processed(data_path, top=7, verbose=True):
    """
    Load pre-processed parquet files. Fast — preserves dtypes from reduce_memory.

    Parameters
    ----------
    data_path : str or Path — folder containing train.parquet and test.parquet
    top : int — number of rows to preview (default=7)
    verbose : bool — print shapes and memory

    Returns
    -------
    tuple (train, test) — DataFrames ready for preprocessing
    """
    data_path = Path(data_path)

    if verbose:
        print(">> Loading processed parquet files...")

    train = pd.read_parquet(data_path / "train.parquet")
    test = pd.read_parquet(data_path / "test.parquet")

    if verbose:
        train_mem = train.memory_usage(deep=True).sum() / 1024**2
        test_mem = test.memory_usage(deep=True).sum() / 1024**2
        print(f"   Train: {train.shape[0]:,} rows × {train.shape[1]} cols ({train_mem:.0f} MB)")
        print(f"   Test:  {test.shape[0]:,} rows × {test.shape[1]} cols ({test_mem:.0f} MB)")
        print(f"   Dtypes: {dict(train.dtypes.value_counts())}")
        print(f"\n   Train preview (top {top} rows):")
        display(train.head(top))

    return train, test