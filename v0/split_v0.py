"""
split_v0.py — Time-based train/validation split for baseline v0
════════════════════════════════════════════════════════════════
Splits data chronologically — model trains on past, validates on future.
No random shuffle — mimics real-world deployment.

Functions:
    time_split() — split by TransactionDT quantile
"""

import numpy as np
import pandas as pd


def time_split(df, y, time_col="TransactionDT", train_ratio=0.80, verbose=True):
    """
    Split data into train and validation sets based on time.
    First train_ratio% of data (by time) → train, rest → validation.

    Parameters
    ----------
    df : pd.DataFrame — feature matrix (must contain time_col or be aligned with y)
    y : pd.Series — target variable
    time_col : str — column with temporal ordering
    train_ratio : float — fraction of data for training (default=0.80)
    verbose : bool — print split details

    Returns
    -------
    tuple (X_train, X_val, y_train, y_val) — split DataFrames/Series
    """
    if verbose:
        print(">> Time-based split...")
        print(f"   Total samples: {len(df):,}")
        print(f"   Train ratio: {train_ratio:.0%}")

    # Find time boundary
    if time_col in df.columns:
        time_values = df[time_col]
    else:
        raise KeyError(f"Column '{time_col}' not found in DataFrame. "
                       f"Available columns: {df.columns.tolist()[:10]}...")

    split_point = time_values.quantile(train_ratio)

    # Create masks
    train_mask = time_values <= split_point
    val_mask = time_values > split_point

    # Split features and target
    X_train = df[train_mask]
    X_val = df[val_mask]
    y_train = y[train_mask]
    y_val = y[val_mask]

    if verbose:
        # Time boundaries
        print(f"\n   Time boundary: {split_point:,.0f}")
        print(f"   Train period: {time_values[train_mask].min():,.0f} — {time_values[train_mask].max():,.0f}")
        print(f"   Val period:   {time_values[val_mask].min():,.0f} — {time_values[val_mask].max():,.0f}")

        # Sizes
        print(f"\n   Train size: {len(X_train):,} ({len(X_train)/len(df):.1%})")
        print(f"   Val size:   {len(X_val):,} ({len(X_val)/len(df):.1%})")

        # Fraud rates
        train_fraud = y_train.mean()
        val_fraud = y_val.mean()
        print(f"\n   Fraud rate — train: {train_fraud:.4f} ({train_fraud:.2%})")
        print(f"   Fraud rate — val:   {val_fraud:.4f} ({val_fraud:.2%})")

    return X_train, X_val, y_train, y_val