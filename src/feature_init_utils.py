"""
feature_init_utils.py — Shared initial feature creation
════════════════════════════════════════════════════════
Features discovered during EDA, reusable across all versions.
Each function is independent — call them separately when needed.

Functions:
    add_time_features()    — tx_day, tx_dow, tx_hour, tx_dom
    add_device_features()  — DeviceType_filled
"""

import pandas as pd
import numpy as np


def add_time_features(df, time_col="TransactionDT", verbose=True):
    """
    Create time-derived features from TransactionDT.
    Found during EDA: tx_hour is a strong fraud signal (peaks at hours 5-9).

    Parameters
    ----------
    df : pd.DataFrame — must contain time_col
    time_col : str — column with seconds from reference datetime
    verbose : bool — print progress

    Returns
    -------
    pd.DataFrame with new columns: tx_day, tx_dow, tx_hour, tx_dom
    """
    if verbose:
        print(f">> Adding time features from {time_col}...")
        print(f"   Shape before: {df.shape}")

    # Absolute day number
    df["tx_day"] = df[time_col] // 86400

    # Day of week (0-6), approximate — actual day names unknown (anonymized)
    df["tx_dow"] = (df["tx_day"] % 7).astype(int)

    # Hour of day (0-23) — strong fraud signal from EDA
    df["tx_hour"] = (df[time_col] // 3600) % 24

    # Day of month (1-30), approximate using 30-day cycle
    df["tx_dom"] = (df["tx_day"] % 30).astype(int) + 1

    if verbose:
        print(f"   Added: tx_day, tx_dow, tx_hour, tx_dom")
        print(f"   Shape after: {df.shape}")

    return df


def add_device_features(df, verbose=True):
    """
    Fill missing DeviceType with readable label.
    Found during EDA: presence of device info is a strong fraud signal.

    Parameters
    ----------
    df : pd.DataFrame — must contain DeviceType column
    verbose : bool — print progress

    Returns
    -------
    pd.DataFrame with new column: DeviceType_filled
    """
    if verbose:
        print(f">> Adding device features...")
        print(f"   Shape before: {df.shape}")

    # Replace NaN with readable label
    df["DeviceType_filled"] = df["DeviceType"].fillna("No device info")

    if verbose:
        print(f"   Added: DeviceType_filled")
        print(f"   Shape after: {df.shape}")

    return df