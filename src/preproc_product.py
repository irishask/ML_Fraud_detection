"""
preproc_product.py — Product Profile Features
══════════════════════════════════════════════
Computes 2 features that capture product-level transaction patterns.

MODEL-AGNOSTIC — used by all models (LightGBM, XGBoost, CatBoost).

Must be called on the FULL dataset (train + test concatenated) BEFORE
time_split(), so that test transactions see their complete prior history.

No-leakage guarantee:
    is_new_product uses groupby.cumcount() on uid + ProductCD pairs.
    amt_vs_product_median uses groupby(ProductCD).expanding().shift(1).
    Both use only strictly prior transactions. isFraud is never used.

New features (2 total):
    is_new_product       — 1 if user is buying this ProductCD for the first time
    amt_vs_product_median — current amount / historical median for this ProductCD

Public functions:
    compute_product_features() — compute both features on a DataFrame
"""

import numpy as np
import pandas as pd


# ── Configuration ─────────────────────────────────────────────────────────────

# User proxy: card1 + addr1.
# WHY same as preproc_agg.py and preproc_behavioral.py: consistent UID
# definition across all feature modules — no fragmentation of user identity.
USER_GROUP_COLS = ["card1", "addr1"]

# Product type column.
# WHY ProductCD: the only product-type categorical in the dataset.
PRODUCT_COL = "ProductCD"

# Timestamp column — required to sort before cumcount.
TIME_COL = "TransactionDT"

# Fill value for rows where uid or ProductCD is NaN.
# WHY 1: if we cannot identify the user or product, we treat the transaction
# as a first encounter — the conservative (more suspicious) assumption.
UNKNOWN_FILL = 1

# Fill value for amt_vs_product_median when no prior history exists.
# WHY -1: consistent with all other aggregation features — signals
# "no prior history" to tree models without distorting the distribution.
MISSING_FILL = -1


# ── Public API ────────────────────────────────────────────────────────────────

def compute_product_features(df,
                              group_cols=None,
                              product_col=PRODUCT_COL,
                              time_col=TIME_COL,
                              unknown_fill=UNKNOWN_FILL,
                              missing_fill=MISSING_FILL,
                              verbose=True):
    """
    Compute product profile features on a DataFrame.

    MODEL-AGNOSTIC: output is used by LightGBM and XGBoost.

    IMPORTANT: must be called on the FULL dataset (train + test concatenated)
    BEFORE time_split(), so that test transactions see train history.
    If called after compute_user_aggregations(), df is already sorted by
    time_col — the monotonicity check below will confirm this.

    No leakage:
        is_new_product: cumcount() on uid + ProductCD counts prior occurrences
            only. cumcount=0 means first occurrence. isFraud never accessed.
        amt_vs_product_median: groupby(ProductCD).expanding().shift(1) uses
            only strictly prior transactions for each product type.

    Parameters
    ----------
    df           : pd.DataFrame — full dataset sorted by time_col
    group_cols   : list[str]    — user proxy columns
                   WHY default USER_GROUP_COLS: card1+addr1 is the confirmed
                   UID from EDA — consistent with all other feature modules
    product_col  : str          — product type column (default: 'ProductCD')
                   WHY named param: allows reuse if product column is renamed
    time_col     : str          — timestamp column used for sort verification
    unknown_fill : int          — fill for NaN uid or product in is_new_product
                   WHY 1: unknown user or product → treat as first encounter
                   → conservative assumption (suspicious by default)
    missing_fill : numeric      — fill for amt_vs_product_median when no prior
                   history exists (default: -1)
                   WHY -1: consistent with all other aggregation features —
                   signals "no prior history" without distorting distribution
    verbose      : bool         — print progress (default: True)

    Returns
    -------
    tuple (df, new_feature_cols)
        df               — input DataFrame with new columns appended
        new_feature_cols — ['is_new_product', 'amt_vs_product_median']
    """
    group_cols = group_cols or USER_GROUP_COLS

    if verbose:
        print("=" * 60)
        print("FEATURE ENGINEERING — Product Profile")
        print("=" * 60)
        print(f"   Group key    : {group_cols}")
        print(f"   Product col  : {product_col}")
        print(f"   Shape before : {df.shape}")

    # CRITICAL: df must be sorted by time_col — cumcount respects row order.
    # Each row's cumcount value is determined by how many prior rows in the
    # same group exist above it — sorting guarantees "prior" = "earlier in time".
    if not df[time_col].is_monotonic_increasing:
        if verbose:
            print(f"   WARNING: df not sorted by {time_col} — sorting now.")
        df = df.sort_values(time_col).reset_index(drop=True)

    # ── is_new_product ────────────────────────────────────────────────────────
    # Group by uid + ProductCD and count occurrences up to each row.
    # cumcount() starts at 0 for the first row in each group:
    #   cumcount = 0 → this is the first time this user buys this product → 1
    #   cumcount > 0 → user has bought this product before               → 0
    #
    # No leakage: we are labeling a property of the current transaction
    # ("is this the first time?") using only the count of prior rows.
    # isFraud is never accessed.
    if verbose:
        print("\n   Computing is_new_product ...")

    cumcount = df.groupby(group_cols + [product_col]).cumcount()
    df["is_new_product"] = (cumcount == 0).astype(np.int8)

    # Fill rows where uid or product is NaN — groupby silently drops NaN groups,
    # leaving those rows with NaN cumcount → fill with unknown_fill (1).
    nan_mask = df["is_new_product"].isna()
    if nan_mask.any():
        df.loc[nan_mask, "is_new_product"] = unknown_fill
        if verbose:
            print(f"   Filled {nan_mask.sum()} NaN rows (unknown uid/product) with {unknown_fill}")

    # ── amt_vs_product_median ─────────────────────────────────────────────────
    # Current amount / historical median amount for this ProductCD.
    # WHY: fraudsters may buy products at unusual price points vs the product's
    # typical transaction history — detects price anomalies per product type.
    # WHY expanding().shift(1): uses only strictly prior transactions for each
    # ProductCD — consistent with all other aggregation features.
    # WHY -1 fill: consistent with all other aggregation features — signals
    # "no prior history" to tree models without distorting the distribution.
    if verbose:
        print("\n   Computing amt_vs_product_median ...")

    product_median = (
        df.groupby(product_col)["TransactionAmt"]
        .expanding()
        .median()
        .shift(1)
        .reset_index(level=0, drop=True)
    )

    df["amt_vs_product_median"] = (
        df["TransactionAmt"] / product_median
    ).replace([np.inf, -np.inf], missing_fill).fillna(missing_fill).astype(np.float32)

    new_cols = ["is_new_product", "amt_vs_product_median"]

    if verbose:
        value_counts = df["is_new_product"].value_counts().to_dict()
        new_pct = value_counts.get(1, 0) / len(df) * 100
        print(f"\n   Shape after  : {df.shape}")
        print(f"   is_new_product distribution    : {value_counts}")
        print(f"   First-time product purchases   : {new_pct:.1f}% of all transactions")
        print(f"   amt_vs_product_median — NaN    : {df['amt_vs_product_median'].isna().sum()}")
        print(f"   amt_vs_product_median — filled : {(df['amt_vs_product_median'] == missing_fill).sum()} rows with {missing_fill}")
        nan_remaining = df[["is_new_product", "amt_vs_product_median"]].isna().sum().sum()
        if nan_remaining == 0:
            print("   NaN check: 0 unexpected NaN ✓")
        else:
            print(f"   WARNING: {nan_remaining} unexpected NaN remaining")
        print("=" * 60)

    return df, new_cols