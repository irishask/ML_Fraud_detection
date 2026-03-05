"""
preproc_behavioral.py — Behavioral Fingerprint Features
════════════════════════════════════════════════════════
Computes 4 features that measure how much each transaction deviates
from the personal behavioral norm of the user (card1 + addr1).

MODEL-AGNOSTIC — used by all models (LightGBM, XGBoost, CatBoost).

Must be called on the FULL dataset (train + test concatenated) BEFORE
time_split(), so that test transactions see their complete prior history.

No-leakage guarantee:
    All features use groupby + expanding().shift(1) — each row sees only
    chronologically prior transactions. The current transaction never
    contributes to its own feature value. isFraud is never used.

New features (4 total):
    amt_vs_personal_median — TransactionAmt / user's historical median amount
    amt_z_score            — (TransactionAmt - user mean) / user std
    hour_vs_typical        — abs(tx_hour - user's historical mean hour)
    uid_time_entropy       — entropy of user's historical tx_hour distribution

Public functions:
    compute_behavioral_features() — compute all 4 features on a DataFrame
"""

import numpy as np
import pandas as pd


# ── Configuration ─────────────────────────────────────────────────────────────

# User proxy: card1 + addr1.
# WHY same as preproc_agg.py: consistent UID definition across all feature modules.
# Imported as default — caller can override via group_cols parameter.
USER_GROUP_COLS = ["card1", "addr1"]

# Column names.
TIME_COL = "TransactionDT"
AMT_COL  = "TransactionAmt"

# Derived hour column — must match feature_init_utils.py if already computed,
# otherwise computed here from TransactionDT.
# WHY seconds % 86400 // 3600: TransactionDT is seconds since a reference point;
# modulo extracts time-of-day, integer division gives hour 0–23.
HOUR_COL         = "tx_hour"
SECONDS_PER_DAY  = 86_400
SECONDS_PER_HOUR = 3_600

# Number of hour bins for entropy calculation (0–23).
# WHY 24: one bin per hour — natural granularity for time-of-day patterns.
N_HOUR_BINS = 24

# Fill values for rows with no prior history (first transaction per user).
# WHY -1.0 for ratio/z-score: consistent with preproc_agg.py NO_HISTORY_FILL;
#   LightGBM treats -1 as a valid split candidate and learns its meaning.
# WHY 0.0 for hour deviation: no prior history → no known typical hour →
#   deviation of 0 is the least biased assumption (not anomalous by default).
# WHY 0.0 for entropy: no prior history → distribution is undefined →
#   0 entropy (no information) is the correct neutral value.
NO_HISTORY_FILL_RATIO   = -1.0
NO_HISTORY_FILL_ZSCORE  = -1.0
NO_HISTORY_FILL_HOUR    =  0.0
NO_HISTORY_FILL_ENTROPY =  0.0

# Fill for z-score when std = 0 (all prior transactions have identical amount).
# WHY 0.0: current amount equals the only known value → zero deviation, not anomalous.
ZERO_STD_FILL = 0.0


# ── Internal helpers ──────────────────────────────────────────────────────────

def _ensure_hour_col(df, time_col, hour_col, seconds_per_day, seconds_per_hour):
    """
    Add tx_hour column if not already present.

    WHY check before computing: feature_init_utils.py may have already added
    tx_hour in an earlier pipeline step. Recomputing would be redundant and
    could introduce inconsistency if the formula differs.

    Parameters
    ----------
    df               : pd.DataFrame
    time_col         : str   — seconds since reference point
    hour_col         : str   — name of the hour column to create
    seconds_per_day  : int   — seconds in one day (86400)
    seconds_per_hour : int   — seconds in one hour (3600)

    Returns
    -------
    pd.DataFrame — df with hour_col present (added if missing)
    """
    if hour_col not in df.columns:
        df = df.copy()
        # time_col % seconds_per_day → seconds elapsed today (0..86399)
        # // seconds_per_hour        → hour of day (0..23)
        df[hour_col] = (df[time_col] % seconds_per_day // seconds_per_hour).astype(np.int8)
    return df


def _expanding_stat(df, group_cols, col, func):
    """
    Compute an expanding statistic per group, shifted by 1 (no leakage).

    Pattern: groupby → transform(expanding → stat → shift(1))
    WHY transform: returns a Series aligned to df.index — no merge needed.
    WHY shift(1): excludes the current row from its own statistic.

    Parameters
    ----------
    df         : pd.DataFrame — must be sorted by TIME_COL before calling
    group_cols : list[str]    — groupby keys (UID)
    col        : str          — column to aggregate
    func       : callable     — applied inside transform, e.g.:
                                lambda x: x.expanding().mean().shift(1)

    Returns
    -------
    pd.Series — aligned to df.index
    """
    return df.groupby(group_cols)[col].transform(func)


def _compute_entropy(series):
    """
    Compute Shannon entropy (bits) over hour bins for a single user's history.

    Called inside groupby.transform on the tx_hour column.
    Each call receives the full sorted history up to (but not including)
    the current row — shift(1) is applied after transform returns.

    Formula: H = -sum(p_i * log2(p_i)) for p_i > 0
    WHY log2: entropy in bits — standard for discrete distributions.
    WHY ignore p=0: 0 * log2(0) is defined as 0 by convention (limit).

    Parameters
    ----------
    series : pd.Series — tx_hour values for one user (integers 0–23)

    Returns
    -------
    pd.Series — entropy value at each position (expanding, not yet shifted)
    """
    result = np.zeros(len(series), dtype=np.float32)

    for i in range(1, len(series)):
        # History = all rows before current (shift(1) will move this to row i)
        hist = series.iloc[:i].values
        counts = np.bincount(hist.astype(int), minlength=N_HOUR_BINS).astype(np.float32)
        total  = counts.sum()
        if total == 0:
            result[i] = 0.0
            continue
        probs = counts / total
        # Entropy: ignore zero probabilities (0 * log2(0) = 0 by convention)
        nonzero = probs[probs > 0]
        result[i] = float(-np.sum(nonzero * np.log2(nonzero)))

    # result[0] remains 0.0 — first transaction has no history
    return pd.Series(result, index=series.index, dtype=np.float32)


# ── Feature 1: amt_vs_personal_median ────────────────────────────────────────

def _compute_amt_vs_personal_median(df, group_cols, amt_col, fill_val):
    """
    Ratio of current transaction amount to user's historical median amount.

    Formula: TransactionAmt / expanding_median(TransactionAmt).shift(1)

    WHY median not mean: median is robust to prior outliers in the user's
    history. A single large legitimate purchase does not inflate the baseline,
    so future anomalies remain detectable.

    WHY ratio not absolute difference: ratio is scale-invariant — a user who
    normally spends $10 and makes a $100 transaction (ratio=10) is more
    anomalous than a user who normally spends $1000 and makes a $2000
    transaction (ratio=2), even though the absolute difference is larger
    in the second case.

    Edge case: median = 0 → ratio undefined → filled with fill_val.
    """
    expanding_median = _expanding_stat(
        df, group_cols, amt_col,
        lambda x: x.expanding().median().shift(1)
    )
    ratio = df[amt_col] / expanding_median.replace(0, np.nan)
    return ratio.fillna(fill_val).astype(np.float32)


# ── Feature 2: amt_z_score ────────────────────────────────────────────────────

def _compute_amt_z_score(df, group_cols, amt_col, fill_val, zero_std_fill):
    """
    Z-score of current transaction amount relative to user's history.

    Formula: (TransactionAmt - expanding_mean) / expanding_std  [both .shift(1)]

    WHY z-score alongside median ratio: median is robust to outliers but
    insensitive to distribution shape. Z-score uses mean and std — it is
    sensitive to outliers in the user's history, capturing a different
    signal. The two features are complementary.

    Edge cases:
        - std = 0 (all prior amounts identical) → zero_std_fill (0.0)
          WHY: current amount equals the only known pattern → not anomalous.
        - No prior history (NaN mean/std) → fill_val (-1.0)
          WHY: consistent with NO_HISTORY_FILL across all ratio features.
    """
    expanding_mean = _expanding_stat(
        df, group_cols, amt_col,
        lambda x: x.expanding().mean().shift(1)
    )
    expanding_std = _expanding_stat(
        df, group_cols, amt_col,
        lambda x: x.expanding().std().shift(1)
    )

    # Replace std=0 with NaN so division produces NaN (handled below)
    safe_std = expanding_std.replace(0, np.nan)
    z = (df[amt_col] - expanding_mean) / safe_std

    # NaN from std=0 → zero_std_fill; NaN from no history → fill_val
    z = z.where(expanding_std.notna() & (expanding_std != 0),
                other=z.where(expanding_std.isna(), zero_std_fill))
    return z.fillna(fill_val).astype(np.float32)


# ── Feature 3: hour_vs_typical ────────────────────────────────────────────────

def _compute_hour_vs_typical(df, group_cols, hour_col, fill_val):
    """
    Absolute deviation of current transaction hour from user's typical hour.

    Formula: abs(tx_hour - expanding_mean(tx_hour).shift(1))

    WHY mean not median for hours: hour distribution for most users is
    multimodal (morning and evening peaks). Mean captures the centroid of
    activity — deviating from it is the anomaly signal regardless of
    which mode is active.

    WHY absolute value: we care about distance from typical, not direction.
    A transaction 6 hours before or after the typical time is equally
    suspicious.

    Edge case: no prior history → fill_val (0.0 — not anomalous by default).
    """
    expanding_mean_hour = _expanding_stat(
        df, group_cols, hour_col,
        lambda x: x.expanding().mean().shift(1)
    )
    deviation = (df[hour_col] - expanding_mean_hour).abs()
    return deviation.fillna(fill_val).astype(np.float32)


# ── Feature 4: uid_time_entropy ───────────────────────────────────────────────

def _compute_uid_time_entropy(df, group_cols, hour_col, fill_val):
    """
    Shannon entropy of user's historical transaction hour distribution.

    Low entropy  → user transacts at predictable hours → bot-like pattern
    High entropy → user transacts at varied hours → human-like pattern

    WHY entropy over nunique hours: nunique counts distinct hours but ignores
    frequency. Entropy accounts for both variety and balance — a user who
    always uses 24 different hours equally has maximum entropy; a user who
    uses 24 hours but 99% of the time at one hour has near-zero entropy.

    WHY expanding (not rolling): bot detection benefits from long-term
    behavioral baseline. A fraudster who varies hours in a short window
    but is consistent over weeks is still detectable with expanding history.

    Edge case: first transaction (no history) → fill_val (0.0).
    """
    entropy = df.groupby(group_cols)[hour_col].transform(_compute_entropy)
    return entropy.fillna(fill_val).astype(np.float32)


# ── Public API ────────────────────────────────────────────────────────────────

def compute_behavioral_features(df,
                                 group_cols=None,
                                 time_col=TIME_COL,
                                 amt_col=AMT_COL,
                                 hour_col=HOUR_COL,
                                 seconds_per_day=SECONDS_PER_DAY,
                                 seconds_per_hour=SECONDS_PER_HOUR,
                                 no_history_fill_ratio=NO_HISTORY_FILL_RATIO,
                                 no_history_fill_zscore=NO_HISTORY_FILL_ZSCORE,
                                 no_history_fill_hour=NO_HISTORY_FILL_HOUR,
                                 no_history_fill_entropy=NO_HISTORY_FILL_ENTROPY,
                                 zero_std_fill=ZERO_STD_FILL,
                                 verbose=True):
    """
    Compute all 4 behavioral fingerprint features on a DataFrame.

    MODEL-AGNOSTIC: output is used by LightGBM, XGBoost, and CatBoost.

    IMPORTANT: must be called on the FULL dataset (train + test concatenated)
    BEFORE time_split(), sorted by time_col. If called after compute_user_aggregations(),
    the DataFrame is already sorted — no re-sorting needed.

    No leakage: all features use expanding().shift(1) — each row sees only
    chronologically prior transactions. isFraud is never accessed.

    Parameters
    ----------
    df                      : pd.DataFrame — full dataset sorted by time_col
    group_cols              : list[str]    — user proxy columns
                              WHY default USER_GROUP_COLS: card1+addr1 is the
                              confirmed UID from EDA (D1 too unstable as anchor)
    time_col                : str          — timestamp column in seconds
    amt_col                 : str          — transaction amount column
    hour_col                : str          — hour-of-day column (0–23);
                              computed from time_col if not present
    seconds_per_day         : int          — seconds per day for hour extraction
                              WHY named param: makes unit conversion explicit,
                              not a magic number
    seconds_per_hour        : int          — seconds per hour for hour extraction
    no_history_fill_ratio   : float        — fill for amt_vs_personal_median
                              when no prior history exists (default: -1.0)
                              WHY -1.0: consistent with preproc_agg fill values;
                              LightGBM learns -1 as "missing history" signal
    no_history_fill_zscore  : float        — fill for amt_z_score when no
                              prior history exists (default: -1.0)
    no_history_fill_hour    : float        — fill for hour_vs_typical when no
                              prior history exists (default: 0.0)
                              WHY 0.0: no history → no deviation → neutral
    no_history_fill_entropy : float        — fill for uid_time_entropy when no
                              prior history exists (default: 0.0)
                              WHY 0.0: no history → undefined distribution →
                              zero entropy is the correct neutral value
    zero_std_fill           : float        — fill for amt_z_score when prior
                              std = 0 (all prior amounts identical, default: 0.0)
                              WHY 0.0: identical history → current amount is
                              not anomalous relative to known pattern
    verbose                 : bool         — print progress (default: True)

    Returns
    -------
    tuple (df, new_feature_cols)
        df               — input DataFrame with 4 new columns appended
        new_feature_cols — list of new column names (for downstream tracking)
    """
    group_cols = group_cols or USER_GROUP_COLS

    if verbose:
        print("=" * 60)
        print("FEATURE ENGINEERING — Behavioral Fingerprint")
        print("=" * 60)
        print(f"   Group key    : {group_cols}")
        print(f"   Shape before : {df.shape}")

    # CRITICAL: df must be sorted by time_col before any expanding operation.
    # If called after compute_user_aggregations(), already sorted — verify only.
    if not df[time_col].is_monotonic_increasing:
        if verbose:
            print(f"   WARNING: df not sorted by {time_col} — sorting now.")
        df = df.sort_values(time_col).reset_index(drop=True)

    # Ensure tx_hour column exists (add if missing)
    df = _ensure_hour_col(df, time_col, hour_col, seconds_per_day, seconds_per_hour)

    new_cols = []

    # ── Feature 1: amt_vs_personal_median ─────────────────────────────────────
    if verbose:
        print("\n   Computing amt_vs_personal_median ...")
    df["amt_vs_personal_median"] = _compute_amt_vs_personal_median(
        df, group_cols, amt_col, no_history_fill_ratio
    )
    new_cols.append("amt_vs_personal_median")

    # ── Feature 2: amt_z_score ─────────────────────────────────────────────────
    if verbose:
        print("   Computing amt_z_score ...")
    df["amt_z_score"] = _compute_amt_z_score(
        df, group_cols, amt_col, no_history_fill_zscore, zero_std_fill
    )
    new_cols.append("amt_z_score")

    # ── Feature 3: hour_vs_typical ─────────────────────────────────────────────
    if verbose:
        print("   Computing hour_vs_typical ...")
    df["hour_vs_typical"] = _compute_hour_vs_typical(
        df, group_cols, hour_col, no_history_fill_hour
    )
    new_cols.append("hour_vs_typical")

    # ── Feature 4: uid_time_entropy ────────────────────────────────────────────
    if verbose:
        print("   Computing uid_time_entropy (entropy loop — may take ~1-2 min) ...")
    df["uid_time_entropy"] = _compute_uid_time_entropy(
        df, group_cols, hour_col, no_history_fill_entropy
    )
    new_cols.append("uid_time_entropy")

    if verbose:
        print(f"\n   Shape after  : {df.shape}")
        print(f"   New features : {new_cols}")
        print()
        # Quick NaN check — all fills should have been applied
        nan_counts = df[new_cols].isna().sum()
        if nan_counts.sum() == 0:
            print("   NaN check: 0 unexpected NaN in new features ✓")
        else:
            print(f"   WARNING — unexpected NaN in new features:\n{nan_counts[nan_counts > 0]}")
        print("=" * 60)

    return df, new_cols