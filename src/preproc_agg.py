"""
preproc_agg.py — User Behavior Aggregations (Sprint 1)
═══════════════════════════════════════════════════════
Computes 18 new features per user (card1 + addr1), strictly before each
transaction (no data leakage).

This module is MODEL-AGNOSTIC — used by all models:
  LightGBM, XGBoost, CatBoost, and any future models.

Must be called on the FULL train dataset BEFORE time_split() so that
val/test transactions see their complete prior history from train.

All features are computed strictly BEFORE the current transaction:
  - Cumulative features : groupby + transform(expanding().shift(1))
  - Rolling window      : groupby + transform(rolling(window=N_seconds, closed='left'))

New features (18 total):
  Cumulative aggregations (per card1+addr1, till date):
    tx_count, tx_amt_mean, tx_amt_std, tx_amt_min, tx_amt_max,
    tx_amt_ratio, time_since_last_tx, delta_amt

  Email instability (per card1+addr1, till date):
    nunique_P_email, is_new_P_email,
    nunique_R_email, is_new_R_email,
    is_same_email_domain

  Device instability (per card1+addr1, till date):
    nunique_device, is_new_device

  Rolling window velocity (per card1+addr1):
    tx_count_last_3d, tx_count_last_7d, tx_count_last_30d

Public functions:
    compute_user_aggregations()   — compute all 18 features on a DataFrame
    test_find_users()             — find representative fraud/legit users
    test_show_user_aggregations() — display aggregations for sanity check
"""

import numpy as np
import pandas as pd


# ── Configuration ─────────────────────────────────────────────────────────────

# User proxy: card1 + addr1.
# WHY card1: primary card identifier — most unique card key in the dataset.
# WHY addr1: billing region — refines card identity without over-fragmenting groups.
#   addr1 is billing address, NOT shipping address.
# WHY NOT P_emaildomain in group key: email is unstable per user (one person has
#   multiple accounts; fraudster changes email) — used as instability signal instead.
USER_GROUP_COLS = ["card1", "addr1"]

# Timestamp column (seconds since epoch).
TIME_COL = "TransactionDT"

# Transaction amount column.
AMT_COL = "TransactionAmt"

# Rolling window sizes in days.
# WHY 3d : card testing — fraudster makes small transactions to verify card is active.
# WHY 7d : weekly burst — rapid purchases before card is blocked by issuing bank.
# WHY 30d: monthly baseline — detects sleeping fraudster activating after a long pause.
# WHY NOT 1h : very sparse counts (most users have 0 transactions in any 1h window).
# WHY NOT 60d: dataset spans ~6 months; 60d windows are unreliable for most UIDs.
VELOCITY_WINDOWS_DAYS = [3, 7, 30]

# Seconds in one day — used to convert window days to seconds for rolling().
# WHY named constant: TransactionDT is in seconds; windows must match that unit.
SECONDS_PER_DAY = 86_400

# Fill value for rows with no prior history (first transaction per user)
# and for unseen groups in val/test.
# WHY -1: consistent with fill_missing() in preproc_v0; LightGBM treats -1 as missing.
NO_HISTORY_FILL = -1.0


# ── Feature Engineering: Cumulative Aggregations ──────────────────────────────

def _compute_cumulative_features(df, group_cols, time_col, amt_col, fill_val):
    """
    Compute per-user cumulative statistics strictly before each transaction.

    Uses groupby + transform(expanding().shift(1)):
      - expanding() accumulates all rows up to and including the current row
      - shift(1) moves the result one row down — each row sees only past rows
      This is the correct no-leakage pattern for cumulative features.

    Features added:
        tx_count           — number of prior transactions for this user
        tx_amt_mean        — mean amount of prior transactions
        tx_amt_std         — std of prior amounts (variability = fraud signal)
        tx_amt_min         — min prior amount
        tx_amt_max         — max prior amount
        tx_amt_ratio       — current amount / prior mean (anomaly signal: ratio >> 1)
        time_since_last_tx — seconds since the previous transaction
        delta_amt          — current amount minus previous amount

    Parameters
    ----------
    df         : pd.DataFrame — sorted by time_col ascending
    group_cols : list[str]    — user group columns
    time_col   : str          — timestamp column
    amt_col    : str          — amount column
    fill_val   : float        — fill for first-transaction rows (no prior history)

    Returns
    -------
    pd.DataFrame — df with new columns appended in-place
    """
    grp = df.groupby(group_cols)

    # tx_count: cumcount() returns 0 for the first row, 1 for second, etc.
    # Equals the number of PRIOR transactions — exactly what we need.
    df["tx_count"] = grp[amt_col].cumcount().astype(np.float32)

    # Cumulative stats — expanding with shift(1) excludes current row
    for col_name, func in [
        ("tx_amt_mean", lambda x: x.expanding().mean().shift(1)),
        ("tx_amt_std",  lambda x: x.expanding().std().shift(1)),
        ("tx_amt_min",  lambda x: x.expanding().min().shift(1)),
        ("tx_amt_max",  lambda x: x.expanding().max().shift(1)),
    ]:
        df[col_name] = grp[amt_col].transform(func).astype(np.float32)

    # tx_amt_ratio: how unusual is the current amount vs user's history?
    # NaN when mean is NaN (first transaction) — filled below.
    df["tx_amt_ratio"] = (df[amt_col] / df["tx_amt_mean"]).astype(np.float32)

    # time_since_last_tx: diff() = current - previous within each group.
    # First transaction per group → NaN (no previous) — filled below.
    df["time_since_last_tx"] = (
        grp[time_col].transform(lambda x: x.diff()).astype(np.float32)
    )

    # delta_amt: current amount - previous amount.
    # Large jump signals card abuse or card-testing behavior.
    df["delta_amt"] = (
        grp[amt_col].transform(lambda x: x.diff()).astype(np.float32)
    )

    # Fill first-transaction rows (no prior history available)
    no_history_cols = [
        "tx_amt_mean", "tx_amt_std", "tx_amt_min", "tx_amt_max",
        "tx_amt_ratio", "time_since_last_tx", "delta_amt",
    ]
    df[no_history_cols] = df[no_history_cols].fillna(fill_val)

    return df


# ── Feature Engineering: Email Instability ────────────────────────────────────

def _compute_email_instability(df, group_cols, fill_val):
    """
    Compute per-user email instability features strictly before each transaction.

    WHY email as instability signal (not group key):
      - Legitimate: one person may have personal + work + multiple email accounts.
      - Fraud: attacker uses stolen card but routes goods to their own email domain.
      Therefore P/R emaildomain are NOT stable user identifiers but anomaly signals.

    Features added:
        nunique_P_email      — distinct P_emaildomain values seen before this tx
        is_new_P_email       — 1 if current P_emaildomain is new for this card
        nunique_R_email      — distinct R_emaildomain values seen before this tx
        is_new_R_email       — 1 if current R_emaildomain is new for this card
        is_same_email_domain — 1 if P_emaildomain == R_emaildomain
            WHY: payer and recipient using different domains signals stolen card.
            NOTE: same domain (e.g. both gmail.com) is a WEAK signal — one person
            can have personal + work Gmail. The mismatch (P != R) is the stronger signal.

    Parameters
    ----------
    df         : pd.DataFrame — sorted by time_col ascending
    group_cols : list[str]    — user group columns
    fill_val   : float        — fill for NaN/missing values

    Returns
    -------
    pd.DataFrame — df with new columns appended in-place
    """
    for email_col, nunique_col, is_new_col in [
        ("P_emaildomain", "nunique_P_email", "is_new_P_email"),
        ("R_emaildomain", "nunique_R_email", "is_new_R_email"),
    ]:
        if email_col not in df.columns:
            df[nunique_col] = fill_val
            df[is_new_col]  = fill_val
            continue

        # Pre-allocate arrays aligned to df index — avoids length mismatch
        # that occurs when sort_index().values is assigned after reset_index()
        nunique_arr = np.full(len(df), fill_val, dtype=np.float32)
        is_new_arr  = np.full(len(df), fill_val, dtype=np.float32)

        # Track seen values strictly before current row (per group)
        for _, group in df.groupby(group_cols, sort=False):
            seen = set()
            for i, val in zip(group.index, group[email_col]):
                nunique_arr[i] = float(len(seen))
                is_new_arr[i]  = (0.0 if val in seen else 1.0) if pd.notna(val) else fill_val
                if pd.notna(val):
                    seen.add(val)

        df[nunique_col] = nunique_arr
        df[is_new_col]  = is_new_arr

    # is_same_email_domain: direct equality — no prior history needed
    if "P_emaildomain" in df.columns and "R_emaildomain" in df.columns:
        same = (df["P_emaildomain"] == df["R_emaildomain"]).astype(np.float32)
        # NaN when either domain is missing — insufficient info to compare
        either_missing = df["P_emaildomain"].isna() | df["R_emaildomain"].isna()
        same[either_missing] = fill_val
        df["is_same_email_domain"] = same
    else:
        df["is_same_email_domain"] = fill_val

    return df


# ── Feature Engineering: Device Instability ───────────────────────────────────

def _compute_device_instability(df, group_cols, fill_val):
    """
    Compute per-user device instability features strictly before each transaction.

    WHY: a fraudster using a stolen card typically uses a different device than
    the legitimate cardholder. New device + new email = strong combined fraud signal.

    Features added:
        nunique_device — distinct DeviceType values seen before this tx
        is_new_device  — 1 if current DeviceType is new for this card

    Parameters
    ----------
    df         : pd.DataFrame — sorted by time_col ascending
    group_cols : list[str]    — user group columns
    fill_val   : float        — fill for NaN/missing values

    Returns
    -------
    pd.DataFrame — df with new columns appended in-place
    """
    device_col = "DeviceType"

    if device_col not in df.columns:
        df["nunique_device"] = fill_val
        df["is_new_device"]  = fill_val
        return df

    # Pre-allocate arrays aligned to df index — same fix as email instability
    nunique_arr = np.full(len(df), fill_val, dtype=np.float32)
    is_new_arr  = np.full(len(df), fill_val, dtype=np.float32)

    for _, group in df.groupby(group_cols, sort=False):
        seen = set()
        for i, val in zip(group.index, group[device_col]):
            nunique_arr[i] = float(len(seen))
            is_new_arr[i]  = (0.0 if val in seen else 1.0) if pd.notna(val) else fill_val
            if pd.notna(val):
                seen.add(val)

    df["nunique_device"] = nunique_arr
    df["is_new_device"]  = is_new_arr

    return df


# ── Feature Engineering: Rolling Window Velocity ──────────────────────────────

def _compute_velocity_features(df, group_cols, time_col,
                                velocity_windows_days, seconds_per_day, fill_val):
    """
    Compute rolling window transaction velocity per user group.

    WHY pandas rolling > loops:
        groupby + transform(rolling) is vectorized — O(n log n) vs O(n * k) loops.
        On 590k rows explicit loops are not practical.

    WHY closed='left':
        Excludes the current row from the window — only past transactions counted.
        This is the correct no-leakage setting for all look-back features.

    WHY offset string (e.g. '259200s') instead of integer rows:
        pandas rolling with DatetimeIndex supports time-based windows via offset
        strings, which correctly handles irregular transaction spacing between users.

    Features added (one per window):
        tx_count_last_{N}d — transaction count in last N days before current tx

    Parameters
    ----------
    df                    : pd.DataFrame — sorted by time_col ascending
    group_cols            : list[str]    — user group columns
    time_col              : str          — timestamp column (seconds)
    velocity_windows_days : list[int]    — window sizes in days
    seconds_per_day       : int          — seconds per day (86400)
    fill_val              : float        — fill when count is NaN

    Returns
    -------
    pd.DataFrame — df with new velocity columns appended in-place
    """
    # Convert TransactionDT (seconds) to DatetimeIndex for time-based rolling
    dt_index = pd.to_datetime(df[time_col], unit="s")
    df_dt    = df.copy()
    df_dt.index = dt_index

    for days in velocity_windows_days:
        window_str = f"{days * seconds_per_day}s"
        col_name   = f"tx_count_last_{days}d"

        # closed='left': window = [current_time - window_size, current_time)
        # count() counts non-NaN time values in the window = transaction count
        counts = (
            df_dt.groupby(group_cols)[time_col]
            .transform(
                lambda x: x.rolling(window=window_str, closed="left").count()
            )
        )

        # NaN → 0.0: rolling returns NaN when window contains zero prior transactions
        df[col_name] = counts.fillna(0.0).values.astype(np.float32)

    return df


# ── Main: Compute All User Aggregations ───────────────────────────────────────

def compute_user_aggregations(df,
                               group_cols=None,
                               time_col=TIME_COL,
                               amt_col=AMT_COL,
                               velocity_windows_days=None,
                               seconds_per_day=SECONDS_PER_DAY,
                               no_history_fill=NO_HISTORY_FILL,
                               verbose=True):
    """
    Compute all 18 user behavior features on a DataFrame.

    MODEL-AGNOSTIC: output is used by LightGBM, XGBoost, and CatBoost.

    IMPORTANT: must be called on the FULL train dataset BEFORE time_split()
    so that val/test transactions see their complete prior history from train.
    No leakage — expanding().shift(1) and closed='left' exclude the current row.

    Internally sorts by time_col to guarantee correct chronological order.
    y must be extracted AFTER this call to remain aligned with the sorted df.

    Parameters
    ----------
    df                    : pd.DataFrame — input data (full train before split)
    group_cols            : list[str]    — user proxy columns (default: USER_GROUP_COLS)
    time_col              : str          — timestamp column in seconds
    amt_col               : str          — amount column
    velocity_windows_days : list[int]    — rolling window sizes in days
    seconds_per_day       : int          — seconds per day for window conversion
    no_history_fill       : float        — fill for missing history (default: -1.0)
    verbose               : bool         — print progress

    Returns
    -------
    pd.DataFrame  — df with 18 new feature columns appended
    list[str]     — names of all new feature columns added
    """
    group_cols            = group_cols            or USER_GROUP_COLS
    velocity_windows_days = velocity_windows_days or VELOCITY_WINDOWS_DAYS

    if verbose:
        print("=" * 60)
        print("FEATURE ENGINEERING — User Behavior Aggregations")
        print("=" * 60)
        print(f"   Group key:    {group_cols}")
        print(f"   Shape before: {df.shape}")

    # CRITICAL: sort by time — all features require chronological order.
    # y must be extracted AFTER this call to stay aligned with sorted df.
    df = df.sort_values(time_col).reset_index(drop=True)

    existing_cols = set(df.columns)

    if verbose:
        print("\n>> Feature Engineering: Cumulative Aggregations...")
    df = _compute_cumulative_features(
        df, group_cols, time_col, amt_col, no_history_fill
    )

    if verbose:
        print(">> Feature Engineering: Email Instability...")
    df = _compute_email_instability(df, group_cols, no_history_fill)

    if verbose:
        print(">> Feature Engineering: Device Instability...")
    df = _compute_device_instability(df, group_cols, no_history_fill)

    if verbose:
        print(">> Feature Engineering: Rolling Window Velocity...")
    df = _compute_velocity_features(
        df, group_cols, time_col,
        velocity_windows_days, seconds_per_day, no_history_fill
    )

    new_cols = [c for c in df.columns if c not in existing_cols]

    if verbose:
        print(f"\n   Shape after:  {df.shape}")
        print(f"   New features ({len(new_cols)}):")
        for col in new_cols:
            print(f"     + {col}")

    return df, new_cols


# ── Sanity Check: Test Functions ──────────────────────────────────────────────

# Transaction count ranges for user selection.
# WHY separate ranges for fraud vs legit:
#   Fraud cards are used in short bursts before the card is blocked —
#   3–10 transactions is realistic and gives enough rows to verify aggregations.
#   Legit cards need richer history (10–20 tx) to show meaningful cumulative
#   and instability patterns across multiple sessions.
_FRAUD_TX_MIN = 3
_FRAUD_TX_MAX = 10
_LEGIT_TX_MIN = 10
_LEGIT_TX_MAX = 20

# Column groups for structured sectioned display.
# Each section shows raw inputs alongside derived features — correctness
# can be verified by reading left-to-right without jumping between columns.
_SECTION_IDENTITY = ["TransactionDT", "TransactionAmt", "isFraud"]

_SECTION_CUMULATIVE = [
    # Raw input first — derived values can be verified against it
    "TransactionAmt",
    "tx_count",
    "tx_amt_mean", "tx_amt_std", "tx_amt_min", "tx_amt_max",
    "tx_amt_ratio",
    "time_since_last_tx",
    "delta_amt",
]

_SECTION_EMAIL = [
    # Raw email columns first — verify instability flags against actual values
    "P_emaildomain", "R_emaildomain",
    "nunique_P_email", "is_new_P_email",
    "nunique_R_email", "is_new_R_email",
    "is_same_email_domain",
]

_SECTION_DEVICE = [
    # Raw device column first — verify instability flags against actual value
    "DeviceType",
    "nunique_device", "is_new_device",
]

_SECTION_VELOCITY = [
    "TransactionDT",
    "tx_count_last_3d", "tx_count_last_7d", "tx_count_last_30d",
]

# Ordered list of (section_title, column_list) for the display loop.
_DISPLAY_SECTIONS = [
    ("Transaction Identity",       _SECTION_IDENTITY),
    ("Cumulative Aggregations",    _SECTION_CUMULATIVE),
    ("Email Instability",          _SECTION_EMAIL),
    ("Device Instability",         _SECTION_DEVICE),
    ("Velocity (Rolling Windows)", _SECTION_VELOCITY),
]


def _pick_closest_to_midpoint(groups_df, tx_col, lo, hi):
    """
    From a filtered DataFrame of user groups, pick the user whose total
    transaction count is closest to the midpoint of [lo, hi].

    WHY midpoint selection: avoids edge cases — a user with exactly lo
    transactions has too little history; a user at hi is noisier.
    Midpoint gives the most representative example.

    Parameters
    ----------
    groups_df : pd.DataFrame — must contain tx_col (transaction count per user)
    tx_col    : str          — column with total transaction counts
    lo        : int          — minimum count (inclusive)
    hi        : int          — maximum count (inclusive)

    Returns
    -------
    pd.Series — the row of the best-matching user
    """
    mid = (lo + hi) / 2.0
    candidates = groups_df[(groups_df[tx_col] >= lo) & (groups_df[tx_col] <= hi)].copy()
    candidates["_dist"] = (candidates[tx_col] - mid).abs()
    return candidates.sort_values("_dist").iloc[0]


def test_find_users(df_raw, y,
                    group_cols=None,
                    fraud_tx_min=_FRAUD_TX_MIN,
                    fraud_tx_max=_FRAUD_TX_MAX,
                    legit_tx_min=_LEGIT_TX_MIN,
                    legit_tx_max=_LEGIT_TX_MAX):
    """
    Find two representative users for sanity checking:
      - FRAUD user: has at least one fraud transaction,
                    total transactions in [fraud_tx_min, fraud_tx_max]
      - LEGIT user: zero fraud transactions,
                    total transactions in [legit_tx_min, legit_tx_max]

    Within each range, the user closest to the midpoint is selected —
    avoids edge cases (too few rows = trivial aggregations,
    too many rows = hard to read manually).

    Parameters
    ----------
    df_raw        : pd.DataFrame — raw train data BEFORE preprocessing
    y             : pd.Series   — target variable (isFraud), aligned with df_raw
    group_cols    : list[str]   — user proxy columns (default: USER_GROUP_COLS)
    fraud_tx_min  : int         — min total tx for fraud user (default: 3).
                    WHY 3: fewer rows give only trivial first-transaction NaN results.
    fraud_tx_max  : int         — max total tx for fraud user (default: 10).
                    WHY 10: fraud cards are blocked quickly — realistic upper bound.
    legit_tx_min  : int         — min total tx for legit user (default: 10).
                    WHY 10: need enough history to verify cumulative aggregations.
    legit_tx_max  : int         — max total tx for legit user (default: 20).
                    WHY 20: beyond 20 rows the table becomes hard to read manually.

    Returns
    -------
    tuple (user_fraud, user_legit)
        user_fraud — tuple of group key values (e.g. (card1_val, addr1_val))
        user_legit — tuple of group key values
    """
    group_cols = group_cols or USER_GROUP_COLS

    df = df_raw.copy()
    df["isFraud"] = y.values

    tx_counts = df.groupby(group_cols).agg(
        total=("isFraud", "count"),
        fraud=("isFraud", "sum"),
    ).reset_index()

    # ── Fraud user ────────────────────────────────────────────────────────────
    fraud_candidates = tx_counts[tx_counts["fraud"] >= 1]

    if len(fraud_candidates) == 0:
        raise ValueError("No users with fraud transactions found.")

    fraud_in_range = fraud_candidates[
        (fraud_candidates["total"] >= fraud_tx_min) &
        (fraud_candidates["total"] <= fraud_tx_max)
    ]

    if len(fraud_in_range) == 0:
        fallback = fraud_candidates.sort_values("total").iloc[0]
        user_fraud = tuple(fallback[group_cols].values)
        print(
            f"  [WARNING] No fraud user found in range [{fraud_tx_min}, {fraud_tx_max}]. "
            f"Using fallback with {int(fallback['total'])} transactions."
        )
    else:
        best = _pick_closest_to_midpoint(fraud_in_range, "total", fraud_tx_min, fraud_tx_max)
        user_fraud = tuple(best[group_cols].values)

    # ── Legit user ────────────────────────────────────────────────────────────
    legit_candidates = tx_counts[tx_counts["fraud"] == 0]

    legit_in_range = legit_candidates[
        (legit_candidates["total"] >= legit_tx_min) &
        (legit_candidates["total"] <= legit_tx_max)
    ]

    if len(legit_in_range) == 0:
        raise ValueError(
            f"No legit users found with total transactions in "
            f"[{legit_tx_min}, {legit_tx_max}]. "
            f"Try adjusting legit_tx_min / legit_tx_max."
        )

    best = _pick_closest_to_midpoint(legit_in_range, "total", legit_tx_min, legit_tx_max)
    user_legit = tuple(best[group_cols].values)

    # ── Summary ───────────────────────────────────────────────────────────────
    fraud_total = int(tx_counts[
        (tx_counts[group_cols[0]] == user_fraud[0]) &
        (tx_counts[group_cols[1]] == user_fraud[1])
    ]["total"].values[0])
    legit_total = int(tx_counts[
        (tx_counts[group_cols[0]] == user_legit[0]) &
        (tx_counts[group_cols[1]] == user_legit[1])
    ]["total"].values[0])

    print(f"Selected FRAUD user: {dict(zip(group_cols, user_fraud))}  "
          f"({fraud_total} transactions)")
    print(f"Selected LEGIT user: {dict(zip(group_cols, user_legit))}  "
          f"({legit_total} transactions)")

    return user_fraud, user_legit


def _print_section(title, user_df, cols, section_width):
    """
    Print one named section of a user's sanity check as a plain-text table.

    Only columns that exist in user_df are shown — missing columns are
    silently skipped so the function is safe even if optional features
    (e.g. R_emaildomain) are absent from a particular dataset split.

    Parameters
    ----------
    title         : str          — section header label
    user_df       : pd.DataFrame — aggregated user transactions, sorted by time
    cols          : list[str]    — columns to display in this section
    section_width : int          — total width of the separator line
    """
    available = [c for c in cols if c in user_df.columns]
    if not available:
        return

    print(f"\n  [{title}]")
    print("  " + "-" * (section_width - 2))
    print(user_df[available].to_string(index=False))


def test_show_user_aggregations(df_raw, y,
                                user_fraud=None,
                                user_legit=None,
                                group_cols=None,
                                time_col=TIME_COL,
                                amt_col=AMT_COL,
                                velocity_windows_days=None,
                                fraud_tx_min=_FRAUD_TX_MIN,
                                fraud_tx_max=_FRAUD_TX_MAX,
                                legit_tx_min=_LEGIT_TX_MIN,
                                legit_tx_max=_LEGIT_TX_MAX):
    """
    Display all transactions for two users (fraud and legit) with aggregated
    features, split into labelled sections for easy manual verification.

    Output structure per user:
        ══ USER HEADER (type, group key, tx count, fraud count) ══
        [Transaction Identity]      — time, amount, fraud label
        [Cumulative Aggregations]   — tx_count, mean/std/ratio, delta
        [Email Instability]         — raw domains + nunique/is_new flags
        [Device Instability]        — raw DeviceType + nunique/is_new flags
        [Velocity (Rolling Windows)]— tx_count_last_3d/7d/30d

    Each section shows raw input columns first, then derived features —
    so correctness can be verified left-to-right without cross-referencing.

    Parameters
    ----------
    df_raw                : pd.DataFrame — raw train data BEFORE preprocessing
    y                     : pd.Series   — target variable (isFraud)
    user_fraud            : tuple       — group key values for fraud user.
                            If None: auto-selected via test_find_users().
    user_legit            : tuple       — group key values for legit user.
                            If None: auto-selected via test_find_users().
    group_cols            : list[str]   — user proxy columns
    time_col              : str         — timestamp column
    amt_col               : str         — amount column
    velocity_windows_days : list[int]   — rolling window sizes in days
    fraud_tx_min          : int         — passed to test_find_users() if auto-selecting
    fraud_tx_max          : int         — passed to test_find_users() if auto-selecting
    legit_tx_min          : int         — passed to test_find_users() if auto-selecting
    legit_tx_max          : int         — passed to test_find_users() if auto-selecting

    Returns
    -------
    dict with keys "fraud" and "legit" — DataFrames with aggregated features,
    sorted by time_col. Useful for further programmatic checks after display.
    """
    group_cols            = group_cols            or USER_GROUP_COLS
    velocity_windows_days = velocity_windows_days or VELOCITY_WINDOWS_DAYS

    if user_fraud is None or user_legit is None:
        user_fraud, user_legit = test_find_users(
            df_raw, y,
            group_cols=group_cols,
            fraud_tx_min=fraud_tx_min,
            fraud_tx_max=fraud_tx_max,
            legit_tx_min=legit_tx_min,
            legit_tx_max=legit_tx_max,
        )

    df = df_raw.copy()
    df["isFraud"] = y.values

    results = {}
    width = 72

    for label, user in [("FRAUD USER", user_fraud), ("LEGIT USER", user_legit)]:

        mask = pd.Series(True, index=df.index)
        for col, val in zip(group_cols, user):
            mask = mask & (df[col] == val)

        user_df = df[mask].copy()

        print(f"\n{'═' * width}")

        if len(user_df) == 0:
            print(f"{label}  |  {dict(zip(group_cols, user))}")
            print("  No transactions found.")
            continue

        # Compute aggregations on this user's data only (no leakage concern —
        # sanity check runs on isolated user slice, not full dataset)
        user_df, _ = compute_user_aggregations(
            user_df,
            group_cols=group_cols,
            time_col=time_col,
            amt_col=amt_col,
            velocity_windows_days=velocity_windows_days,
            verbose=False,
        )

        user_df = user_df.sort_values(time_col).reset_index(drop=True)

        fraud_count = int(user_df["isFraud"].sum()) if "isFraud" in user_df.columns else "?"
        print(
            f"{label}  |  {dict(zip(group_cols, user))}  |  "
            f"{len(user_df)} transactions  |  fraud: {fraud_count}"
        )
        print(f"{'═' * width}")

        for section_title, section_cols in _DISPLAY_SECTIONS:
            _print_section(section_title, user_df, section_cols, width)

        results[label.split()[0].lower()] = user_df

    print(f"\n{'═' * width}")
    return results