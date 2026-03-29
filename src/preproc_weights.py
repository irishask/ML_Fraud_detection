"""
preproc_weights.py — Instance Weights for Temporal Chunk Weighting (V3)
════════════════════════════════════════════════════════════════════════
Assigns sample weights to train transactions based on temporal chunks.

WHY instance weighting:
    Fraud patterns evolve over time (concept drift). Transactions closer
    to the validation period (days 101–141) are more representative of
    future fraud behavior than older ones. Higher weights = stronger
    gradient influence on model training.

WHY 4 chunks with linear weights [1.0, 2.0, 3.0, 4.0]:
    4 chunks of ~25 days each provide a smooth enough gradient without
    being too granular. Linear weights avoid extreme suppression of old
    transactions while still giving recent ones significantly more influence.
    days 1–25:   weight=1.0 — oldest, least representative
    days 26–50:  weight=2.0
    days 51–75:  weight=3.0
    days 76–101: weight=4.0 — closest to val period, most representative

WHY NOT removing old transactions:
    Our EDA confirmed no structural drift over 6 months — old transactions
    still carry valid fraud signal. Weighting preserves all signal while
    prioritizing recent patterns.

WHY weights applied ONLY to train:
    Val and test are used for unbiased evaluation — applying weights
    there would distort the evaluation metric.

Functions:
    compute_sample_weights() — assign per-row weights based on tx_day chunks
"""

import numpy as np
import pandas as pd


# ── Default chunk configuration ───────────────────────────────────────────────

# Temporal chunks defined as (day_start, day_end) inclusive.
# WHY these boundaries: train covers days 1–101 split into 4 equal ~25-day chunks.
# Boundaries chosen to divide train period evenly — not based on fraud patterns.
DEFAULT_CHUNK_DAYS = [
    (1,   25),   # chunk 1 — oldest
    (26,  50),   # chunk 2
    (51,  75),   # chunk 3
    (76, 101),   # chunk 4 — most recent, closest to val period
]

# Weight assigned to each chunk — must match len(DEFAULT_CHUNK_DAYS).
# WHY linear [1,2,3,4]: smooth gradient that avoids extreme suppression
# of old transactions while giving 4× more influence to recent ones.
DEFAULT_WEIGHTS = [1.0, 2.0, 3.0, 4.0]

# Default fill weight for transactions that fall outside all defined chunks.
# WHY 1.0: conservative — treat unknown periods as least important.
DEFAULT_FILL_WEIGHT = 1.0


# ── Public API ────────────────────────────────────────────────────────────────

def compute_sample_weights(df,
                           time_col="tx_day",
                           chunk_days=None,
                           weights=None,
                           fill_weight=DEFAULT_FILL_WEIGHT,
                           verbose=True):
    """
    Assign per-row sample weights based on temporal chunks.

    WHY tx_day not TransactionDT:
        tx_day is already computed in feature engineering (add_time_features)
        as TransactionDT // 86400 — integer days from reference. It is always
        present in enriched and preprocessed parquets. Using TransactionDT
        directly would require the same division — tx_day is cleaner.

    No-leakage guarantee:
        Weights are derived purely from row position in time — not from
        isFraud or any future data. Applying them to train only ensures
        val and test evaluation remains unbiased.

    Parameters
    ----------
    df          : pd.DataFrame — train split (must contain time_col)
    time_col    : str          — day column (default: 'tx_day')
                  WHY named param: allows override if column is renamed
    chunk_days  : list of (int, int) | None — (day_start, day_end) inclusive
                  boundaries for each chunk (default: DEFAULT_CHUNK_DAYS)
                  WHY named param: allows experimenting with different chunk sizes
    weights     : list of float | None — weight per chunk (default: DEFAULT_WEIGHTS)
                  WHY named param: allows experimenting with exponential weights
                  Must have same length as chunk_days.
    fill_weight : float — weight for rows outside all chunks (default: 1.0)
    verbose     : bool  — print weight distribution (default: True)

    Returns
    -------
    pd.Series — sample weights aligned with df.index, dtype float32
    """
    chunk_days = chunk_days or DEFAULT_CHUNK_DAYS
    weights    = weights    or DEFAULT_WEIGHTS

    if len(chunk_days) != len(weights):
        raise ValueError(
            f"chunk_days and weights must have the same length. "
            f"Got {len(chunk_days)} chunks and {len(weights)} weights."
        )

    if time_col not in df.columns:
        raise KeyError(
            f"Column '{time_col}' not found in DataFrame. "
            f"Available columns: {df.columns.tolist()[:10]}..."
        )

    if verbose:
        print("=" * 60)
        print("SAMPLE WEIGHTS — Temporal Chunk Weighting")
        print("=" * 60)
        print(f"   Time column : {time_col}")
        print(f"   Chunks      : {len(chunk_days)}")
        for i, ((d_start, d_end), w) in enumerate(zip(chunk_days, weights)):
            print(f"     Chunk {i+1}: days {d_start:>3}–{d_end:>3}  →  weight={w:.1f}")

    # Initialize all weights to fill_weight
    sample_weights = pd.Series(
        fill_weight, index=df.index, dtype=np.float32, name="sample_weight"
    )

    # Assign weights per chunk
    for (d_start, d_end), w in zip(chunk_days, weights):
        mask = (df[time_col] >= d_start) & (df[time_col] <= d_end)
        sample_weights.loc[mask] = w

    if verbose:
        print(f"\n   Weight distribution:")
        vc = sample_weights.value_counts().sort_index()
        total = len(sample_weights)
        for w_val, count in vc.items():
            print(f"     weight={w_val:.1f} : {count:>7,} rows ({count/total*100:.1f}%)")

        outside = (sample_weights == fill_weight).sum()
        chunks_sum = sum(
            ((df[time_col] >= d_start) & (df[time_col] <= d_end)).sum()
            for (d_start, d_end) in chunk_days
        )
        unassigned = total - chunks_sum
        if unassigned > 0:
            print(f"\n   WARNING: {unassigned} rows outside defined chunks "
                  f"→ assigned fill_weight={fill_weight}")
        else:
            print(f"\n   Coverage: all {total:,} rows assigned ✓")
        print("=" * 60)

    return sample_weights
