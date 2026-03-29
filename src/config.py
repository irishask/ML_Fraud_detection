"""
config.py — Shared configuration for all versions
═══════════════════════════════════════════════════
Central place for paths, constants, and defaults.
All version notebooks import from here.
"""

from __future__ import annotations

from pathlib import Path


def detect_project_root(
    start_path: Path | None = None,
    markers: tuple[str, ...] = (
        "pyproject.toml",
        ".git",
        "requirements.txt",
        "README.md",
        "data",
        "src",
    ),
    max_up_levels: int = 12,
) -> Path:
    """
    Detect repository root by walking up parent directories until a marker is found.

    Markers can be files or directories. The first parent that contains ANY marker is returned.
    """
    start = (start_path or Path(__file__).resolve()).parent
    current = start

    for _ in range(max_up_levels + 1):
        for m in markers:
            if (current / m).exists():
                return current
        if current.parent == current:
            break
        current = current.parent

    raise RuntimeError(
        f"PROJECT_ROOT not found. Started at: {start}. "
        f"Searched up {max_up_levels} levels. "
        f"Expected one of markers: {markers}"
    )


# ── PROJECT ROOT ──────────────────────────────────
PROJECT_ROOT = detect_project_root()

# ── PATHS ─────────────────────────────────────────
DATA_RAW       = PROJECT_ROOT / "data" / "raw"

# WHY "data/" not "data/processed/": load_processed() saves merged parquet
# directly to data/train.parquet and data/test.parquet, not to data/processed/.
DATA_PROCESSED = PROJECT_ROOT / "data"

OUTPUTS_DIR  = PROJECT_ROOT / "outputs"

# WHY separate constants per subdirectory: each notebook imports only what it
# needs — no string construction scattered across notebooks.
ENRICHED_DIR = OUTPUTS_DIR / "enriched"   # output of 02_feature_engineering.ipynb
PREPROC_DIR  = OUTPUTS_DIR / "preproc"    # output of 03_preprocess_train.ipynb

# ── ENRICHED FILE PATHS ───────────────────────────
# WHY explicit path constants: avoids string construction in notebooks and
# ensures a single source of truth for all artifact locations.
TRAIN_ENRICHED = ENRICHED_DIR / "train_enriched.parquet"  # 60% train features
Y_TRAIN        = ENRICHED_DIR / "y_train.parquet"         # 60% train labels
VAL_ENRICHED   = ENRICHED_DIR / "val_enriched.parquet"    # 20% val features (early stopping + Optuna)
Y_VAL          = ENRICHED_DIR / "y_val.parquet"           # 20% val labels
TEST_ENRICHED  = ENRICHED_DIR / "test_enriched.parquet"   # 20% frozen TEST features (touched once)
Y_TEST         = ENRICHED_DIR / "y_test.parquet"          # 20% frozen TEST labels

# ── PREPROC FILE PATHS ────────────────────────────
# WHY explicit path constants: same reason as above — single source of truth.
X_TRAIN_LGBM   = PREPROC_DIR / "X_train_lgbm.parquet"    # encoded 60% train
X_VAL_LGBM     = PREPROC_DIR / "X_val_lgbm.parquet"      # encoded 20% val
X_TEST_LGBM    = PREPROC_DIR / "X_test_lgbm.parquet"     # encoded 20% frozen TEST

# ── REPRODUCIBILITY ───────────────────────────────
SEED = 42

# ── COLUMN NAMES ──────────────────────────────────
TARGET = "isFraud"
ID_COL = "TransactionID"
TIME_COL = "TransactionDT"

# ── COLUMNS TO EXCLUDE FROM MODEL FEATURES ────────
NON_FEATURE_COLS = [TARGET, ID_COL, TIME_COL]