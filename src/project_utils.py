"""
project_utils.py — Shared project utilities for all notebooks
══════════════════════════════════════════════════════════════
General-purpose helpers reusable across all sprint versions.
Kept separate from config.py (paths/constants) by single responsibility principle.

Functions:
    print_project_structure() — display annotated project directory tree
"""

import os
from pathlib import Path


# ── FILE ANNOTATIONS ──────────────────────────────────────────────────────────
# Describes known files in the project — shown next to filenames in the tree.
# Add new entries here as the project grows; unknown files are shown without annotation.
_FILE_ANNOTATIONS = {
    # Config & shared
    "config.py":            "shared — paths, constants, seeds",
    "data_loader.py":       "shared — load/merge/save raw and processed data",
    "feature_init_utils.py":"shared — initial features from EDA (time, device)",
    "project_utils.py":     "shared — project utilities (this file)",
    # Split
    "split_v0.py":          "v0 — time-based train/val split",
    # Preprocessing
    "preproc_v0.py":        "v0 — label encoding + fill NaN",
    "preproc_v1.py":        "v1 — user aggregations + velocity + email/device instability",
    # Training
    "train_v0.py":          "v0/v1 — LightGBM, default params, early stopping",
    # Evaluation
    "evaluate_v0.py":       "v0/v1 — ROC AUC, PR AUC, plots, feature importance",
    # Notebooks
    "baseline_v0.ipynb":    "baseline — raw features, ROC 0.9196, PR 0.5804",
    "v1_aggr.ipynb":        "sprint 1 — user aggregations + velocity (18 new features)",
    "01_eda.ipynb":         "EDA — exploratory data analysis",
    # Data
    "train.parquet":        "processed train (merged + memory reduced)",
    "test.parquet":         "processed test  (merged + memory reduced)",
    "train_transaction.csv":"raw",
    "train_identity.csv":   "raw",
    "test_transaction.csv": "raw",
    "test_identity.csv":    "raw",
    "sample_submission.csv":"raw — submission format",
}

# ── DIRECTORY ANNOTATIONS ─────────────────────────────────────────────────────
_DIR_ANNOTATIONS = {
    "data":       "all data files",
    "raw":        "original Kaggle CSVs — never modified",
    "processed":  "merged + memory-reduced parquet files",
    "v0":         "baseline version modules",
    "v1":         "sprint 1 version modules",
    
}

# ── DISPLAY SETTINGS ──────────────────────────────────────────────────────────
# Directories to skip entirely — not relevant to project structure
_SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", ".idea", "node_modules"}


def print_project_structure(
    root=None,
    max_depth=4,
    show_annotations=True,
    verbose=True,
):
    """
    Print an annotated directory tree of the project.

    Designed to be called at the top of every notebook to confirm:
      - correct working directory and paths
      - all expected files are present
      - project layout is consistent across versions

    Parameters
    ----------
    root           : str or Path or None — project root to display.
                     Default=None: auto-detected from config.PROJECT_ROOT.
                     Pass an explicit path to override.
    max_depth      : int — how many directory levels to show (default=4).
                     Prevents noise from deeply nested folders.
    show_annotations: bool — show file/dir descriptions (default=True).
                     Set False for compact output.
    verbose        : bool — print header and summary line (default=True).

    Returns
    -------
    None — output is printed only.
    """
    # Resolve root: use config.PROJECT_ROOT by default
    if root is None:
        try:
            from config import PROJECT_ROOT
            root = Path(PROJECT_ROOT)
        except ImportError:
            # Fallback: use current working directory
            root = Path(os.getcwd())

    root = Path(root)

    if verbose:
        print("=" * 60)
        print(f"PROJECT STRUCTURE")
        print(f"Root: {root}")
        print("=" * 60)

    _print_tree(root, prefix="", depth=0, max_depth=max_depth,
                show_annotations=show_annotations)

    if verbose:
        print()


def _print_tree(path, prefix, depth, max_depth, show_annotations):
    """
    Recursive helper — prints one level of the directory tree.

    Parameters
    ----------
    path             : Path — current directory to list
    prefix           : str — indentation string built up recursively
    depth            : int — current depth level
    max_depth        : int — stop recursing beyond this depth
    show_annotations : bool — whether to append file/dir descriptions
    """
    if depth > max_depth:
        return

    try:
        # Sort: directories first, then files — both alphabetically
        entries = sorted(path.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
    except PermissionError:
        return

    # Filter out noise directories
    entries = [e for e in entries if e.name not in _SKIP_DIRS]

    for i, entry in enumerate(entries):
        is_last = (i == len(entries) - 1)
        connector = "└── " if is_last else "├── "
        extender  = "    " if is_last else "│   "

        # Build annotation string
        annotation = ""
        if show_annotations:
            if entry.is_dir():
                note = _DIR_ANNOTATIONS.get(entry.name, "")
            else:
                note = _FILE_ANNOTATIONS.get(entry.name, "")
            if note:
                annotation = f"  ← {note}"

        print(f"{prefix}{connector}{entry.name}{annotation}")

        # Recurse into directories
        if entry.is_dir():
            _print_tree(entry, prefix + extender, depth + 1,
                        max_depth, show_annotations)