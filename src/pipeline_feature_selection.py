"""
pipeline_feature_selection.py — XGBoost on LightGBM Top-N Features
════════════════════════════════════════════════════════════════════
Contains all logic for 05_feature_selection_xgb.ipynb.
The notebook is a pure orchestrator — no logic lives in notebook cells.

Approach:
    Use trained LightGBM as a feature selector for XGBoost.
    LightGBM's leaf-wise growth naturally surfaces engineered features
    (confirmed in nb04: positions 7–19 in top-30 importance).
    XGBoost trained on LightGBM's top-N features gets guaranteed exposure
    to engineered features — no colsample_bytree workaround needed.

No-leakage guarantee:
    LightGBM importance is derived from training data only (model trained
    on X_train). Applying the same feature filter to val/test is standard
    feature selection — no test information leaks into feature selection.

WHY default params for XGBoost first (not Optuna):
    Before investing ~8-10h in Optuna, verify that feature selection itself
    improves Test PR AUC vs nb04 XGBoost (0.4944). Default params are
    well-calibrated for fraud detection. If default already beats nb04 → Optuna justified.

Functions:
    load_lgbm_model()         — load trained LightGBM from disk
    get_top_features()        — extract top-N features by LightGBM importance
    print_feature_breakdown() — show engineered vs V-column vs raw in top-N
    filter_splits()           — filter X_train/val/test to selected features
    train_xgb_on_top_n()      — train XGBoost default params on filtered splits
    evaluate_model()          — predict + print metrics vs V0 TEST baseline
    print_comparison_table()  — final table: all models vs V0 TEST baseline
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score

# Ensure src/ and v0/ are importable regardless of call location.
_SRC_PATH = os.path.dirname(os.path.abspath(__file__))
_V0_PATH  = os.path.join(_SRC_PATH, "..", "v0")
for _p in [_SRC_PATH, _V0_PATH]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ── Known engineered feature names ───────────────────────────────────────────
# WHY explicit set: used only for diagnostic breakdown — not for any model
# decision. Derived from preproc_agg.py, preproc_behavioral.py, preproc_product.py.
ENGINEERED_FEATURES = {
    "tx_count", "tx_amt_mean", "tx_amt_std", "tx_amt_min", "tx_amt_max",
    "tx_amt_ratio", "time_since_last_tx", "delta_amt",
    "nunique_P_email", "is_new_P_email", "nunique_R_email", "is_new_R_email",
    "is_same_email_domain", "nunique_device", "is_new_device",
    "tx_count_last_3d", "tx_count_last_7d", "tx_count_last_30d",
    "amt_vs_personal_median", "amt_z_score", "hour_vs_typical",
    "uid_time_entropy", "is_new_product",
}

# Default XGBoost params for fraud detection on ~100 features.
# WHY colsample_bytree=0.3: with 100 features → ~30 cols/tree.
# 23 engineered features = 23% of 100 → guaranteed exposure at every split.
# WHY scale_pos_weight=28: neg/pos ratio ≈ 96.5/3.5 ≈ 28 — class imbalance.
DEFAULT_XGB_PARAMS = {
    "objective":         "binary:logistic",
    "eval_metric":       "auc",
    "tree_method":       "hist",
    "scale_pos_weight":  28,
    "max_depth":         6,
    "eta":               0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.3,
    "colsample_bylevel": 0.5,
    "min_child_weight":  100,
    "seed":              42,
    "verbosity":         0,
}

# WHY 100: large enough to retain strong V-column signal, small enough
# that 23 engineered features represent ~23% of the selected set.
DEFAULT_TOP_N          = 100
DEFAULT_NUM_ROUNDS     = 3000
DEFAULT_EARLY_STOPPING = 100


# ── Load LightGBM model ───────────────────────────────────────────────────────

def load_lgbm_model(model_path, verbose=True):
    """
    Load trained LightGBM model from disk.

    Parameters
    ----------
    model_path : Path — path to model_lgbm.pkl
    verbose    : bool — print confirmation (default: True)

    Returns
    -------
    LGBMClassifier — trained model
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    if verbose:
        print(f"   Loaded LightGBM <- {model_path}")
    return model


# ── Get top features ──────────────────────────────────────────────────────────

def get_top_features(model_lgbm, feature_names,
                     top_n=DEFAULT_TOP_N,
                     verbose=True):
    """
    Extract top-N features from trained LightGBM by split importance.

    WHY split importance (not gain): split count is more stable across runs
    and less sensitive to a single high-value outlier split.

    Parameters
    ----------
    model_lgbm    : trained LGBMClassifier
    feature_names : list[str] — column names used in LightGBM training
    top_n         : int       — number of features to select (default: 100)
    verbose       : bool      — print ranked list (default: True)

    Returns
    -------
    list[str] — top-N feature names sorted by importance descending
    """
    imp_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": model_lgbm.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    top_features = imp_df.head(top_n)["feature"].tolist()

    if verbose:
        print("=" * 60)
        print(f"TOP-{top_n} FEATURES FROM LIGHTGBM")
        print("=" * 60)
        print(f"   {'Rank':<6} {'Feature':<35} {'Importance':>12}")
        print(f"   {'─' * 55}")
        for i, row in imp_df.head(top_n).iterrows():
            marker = "  ◄ engineered" if row["feature"] in ENGINEERED_FEATURES else ""
            print(f"   {i+1:<6} {row['feature']:<35} {row['importance']:>12.0f}{marker}")

    return top_features


# ── Print feature breakdown ───────────────────────────────────────────────────

def print_feature_breakdown(top_features):
    """
    Show how many engineered vs V-column vs raw features are in top-N.

    Parameters
    ----------
    top_features : list[str] — from get_top_features()

    Returns
    -------
    dict — {"engineered": [...], "v_cols": [...], "raw": [...]}
    """
    engineered = [f for f in top_features if f in ENGINEERED_FEATURES]
    v_cols     = [f for f in top_features if f.startswith("V") and f[1:].isdigit()]
    raw        = [f for f in top_features
                  if f not in ENGINEERED_FEATURES
                  and not (f.startswith("V") and f[1:].isdigit())]

    print("=" * 60)
    print("FEATURE BREAKDOWN IN TOP-N")
    print("=" * 60)
    print(f"   Total selected    : {len(top_features)}")
    print(f"   Engineered (ours) : {len(engineered)}  → {engineered}")
    print(f"   V-columns         : {len(v_cols)}")
    print(f"   Raw features      : {len(raw)}")
    print("=" * 60)

    return {"engineered": engineered, "v_cols": v_cols, "raw": raw}


# ── Filter splits ─────────────────────────────────────────────────────────────

def filter_splits(X_train, X_val, X_test, top_features, verbose=True):
    """
    Filter all three splits to selected top-N features only.

    No-leakage: feature list derived from LightGBM trained on X_train only.
    Applying the same filter to val/test is standard feature selection.

    Parameters
    ----------
    X_train      : pd.DataFrame — full training features (459 cols)
    X_val        : pd.DataFrame — full val features
    X_test       : pd.DataFrame — full frozen TEST features
    top_features : list[str]    — features to keep (from get_top_features)
    verbose      : bool         — print shapes (default: True)

    Returns
    -------
    tuple (X_train_f, X_val_f, X_test_f)
    """
    missing = [f for f in top_features if f not in X_train.columns]
    if missing:
        raise ValueError(f"Features missing from X_train: {missing}")

    X_train_f = X_train[top_features]
    X_val_f   = X_val[top_features]
    X_test_f  = X_test[top_features]

    if verbose:
        print("=" * 60)
        print("FILTERED SPLITS")
        print("=" * 60)
        print(f"   X_train : {X_train.shape} → {X_train_f.shape}")
        print(f"   X_val   : {X_val.shape}   → {X_val_f.shape}")
        print(f"   X_test  : {X_test.shape}  → {X_test_f.shape}  (frozen TEST)")

    return X_train_f, X_val_f, X_test_f


# ── Train XGBoost on top-N features ──────────────────────────────────────────

def train_xgb_on_top_n(X_train, y_train, X_val, y_val,
                        models_dir,
                        params=None,
                        num_rounds=DEFAULT_NUM_ROUNDS,
                        early_stopping_rounds=DEFAULT_EARLY_STOPPING,
                        model_save_name="model_xgb_top_n.pkl",
                        verbose=True):
    """
    Train XGBoost with default params on top-N filtered features.

    WHY default params first: verify feature selection approach improves
    Test PR AUC before investing in Optuna (~8-10h). If default params on
    top-100 features beat nb04 XGBoost (Test PR 0.4944) → Optuna justified.

    Parameters
    ----------
    X_train              : pd.DataFrame — filtered training features
    y_train              : pd.Series    — training target
    X_val                : pd.DataFrame — filtered val features
    y_val                : pd.Series    — validation target
    models_dir           : Path         — directory to save model
    params               : dict | None  — override default params
    num_rounds           : int          — max boosting rounds (default: 3000)
    early_stopping_rounds: int          — patience (default: 100)
    model_save_name      : str          — pkl filename
    verbose              : bool         — print progress (default: True)

    Returns
    -------
    tuple (model, y_pred_val)
        model      — trained xgb.Booster
        y_pred_val — predicted probabilities on val (np.ndarray, float32)
    """
    if verbose:
        print("=" * 60)
        print("TRAINING — XGBoost on Top-N Features (default params)")
        print("=" * 60)

    final_params = DEFAULT_XGB_PARAMS.copy()
    if params:
        final_params.update(params)
        if verbose:
            print(f"   Param overrides: {list(params.keys())}")

    if verbose:
        print(f"\n   Params         : {final_params}")
        print(f"   num_rounds     : {num_rounds}")
        print(f"   early_stopping : {early_stopping_rounds}")
        print(f"\n   Train : {X_train.shape[0]:,} rows × {X_train.shape[1]} features")
        print(f"   Val   : {X_val.shape[0]:,} rows × {X_val.shape[1]} features")

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val,   label=y_val)

    model = xgb.train(
        final_params,
        dtrain,
        num_boost_round=num_rounds,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=early_stopping_rounds,
        evals_result={},
        verbose_eval=100 if verbose else False,
    )

    if verbose:
        print(f"\n   Best iteration : {model.best_iteration}")
        print(f"   Best val AUC   : {model.best_score:.6f}")

    y_pred_val = model.predict(
        dval, iteration_range=(0, model.best_iteration + 1)
    ).astype(np.float32)

    models_dir.mkdir(parents=True, exist_ok=True)
    save_path = models_dir / model_save_name
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    if verbose:
        print(f"\n   Model saved → {save_path}")

    return model, y_pred_val


# ── Evaluate model ────────────────────────────────────────────────────────────

def evaluate_model(y_true, y_pred, model_name,
                   v0_test_roc, v0_test_pr,
                   split_name="Val",
                   show_plot=True,
                   feature_names=None,
                   model=None,
                   top_n=30):
    """
    Print metrics vs baseline and optionally plot curves + importance.

    Supports two baseline modes:
    - v0_test_roc and v0_test_pr both provided (float): show full baseline with deltas
    - v0_test_roc=None, v0_test_pr=float: show statistical baseline PR only (no ROC)

    Parameters
    ----------
    y_true       : pd.Series    — true labels
    y_pred       : np.ndarray   — predicted probabilities
    model_name   : str          — display name
    v0_test_roc  : float | None — baseline Test ROC AUC (None to skip ROC baseline)
    v0_test_pr   : float        — baseline Test PR AUC (e.g. 0.035 for statistical baseline)
    split_name   : str          — 'Val' or 'Test' (default: 'Val')
    show_plot    : bool         — plot ROC/PR curves (default: True)
    feature_names: list[str]    — for feature importance plot (optional)
    model        : xgb.Booster  — for feature importance plot (optional)
    top_n        : int          — top N features to plot (default: 30)

    Returns
    -------
    tuple (roc_auc, pr_auc)
    """
    from evaluate_ml import plot_roc_pr, plot_feature_importance

    roc_auc = roc_auc_score(y_true, y_pred)
    pr_auc  = average_precision_score(y_true, y_pred)

    baseline_label = "Statistical Baseline" if v0_test_roc is None else "Baseline (Test)"
    baseline_roc   = "—" if v0_test_roc is None else f"{v0_test_roc:.4f}"
    baseline_pr    = f"{v0_test_pr:.4f}"

    print(f"{'=' * 60}")
    print(f"  {split_name} RESULTS — {model_name}")
    print(f"{'=' * 60}")
    print(f"  {'Model':<28} {'ROC AUC':>8}  {'PR AUC':>8}")
    print(f"  {'─' * 48}")
    print(f"  {baseline_label:<28} {baseline_roc:>8}  {baseline_pr:>8}")

    delta_parts = []
    if v0_test_roc is not None:
        delta_parts.append(f"Δ ROC={roc_auc - v0_test_roc:+.4f}")
    delta_parts.append(f"Δ PR={pr_auc - v0_test_pr:+.4f}")
    delta_str = "   " + "  ".join(delta_parts)

    print(f"  {model_name:<28} {roc_auc:>8.4f}  {pr_auc:>8.4f}{delta_str}")
    print(f"{'=' * 60}")

    if show_plot:
        plot_roc_pr(y_true, y_pred, model_name=f"{model_name} ({split_name})")
        if model is not None and feature_names is not None:
            plot_feature_importance(
                model,
                feature_names=feature_names,
                top_n=top_n,
                model_name=model_name,
            )

    return roc_auc, pr_auc


# def evaluate_model(y_true, y_pred, model_name,
#                    v0_test_roc, v0_test_pr,
#                    split_name="Val",
#                    show_plot=True,
#                    feature_names=None,
#                    model=None,
#                    top_n=30):
#     """
#     Print metrics vs V0 TEST baseline and optionally plot curves + importance.

#     WHY compare to V0 TEST baseline (not val):
#         Val metrics are influenced by Optuna — they are optimistic.
#         Only TEST provides an unbiased comparison point. V0 TEST is the
#         hard reference every experiment must beat.

#     Parameters
#     ----------
#     y_true       : pd.Series    — true labels
#     y_pred       : np.ndarray   — predicted probabilities
#     model_name   : str          — display name
#     v0_test_roc  : float        — V0 baseline Test ROC AUC (0.8953)
#     v0_test_pr   : float        — V0 baseline Test PR AUC (0.5033)
#     split_name   : str          — 'Val' or 'Test' (default: 'Val')
#     show_plot    : bool         — plot ROC/PR curves (default: True)
#     feature_names: list[str]    — for feature importance plot (optional)
#     model        : xgb.Booster  — for feature importance plot (optional)
#     top_n        : int          — top N features to plot (default: 30)

#     Returns
#     -------
#     tuple (roc_auc, pr_auc)
#     """
#     from evaluate_ml import plot_roc_pr, plot_feature_importance

#     roc_auc = roc_auc_score(y_true, y_pred)
#     pr_auc  = average_precision_score(y_true, y_pred)

#     print(f"{'=' * 60}")
#     print(f"  {split_name} RESULTS — {model_name}")
#     print(f"{'=' * 60}")
#     print(f"  {'Model':<28} {'ROC AUC':>8}  {'PR AUC':>8}")
#     print(f"  {'─' * 48}")
#     print(f"  {'V0 Baseline (Test)':<28} {v0_test_roc:>8.4f}  {v0_test_pr:>8.4f}")
#     print(f"  {model_name:<28} {roc_auc:>8.4f}  {pr_auc:>8.4f}"
#           f"   Δ ROC={roc_auc - v0_test_roc:+.4f}  Δ PR={pr_auc - v0_test_pr:+.4f}")
#     print(f"{'=' * 60}")

#     if show_plot:
#         plot_roc_pr(y_true, y_pred, model_name=f"{model_name} ({split_name})")
#         if model is not None and feature_names is not None:
#             plot_feature_importance(
#                 model,
#                 feature_names=feature_names,
#                 top_n=top_n,
#                 model_name=model_name,
#             )

#     return roc_auc, pr_auc


# ── Final comparison table ────────────────────────────────────────────────────

# def print_comparison_table(results, v0_test_roc, v0_test_pr):
#     """
#     Print final comparison: all models vs V0 TEST baseline.

#     Parameters
#     ----------
#     results     : dict — {model_name: {"test_roc": float, "test_pr": float}}
#                   example: {"nb04 LightGBM": {"test_roc": 0.8963, "test_pr": 0.5045}}
#     v0_test_roc : float — V0 baseline Test ROC AUC
#     v0_test_pr  : float — V0 baseline Test PR AUC
#     """
#     print("=" * 72)
#     print("FINAL COMPARISON — ALL MODELS vs V0 TEST BASELINE")
#     print("=" * 72)
#     print(f"  {'Model':<32} {'Test ROC':>9} {'Test PR':>9} {'Δ ROC':>8} {'Δ PR':>8}")
#     print(f"  {'─' * 68}")
#     print(f"  {'V0 Baseline':<32} {v0_test_roc:>9.4f} {v0_test_pr:>9.4f}"
#           f" {'—':>8} {'—':>8}")

#     for name, metrics in results.items():
#         roc = metrics["test_roc"]
#         pr  = metrics["test_pr"]
#         print(f"  {name:<32} {roc:>9.4f} {pr:>9.4f}"
#               f" {roc - v0_test_roc:>+8.4f} {pr - v0_test_pr:>+8.4f}")

#     print("=" * 72)

def print_comparison_table(results, v0_test_roc, v0_test_pr):
    """
    Print final comparison: all models vs baseline.

    Parameters
    ----------
    results     : dict — {model_name: {"test_roc": float, "test_pr": float}}
                  example: {"nb04 LightGBM": {"test_roc": 0.8963, "test_pr": 0.5045}}
    v0_test_roc : float | None — baseline Test ROC AUC (None to skip)
    v0_test_pr  : float        — baseline Test PR AUC (e.g. 0.035)
    """
    baseline_label = "Statistical Baseline" if v0_test_roc is None else "Baseline"
    baseline_roc   = "—" if v0_test_roc is None else f"{v0_test_roc:.4f}"
    baseline_pr    = f"{v0_test_pr:.4f}"

    print("=" * 72)
    print("FINAL COMPARISON — ALL MODELS vs STATISTICAL BASELINE" if v0_test_roc is None
          else "FINAL COMPARISON — ALL MODELS vs BASELINE")
    print("=" * 72)
    print(f"  {'Model':<32} {'Test ROC':>9} {'Test PR':>9} {'Δ ROC':>8} {'Δ PR':>8}")
    print(f"  {'─' * 68}")
    print(f"  {baseline_label:<32} {baseline_roc:>9} {baseline_pr:>9}"
          f" {'—':>8} {'—':>8}")

    for name, metrics in results.items():
        roc = metrics["test_roc"]
        pr  = metrics["test_pr"]
        delta_roc = f"{roc - v0_test_roc:>+8.4f}" if v0_test_roc is not None else f"{'—':>8}"
        delta_pr  = f"{pr - v0_test_pr:>+8.4f}"
        print(f"  {name:<32} {roc:>9.4f} {pr:>9.4f} {delta_roc} {delta_pr}")

    print("=" * 72)