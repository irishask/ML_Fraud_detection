"""
pipeline_evaluate.py — Training and Evaluation Pipeline for V2
════════════════════════════════════════════════════════════════
Contains all logic for 04_predict_evaluate.ipynb.
The notebook is a pure orchestrator — no logic lives in notebook cells.

Pipeline steps (called in order by the notebook):
    1. load_splits()          — load preprocessed splits from outputs/preproc/
    2. print_flags_summary()  — print RETRAIN flags and param/model file status
    3. train_lgbm_model()     — train LightGBM, save model to outputs/models/
       train_xgb_model()      — train XGBoost,  save model to outputs/models/
    4. evaluate_model()       — print val metrics vs baseline + optional plot
    5. build_ensemble()       — weighted average LightGBM + XGBoost
    6. run_test_evaluation()  — frozen TEST predictions + final comparison table

Model save formats:
    LightGBM → outputs/models/model_lgbm.pkl  (pickle — sklearn API)
    XGBoost  → outputs/models/model_xgb.pkl   (pickle — Booster via pickle)

Functions:
    load_splits()         — load all preprocessed splits and encoding artifacts
    print_flags_summary() — print RETRAIN flags and file status table
    train_lgbm_model()    — train LightGBM, save, return (model, y_pred_val)
    train_xgb_model()     — train XGBoost,  save, return (model, y_pred_val)
    evaluate_model()      — print val metrics vs baseline, optionally plot curves
    build_ensemble()      — weighted average ensemble, return (y_pred, weights)
    run_test_evaluation() — load models, predict TEST, print final comparison table
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure src/ and v0/ are importable from any working directory.
# WHY: pipeline_evaluate.py lives in src/; it imports from src/ and v0/.
# ---------------------------------------------------------------------------
_SRC_PATH = os.path.dirname(os.path.abspath(__file__))
_V0_PATH  = os.path.join(_SRC_PATH, "..", "v0")
for _p in [_SRC_PATH, _V0_PATH]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from train_lightgbm import train_lgbm
from train_xgboost  import train_xgb
from config         import NON_FEATURE_COLS


# ── Step 1: Load preprocessed splits ─────────────────────────────────────────

def load_splits(preproc_dir, verbose=True):
    """
    Load all preprocessed splits saved by 03_preprocess_train_clean.ipynb.

    Parameters
    ----------
    preproc_dir : Path — directory containing preprocessed artifacts
                  (outputs/preproc/)
    verbose     : bool — print shapes and fraud rates (default: True)

    Returns
    -------
    tuple (X_train_lgbm, X_val_lgbm, X_test_lgbm,
           encoding_map, y_train, y_val, y_test)
        X_train_lgbm : pd.DataFrame — label-encoded train features (LightGBM / XGBoost)
        X_val_lgbm   : pd.DataFrame — label-encoded val features   (early stopping)
        X_test_lgbm  : pd.DataFrame — label-encoded frozen TEST features
        encoding_map : dict         — fitted label encoders
        y_train      : pd.Series    — training target
        y_val        : pd.Series    — validation target
        y_test       : pd.Series    — frozen TEST target (touched once at final eval)
    """
    if verbose:
        print("=" * 60)
        print("STEP 1 — Load preprocessed splits")
        print("=" * 60)

    X_train_lgbm = pd.read_parquet(preproc_dir / "X_train_lgbm.parquet")
    X_val_lgbm   = pd.read_parquet(preproc_dir / "X_val_lgbm.parquet")
    X_test_lgbm  = pd.read_parquet(preproc_dir / "X_test_lgbm.parquet")
    y_train      = pd.read_parquet(preproc_dir / "y_train.parquet")["isFraud"]
    y_val        = pd.read_parquet(preproc_dir / "y_val.parquet")["isFraud"]
    y_test       = pd.read_parquet(preproc_dir / "y_test.parquet")["isFraud"]

    with open(preproc_dir / "encoding_map.pkl", "rb") as f:
        encoding_map = pickle.load(f)

    if verbose:
        print(f"   X_train_lgbm : {X_train_lgbm.shape}  | fraud rate: {y_train.mean():.4%}")
        print(f"   X_val_lgbm   : {X_val_lgbm.shape}    | fraud rate: {y_val.mean():.4%}  (early stopping)")
        print(f"   X_test_lgbm  : {X_test_lgbm.shape}   | fraud rate: {y_test.mean():.4%}  (frozen TEST)")
        print(f"   encoding_map : {len(encoding_map)} encoders")

    return (X_train_lgbm, X_val_lgbm, X_test_lgbm,
            encoding_map, y_train, y_val, y_test)


# ── Step 2a: Train LightGBM ───────────────────────────────────────────────────

def train_lgbm_model(X_train_lgbm, y_train, X_val_lgbm, y_val,
                     models_dir,
                     params_path="best_params_lgbm.json",
                     verbose=True):
    """
    Train LightGBM on preprocessed splits and save the model to disk.

    WHY save as pickle: LightGBMClassifier uses sklearn API — pickle is the
    standard serialization format. Loading is straightforward:
        with open(path, 'rb') as f: model = pickle.load(f)

    Parameters
    ----------
    X_train_lgbm : pd.DataFrame — label-encoded training features
    y_train      : pd.Series    — training target
    X_val_lgbm   : pd.DataFrame — label-encoded validation features
    y_val        : pd.Series    — validation target
    models_dir   : Path         — directory to save model_lgbm.pkl
    params_path  : str          — path to best_params_lgbm.json
    verbose      : bool         — print progress (default: True)

    Returns
    -------
    tuple (model, y_pred_val)
        model      — trained LGBMClassifier
        y_pred_val — predicted probabilities on val set (np.ndarray, float32)
    """
    if verbose:
        print("=" * 60)
        print("STEP 2 — Train LightGBM")
        print("=" * 60)

    model, y_pred_val = train_lgbm(
        X_train_lgbm, y_train,
        X_val_lgbm,   y_val,
        params_path=params_path,
        verbose=verbose,
    )

    models_dir.mkdir(parents=True, exist_ok=True)
    save_path = models_dir / "model_lgbm.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(model, f)

    if verbose:
        print(f"\n   Model saved → {save_path}")

    return model, y_pred_val


# ── Step 2b: Train XGBoost ────────────────────────────────────────────────────

def train_xgb_model(X_train_lgbm, y_train, X_val_lgbm, y_val,
                    models_dir,
                    params_path="best_params_xgb.json",
                    verbose=True):
    """
    Train XGBoost on preprocessed splits and save the model to disk.

    WHY same X_train_lgbm as LightGBM:
        XGBoost and LightGBM both require label-encoded numeric features.
        preproc_lgbm_xgboost.py was designed for both models — no separate
        preprocessing needed.

    WHY save as pickle: xgb.Booster does not have a sklearn-compatible
    save_model() that preserves best_iteration context. Pickle preserves
    the full Booster object including best_iteration for correct prediction.

    Parameters
    ----------
    X_train_lgbm : pd.DataFrame — label-encoded training features (same as LightGBM)
    y_train      : pd.Series    — training target
    X_val_lgbm   : pd.DataFrame — label-encoded validation features
    y_val        : pd.Series    — validation target
    models_dir   : Path         — directory to save model_xgb.pkl
    params_path  : str          — path to best_params_xgb.json
    verbose      : bool         — print progress (default: True)

    Returns
    -------
    tuple (model, y_pred_val)
        model      — trained xgb.Booster
        y_pred_val — predicted probabilities on val set (np.ndarray, float32)
    """
    if verbose:
        print("=" * 60)
        print("STEP 2 — Train XGBoost")
        print("=" * 60)

    model, y_pred_val = train_xgb(
        X_train_lgbm, y_train,
        X_val_lgbm,   y_val,
        params_path=params_path,
        verbose=verbose,
    )

    models_dir.mkdir(parents=True, exist_ok=True)
    save_path = models_dir / "model_xgb.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(model, f)

    if verbose:
        print(f"\n   Model saved → {save_path}")

    return model, y_pred_val

# ── Print flags summary ───────────────────────────────────────────────────────

def print_flags_summary(retrain_lgbm, retrain_xgb, param_files, model_files):
    """
    Print a table showing RETRAIN flags and the status of param/model files.

    WHY separate function: keeps nb04 Step 1 a single call — no inline logic
    in the notebook. Makes it easy to audit what will run before execution.

    Parameters
    ----------
    retrain_lgbm : bool — True → always retrain LightGBM
    retrain_xgb  : bool — True → always retrain XGBoost
    param_files  : dict[str, Path] — model key → path to best_params JSON
    model_files  : dict[str, Path] — model key → path to saved model pkl
    """
    retrain_map = {"lgbm": retrain_lgbm, "xgb": retrain_xgb}

    print("=" * 60)
    print("EXECUTION FLAGS")
    print("=" * 60)
    print(f"  {'Model':<8} {'JSON params':^14} {'Saved model':^14} {'Action'}")
    print(f"  {'─' * 56}")

    for key in ["lgbm", "xgb"]:
        json_ok  = param_files[key].exists() if key in param_files else False
        model_ok = model_files[key].exists()
        retrain  = retrain_map[key]

        if retrain:
            action = "retrain (flag=True)"
        elif model_ok:
            action = "load saved model"
        else:
            action = "train (no saved model)"

        print(f"  {key:<8} {str(json_ok):^14} {str(model_ok):^14} {action}")

    print("=" * 60)


# ── Evaluate model on val ─────────────────────────────────────────────────────

def evaluate_model(y_val, y_pred, model_name,
                   v0_roc, v0_pr,
                   feature_names=None,
                   model=None,
                   show_plot=True,
                   top_n=30):
    """
    Print metrics vs baseline and optionally plot ROC/PR curves
    and feature importance.

    Supports two baseline modes:
    - v0_roc and v0_pr both provided (float): show baseline ROC + PR with deltas
    - v0_roc=None, v0_pr=float: show statistical baseline PR only (no ROC)

    WHY show_plot=True default: visual inspection of curves is the primary
    way to catch overfitting or unexpected behaviour during development.
    Set False for automated runs or when running all models in sequence.

    Parameters
    ----------
    y_val         : pd.Series    — true validation labels
    y_pred        : np.ndarray   — predicted probabilities on val set
    model_name    : str          — display name (e.g. 'LightGBM', 'XGBoost')
    v0_roc        : float | None — baseline ROC AUC (None to skip ROC baseline)
    v0_pr         : float        — baseline PR AUC (e.g. 0.035 for statistical baseline)
    feature_names : list[str]    — column names for feature importance plot
                    WHY None default: only required when show_plot=True and
                    model is provided; safe to omit for prediction-only calls
    model         : fitted model — LGBMClassifier or xgb.Booster
                    WHY None default: feature importance plot is optional;
                    passing None skips it cleanly
    show_plot     : bool         — plot ROC/PR curves + feature importance
                    (default: True)
    top_n         : int          — top N features for importance plot (default: 30)

    Returns
    -------
    tuple (roc_auc, pr_auc)
        roc_auc : float — Val ROC AUC
        pr_auc  : float — Val PR AUC
    """
    from sklearn.metrics import roc_auc_score, average_precision_score
    from evaluate_ml import plot_roc_pr, plot_feature_importance

    roc_auc = roc_auc_score(y_val, y_pred)
    pr_auc  = average_precision_score(y_val, y_pred)

    baseline_label = "statistical baseline" if v0_roc is None else "baseline"
    baseline_roc   = "—" if v0_roc is None else f"{v0_roc:.4f}"
    baseline_pr    = f"{v0_pr:.4f}"

    print(f"{'=' * 52}")
    print(f"  {'Model':<20} {'ROC AUC':>8}  {'PR AUC':>8}")
    print(f"{'=' * 52}")
    print(f"  {baseline_label:<20} {baseline_roc:>8}  {baseline_pr:>8}")

    delta_parts = []
    if v0_roc is not None:
        delta_parts.append(f"Δ ROC={roc_auc - v0_roc:+.4f}")
    delta_parts.append(f"Δ PR={pr_auc - v0_pr:+.4f}")
    delta_str = "   " + "  ".join(delta_parts)

    print(f"  {model_name:<20} {roc_auc:>8.4f}  {pr_auc:>8.4f}{delta_str}")
    print(f"{'=' * 52}")

    if show_plot:
        plot_roc_pr(y_val, y_pred, model_name=model_name)
        if model is not None and feature_names is not None:
            plot_feature_importance(
                model,
                feature_names=feature_names,
                top_n=top_n,
                model_name=model_name,
            )

    return roc_auc, pr_auc















# ── Build ensemble ────────────────────────────────────────────────────────────

def build_ensemble(y_val, y_pred_lgbm, y_pred_xgb, verbose=True):
    """
    Build weighted average ensemble from LightGBM and XGBoost val predictions.

    Weights are proportional to each model's Val ROC AUC:
        weight_i = AUC_i / (AUC_lgbm + AUC_xgb)

    WHY ROC AUC for weights (not PR AUC): weights are used to combine
    probability scores — ROC AUC measures overall ranking quality across
    all thresholds, which is the correct criterion for weighting probability
    outputs. PR AUC is threshold-sensitive and less stable as a weight.

    Parameters
    ----------
    y_val        : pd.Series  — true validation labels
    y_pred_lgbm  : np.ndarray — LightGBM val predicted probabilities
    y_pred_xgb   : np.ndarray — XGBoost val predicted probabilities
    verbose      : bool       — print weights and ensemble metrics (default: True)

    Returns
    -------
    tuple (y_pred_ensemble, weights)
        y_pred_ensemble : np.ndarray (float32) — ensemble predicted probabilities
        weights         : dict[str, float]     — {'lgbm': w1, 'xgb': w2}
    """
    from sklearn.metrics import roc_auc_score, average_precision_score
    from train_ensemble import compute_weights, weighted_average

    scores  = {
        "lgbm": roc_auc_score(y_val, y_pred_lgbm),
        "xgb":  roc_auc_score(y_val, y_pred_xgb),
    }
    weights         = compute_weights(scores)
    y_pred_ensemble = weighted_average(
        {"lgbm": y_pred_lgbm, "xgb": y_pred_xgb},
        weights,
    )

    if verbose:
        ens_roc = roc_auc_score(y_val, y_pred_ensemble)
        ens_pr  = average_precision_score(y_val, y_pred_ensemble)
        print("=" * 60)
        print("ENSEMBLE — LightGBM + XGBoost (weighted average)")
        print("=" * 60)
        print(f"   Weights : lgbm={weights['lgbm']:.4f}  xgb={weights['xgb']:.4f}")
        print(f"   Val ROC AUC : {ens_roc:.4f}")
        print(f"   Val PR AUC  : {ens_pr:.4f}")
        print("=" * 60)

    return y_pred_ensemble, weights


# ── Frozen TEST evaluation ────────────────────────────────────────────────────

# def run_test_evaluation(model_files, X_test_lgbm, y_test,
#                         y_val, y_pred_lgbm, y_pred_xgb, weights,
#                         v0_val_roc, v0_val_pr,
#                         v0_test_roc, v0_test_pr,
#                         verbose=True):
#     """
#     Load saved models, predict on frozen TEST, print final comparison table.

#     WHY load from disk (not reuse in-memory models): guarantees that the exact
#     saved artifact is evaluated — not an in-memory object that may differ from
#     what was pickled. This matches production: the model file IS the deliverable.

#     WHY weights from val (not recomputed on test): weights must be determined
#     before touching TEST — recomputing on test would use test information to
#     select the ensemble composition, which is leakage.

#     IMPORTANT: this function touches the frozen TEST set. Call it only once,
#     after all model selection and tuning decisions are final.

#     Parameters
#     ----------
#     model_files  : dict[str, Path]  — {'lgbm': Path, 'xgb': Path}
#     X_test_lgbm  : pd.DataFrame     — frozen TEST features
#     y_test       : pd.Series        — frozen TEST labels
#     y_val        : pd.Series        — val labels
#     y_pred_lgbm  : np.ndarray       — LightGBM val predictions
#     y_pred_xgb   : np.ndarray       — XGBoost val predictions
#     weights      : dict[str, float] — ensemble weights from build_ensemble()
#                    WHY passed in: must be the same weights used on val —
#                    recomputing on test would leak test signal into model selection
#     v0_val_roc   : float            — V0 baseline Val ROC AUC
#     v0_val_pr    : float            — V0 baseline Val PR AUC
#     v0_test_roc  : float            — V0 baseline Test ROC AUC
#     v0_test_pr   : float            — V0 baseline Test PR AUC
#     verbose      : bool             — print table (default: True)

#     Returns
#     -------
#     pd.DataFrame — final comparison table (all models + ensemble vs V0 baseline)
#     """
#     import xgboost as xgb_lib
#     from sklearn.metrics import roc_auc_score, average_precision_score
#     from train_ensemble import weighted_average

#     # ── Load models from disk and predict on frozen TEST ──────────────────────
#     with open(model_files["lgbm"], "rb") as f:
#         model_lgbm = pickle.load(f)
#     y_pred_lgbm_test = model_lgbm.predict_proba(X_test_lgbm)[:, 1].astype("float32")

#     with open(model_files["xgb"], "rb") as f:
#         model_xgb = pickle.load(f)
#     dtest           = xgb_lib.DMatrix(X_test_lgbm)
#     y_pred_xgb_test = model_xgb.predict(
#         dtest, iteration_range=(0, model_xgb.best_iteration + 1)
#     ).astype("float32")

#     # ── Ensemble TEST — use weights determined on val (no leakage) ────────────
#     y_pred_ensemble_test = weighted_average(
#         {"lgbm": y_pred_lgbm_test, "xgb": y_pred_xgb_test},
#         weights,
#     )

#     # ── Build final comparison table ──────────────────────────────────────────
#     val_preds  = {"lgbm": y_pred_lgbm,      "xgb": y_pred_xgb,      "ensemble": None}
#     test_preds = {"lgbm": y_pred_lgbm_test, "xgb": y_pred_xgb_test, "ensemble": y_pred_ensemble_test}

#     model_labels = {
#         "lgbm":     "LightGBM Optuna",
#         "xgb":      "XGBoost Optuna",
#         "ensemble": "Ensemble (LGBM+XGB)",
#     }

#     rows = [{
#         "Model":        "V0 Baseline",
#         "Val ROC AUC":  v0_val_roc,
#         "Val PR AUC":   v0_val_pr,
#         "Test ROC AUC": v0_test_roc,
#         "Test PR AUC":  v0_test_pr,
#         "Δ Test ROC":   "+0.0000",
#         "Δ Test PR":    "+0.0000",
#     }]

#     for key, label in model_labels.items():
#         vp      = val_preds[key]
#         tp      = test_preds[key]
#         val_roc = roc_auc_score(y_val, vp)        if vp is not None else None
#         val_pr  = average_precision_score(y_val, vp) if vp is not None else None
#         tst_roc = roc_auc_score(y_test, tp)
#         tst_pr  = average_precision_score(y_test, tp)
#         rows.append({
#             "Model":        label,
#             "Val ROC AUC":  val_roc,
#             "Val PR AUC":   val_pr,
#             "Test ROC AUC": tst_roc,
#             "Test PR AUC":  tst_pr,
#             "Δ Test ROC":   f"{tst_roc - v0_test_roc:+.4f}",
#             "Δ Test PR":    f"{tst_pr  - v0_test_pr:+.4f}",
#         })

#     comparison = pd.DataFrame(rows)

#     if verbose:
#         print("=" * 75)
#         print("FINAL RESULTS — VAL + FROZEN TEST")
#         print("=" * 75)
#         print(f"  {'Model':<22} {'Val ROC':>8} {'Val PR':>8} "
#               f"{'Test ROC':>9} {'Test PR':>8} {'Δ Test ROC':>11} {'Δ Test PR':>10}")
#         print(f"  {'─' * 71}")
#         for _, row in comparison.iterrows():
#             val_roc = f"{row['Val ROC AUC']:.4f}" if row["Val ROC AUC"] is not None else "   —   "
#             val_pr  = f"{row['Val PR AUC']:.4f}"  if row["Val PR AUC"]  is not None else "   —   "
#             print(
#                 f"  {row['Model']:<22} {val_roc:>8} {val_pr:>8} "
#                 f"{row['Test ROC AUC']:>9.4f} {row['Test PR AUC']:>8.4f} "
#                 f"{row['Δ Test ROC']:>11} {row['Δ Test PR']:>10}"
#             )
#         print("=" * 75)

#     return comparison


# ── Frozen TEST evaluation ────────────────────────────────────────────────────

def run_test_evaluation(model_files, X_test_lgbm, y_test,
                        y_val, y_pred_lgbm, y_pred_xgb, weights,
                        v0_val_roc, v0_val_pr,
                        v0_test_roc, v0_test_pr,
                        verbose=True):
    """
    Load saved models, predict on frozen TEST, print final comparison table.

    Supports two baseline modes:
    - v0_*_roc and v0_*_pr all provided (float): show full baseline row with deltas
    - v0_*_roc=None: show statistical baseline PR only, deltas computed vs v0_*_pr

    WHY load from disk (not reuse in-memory models): guarantees that the exact
    saved artifact is evaluated — not an in-memory object that may differ from
    what was pickled. This matches production: the model file IS the deliverable.

    WHY weights from val (not recomputed on test): weights must be determined
    before touching TEST — recomputing on test would use test information to
    select the ensemble composition, which is leakage.

    IMPORTANT: this function touches the frozen TEST set. Call it only once,
    after all model selection and tuning decisions are final.

    Parameters
    ----------
    model_files  : dict[str, Path]  — {'lgbm': Path, 'xgb': Path}
    X_test_lgbm  : pd.DataFrame     — frozen TEST features
    y_test       : pd.Series        — frozen TEST labels
    y_val        : pd.Series        — val labels
    y_pred_lgbm  : np.ndarray       — LightGBM val predictions
    y_pred_xgb   : np.ndarray       — XGBoost val predictions
    weights      : dict[str, float] — ensemble weights from build_ensemble()
                   WHY passed in: must be the same weights used on val —
                   recomputing on test would leak test signal into model selection
    v0_val_roc   : float | None     — baseline Val ROC AUC (None to skip)
    v0_val_pr    : float            — baseline Val PR AUC (e.g. 0.035)
    v0_test_roc  : float | None     — baseline Test ROC AUC (None to skip)
    v0_test_pr   : float            — baseline Test PR AUC (e.g. 0.035)
    verbose      : bool             — print table (default: True)

    Returns
    -------
    pd.DataFrame — final comparison table (all models + ensemble vs baseline)
    """
    import xgboost as xgb_lib
    from sklearn.metrics import roc_auc_score, average_precision_score
    from train_ensemble import weighted_average

    # ── Load models from disk and predict on frozen TEST ──────────────────────
    with open(model_files["lgbm"], "rb") as f:
        model_lgbm = pickle.load(f)
    y_pred_lgbm_test = model_lgbm.predict_proba(X_test_lgbm)[:, 1].astype("float32")

    with open(model_files["xgb"], "rb") as f:
        model_xgb = pickle.load(f)
    dtest           = xgb_lib.DMatrix(X_test_lgbm)
    y_pred_xgb_test = model_xgb.predict(
        dtest, iteration_range=(0, model_xgb.best_iteration + 1)
    ).astype("float32")

    # ── Ensemble TEST — use weights determined on val (no leakage) ────────────
    y_pred_ensemble_test = weighted_average(
        {"lgbm": y_pred_lgbm_test, "xgb": y_pred_xgb_test},
        weights,
    )

    # ── Build final comparison table ──────────────────────────────────────────
    val_preds  = {"lgbm": y_pred_lgbm,      "xgb": y_pred_xgb,      "ensemble": None}
    test_preds = {"lgbm": y_pred_lgbm_test, "xgb": y_pred_xgb_test, "ensemble": y_pred_ensemble_test}

    model_labels = {
        "lgbm":     "LightGBM Optuna",
        "xgb":      "XGBoost Optuna",
        "ensemble": "Ensemble (LGBM+XGB)",
    }

    baseline_label = "Statistical Baseline" if v0_test_roc is None else "Baseline"
    rows = [{
        "Model":        baseline_label,
        "Val ROC AUC":  v0_val_roc,
        "Val PR AUC":   v0_val_pr,
        "Test ROC AUC": v0_test_roc,
        "Test PR AUC":  v0_test_pr,
        "Δ Test ROC":   "—",
        "Δ Test PR":    "—",
    }]

    for key, label in model_labels.items():
        vp      = val_preds[key]
        tp      = test_preds[key]
        val_roc = roc_auc_score(y_val, vp)        if vp is not None else None
        val_pr  = average_precision_score(y_val, vp) if vp is not None else None
        tst_roc = roc_auc_score(y_test, tp)
        tst_pr  = average_precision_score(y_test, tp)

        delta_roc = f"{tst_roc - v0_test_roc:+.4f}" if v0_test_roc is not None else "—"
        delta_pr  = f"{tst_pr  - v0_test_pr:+.4f}"

        rows.append({
            "Model":        label,
            "Val ROC AUC":  val_roc,
            "Val PR AUC":   val_pr,
            "Test ROC AUC": tst_roc,
            "Test PR AUC":  tst_pr,
            "Δ Test ROC":   delta_roc,
            "Δ Test PR":    delta_pr,
        })

    comparison = pd.DataFrame(rows)

    if verbose:
        print("=" * 75)
        print("FINAL RESULTS — VAL + FROZEN TEST")
        print("=" * 75)
        print(f"  {'Model':<22} {'Val ROC':>8} {'Val PR':>8} "
              f"{'Test ROC':>9} {'Test PR':>8} {'Δ Test ROC':>11} {'Δ Test PR':>10}")
        print(f"  {'─' * 71}")
        for _, row in comparison.iterrows():
            val_roc  = f"{row['Val ROC AUC']:.4f}" if row["Val ROC AUC"]  is not None else "   —   "
            val_pr   = f"{row['Val PR AUC']:.4f}"  if row["Val PR AUC"]   is not None else "   —   "
            test_roc = f"{row['Test ROC AUC']:.4f}" if row["Test ROC AUC"] is not None else "   —   "
            test_pr  = f"{row['Test PR AUC']:.4f}"  if row["Test PR AUC"]  is not None else "   —   "
            print(
                f"  {row['Model']:<22} {val_roc:>8} {val_pr:>8} "
                f"{test_roc:>9} {test_pr:>8} "
                f"{row['Δ Test ROC']:>11} {row['Δ Test PR']:>10}"
            )
        print("=" * 75)

    return comparison