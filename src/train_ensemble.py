"""
train_ensemble.py — Weighted Average Ensemble
══════════════════════════════════════════════
Combines predictions from LightGBM and XGBoost using
weighted average, where weights are proportional to each model's ROC AUC.

WHY weighted average over simple average:
    A stronger model should contribute more to the final prediction.
    Weighting by ROC AUC automatically gives more influence to the model
    that best separates fraud from legitimate transactions.

WHY weighted average over stacking:
    Stacking requires out-of-fold predictions to avoid leakage — significantly
    more complex to implement correctly. Weighted average is simpler, faster,
    and produces strong results as a first ensemble step.
    Stacking can be added in V3 as an experiment.

WHY ensemble works:
    Each model makes different errors on the same data:
      LightGBM  — leaf-wise growth, misses some level-wise patterns
      XGBoost   — level-wise growth, misses some leaf-wise patterns
    Weighted average compensates individual model blind spots.

Functions:
    compute_weights()    — compute AUC-proportional weights from scores dict
    weighted_average()   — combine predictions using weights
    evaluate_ensemble()  — compare all models + ensemble in one table
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


# ── Weights ───────────────────────────────────────────────────────────────────

def compute_weights(scores):
    """
    Compute ensemble weights proportional to each model's ROC AUC.

    Formula:
        weight_i = AUC_i / sum(AUC_j for all j)

    WHY proportional to AUC:
        A model with higher AUC better separates fraud from legitimate
        transactions — it deserves more influence in the final prediction.
        Weights sum to 1.0 — guarantees ensemble output stays in [0, 1].

    WHY not equal weights:
        Equal weights assume all models are equally good. In practice models
        differ — especially before Optuna tuning. AUC-proportional weights
        automatically adapt to actual model quality.

    Example (with real numbers):
        scores = {"lgbm": 0.9235, "xgb": 0.9280}
        total  = 0.9235 + 0.9280 = 1.8515
        weights = {
            "lgbm": 0.9235 / 1.8515 = 0.4988,
            "xgb":  0.9280 / 1.8515 = 0.5012,
        }
        sum(weights.values()) == 1.0  ← always guaranteed

    Parameters
    ----------
    scores : dict[str, float] — model name → ROC AUC on val set
             example: {"lgbm": 0.9235, "xgb": 0.9280}

    Returns
    -------
    dict[str, float] — model name → weight (values sum to 1.0)
    """
  
    total = sum(scores.values())
    weights = {name: auc / total for name, auc in scores.items()}
    return weights


# ── Weighted Average ──────────────────────────────────────────────────────────

def weighted_average(predictions, weights):
    """
    Combine model predictions into a single ensemble prediction.

    Formula:
        y_ensemble_i = sum(weight_j * y_pred_j_i for all models j)

    WHY this formula:
        Each transaction gets a weighted vote from all models.
        If LightGBM misses a fraud (predicts 0.3) but XGBoost
        catches it (predicts 0.75), the ensemble raises the probability
        above the threshold — compensating LightGBM's blind spot.

    Parameters
    ----------
    predictions : dict[str, np.ndarray] — model name → predicted probabilities
                  example: {"lgbm": y_pred_lgbm, "xgb": y_pred_xgb}
                  all arrays must have the same length
    weights     : dict[str, float]      — model name → weight from compute_weights()
                  WHY separate parameter: allows caller to inspect or override
                  weights before passing them in

    Returns
    -------
    np.ndarray (float32) — ensemble predicted probabilities, shape (n_samples,)
    """
    # Validate that all models in weights have predictions
    missing = set(weights.keys()) - set(predictions.keys())
    if missing:
        raise ValueError(
            f"Models in weights but missing from predictions: {missing}. "
            f"Available predictions: {list(predictions.keys())}"
        )

    # Validate all prediction arrays have the same length
    lengths = {name: len(pred) for name, pred in predictions.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(
            f"Prediction arrays have different lengths: {lengths}. "
            f"All models must predict on the same validation set."
        )

    # Weighted sum across all models
    ensemble = np.zeros(len(next(iter(predictions.values()))), dtype=np.float64)
    for name, weight in weights.items():
        ensemble += weight * predictions[name].astype(np.float64)

    return ensemble.astype(np.float32)


# ── Evaluate ──────────────────────────────────────────────────────────────────

def evaluate_ensemble(y_val, predictions, weights=None):
    """
    Compare all individual models and ensemble in a single summary table.

    Computes ROC AUC and PR AUC for each model and for the ensemble.
    Adds delta columns showing improvement over the best single model.

    WHY both ROC AUC and PR AUC:
        ROC AUC measures overall discrimination ability.
        PR AUC measures precision-recall tradeoff — more informative for
        imbalanced datasets (3.5% fraud rate) where ROC AUC can look
        inflated even for mediocre models.
        Both metrics together give a complete picture.

    Parameters
    ----------
    y_val       : pd.Series or np.ndarray — true labels on validation set
    predictions : dict[str, np.ndarray]   — model name → predicted probabilities
                  must include all individual models AND "ensemble" key
                  example: {"lgbm": ..., "xgb": ..., "ensemble": ...}
    weights     : dict[str, float] | None — weights used for ensemble
                  if provided, printed in the summary for reference

    Returns
    -------
    pd.DataFrame — columns: model | ROC AUC | PR AUC | ΔROC AUC | ΔPR AUC
                   sorted by ROC AUC descending
    """
    rows = []
    for name, y_pred in predictions.items():
        roc_auc = roc_auc_score(y_val, y_pred)
        pr_auc  = average_precision_score(y_val, y_pred)
        rows.append({"model": name, "ROC AUC": roc_auc, "PR AUC": pr_auc})

    results = pd.DataFrame(rows).sort_values("ROC AUC", ascending=False).reset_index(drop=True)

    # Compute delta vs best single model (excluding ensemble from baseline)
    single_models = results[results["model"] != "ensemble"]
    best_roc = single_models["ROC AUC"].max()
    best_pr  = single_models["PR AUC"].max()

    results["ΔROC AUC"] = (results["ROC AUC"] - best_roc).map(lambda x: f"{x:+.4f}")
    results["ΔPR AUC"]  = (results["PR AUC"]  - best_pr ).map(lambda x: f"{x:+.4f}")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("ENSEMBLE EVALUATION")
    print("=" * 60)

    if weights:
        print("\n   Weights:")
        for name, w in weights.items():
            print(f"     {name:10s}: {w:.4f}")

    print("\n   Results (sorted by ROC AUC):")
    print(f"\n{'─' * 60}")
    header = f"   {'Model':<12} {'ROC AUC':>10} {'PR AUC':>10} {'ΔROC AUC':>12} {'ΔPR AUC':>10}"
    print(header)
    print(f"{'─' * 60}")

    for _, row in results.iterrows():
        marker = " ◄" if row["model"] == "ensemble" else ""
        print(
            f"   {row['model']:<12} "
            f"{row['ROC AUC']:>10.4f} "
            f"{row['PR AUC']:>10.4f} "
            f"{row['ΔROC AUC']:>12} "
            f"{row['ΔPR AUC']:>10}"
            f"{marker}"
        )

    print(f"{'─' * 60}")

    # Check correlation between model predictions — low correlation = good diversity
    print("\n   Prediction correlations (lower = more diverse = better ensemble):")
    model_names = [n for n in predictions if n != "ensemble"]
    for i, n1 in enumerate(model_names):
        for n2 in model_names[i+1:]:
            corr = np.corrcoef(predictions[n1], predictions[n2])[0, 1]
            diversity = "good" if corr < 0.95 else "WARNING: low diversity"
            print(f"     {n1} vs {n2}: {corr:.4f}  ({diversity})")

    print("=" * 60)

    return results