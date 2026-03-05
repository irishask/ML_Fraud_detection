"""
evaluate_ml.py — Evaluation metrics and plots for all model versions
════════════════════════════════════════════════════════════════════
Functions:
    compute_metrics()         — ROC AUC, PR AUC, classification report
    plot_roc_pr()             — side-by-side ROC and PR curves
    plot_feature_importance() — horizontal bar chart of top features
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    classification_report,
)


def compute_metrics(y_true, y_prob, threshold=0.5, verbose=True):
    """
    Compute fraud detection metrics: ROC AUC, PR AUC, classification report.

    Parameters
    ----------
    y_true    : array-like — actual labels (0/1)
    y_prob    : array-like — predicted probabilities
    threshold : float — cutoff for converting probabilities to binary predictions.
                Default=0.5: standard neutral starting point for initial evaluation.
                NOTE: 0.5 is NOT the recommended operational threshold for fraud
                detection — with 3.5% class imbalance, optimal threshold is typically
                much lower (0.1–0.3). Use threshold analysis to find the best value
                for your precision/recall trade-off before deployment.
    verbose   : bool — print results

    Returns
    -------
    dict with keys: roc_auc, pr_auc, threshold, report
    """
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    # Binary predictions at threshold
    y_pred = (y_prob >= threshold).astype(int)
    report = classification_report(y_true, y_pred, output_dict=True)

    if verbose:
        print("=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"   ROC AUC:  {roc_auc:.6f}")
        print(f"   PR AUC:   {pr_auc:.6f}")
        print(f"   Threshold: {threshold}")
        print(f"\n   Classification Report (threshold={threshold}):")
        print(classification_report(y_true, y_pred))

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold": threshold,
        "report": report,
    }


def plot_roc_pr(y_true, y_prob, model_name="LightGBM v0",
                figsize=(14, 5), dpi=150, save_path=None):
    """
    Plot ROC and Precision-Recall curves side by side.

    Parameters
    ----------
    y_true     : array-like — actual labels (0/1)
    y_prob     : array-like — predicted probabilities
    model_name : str — label for the legend
    figsize    : tuple — figure size in inches (default=(14, 5)).
                 Wide aspect ratio fits two side-by-side plots comfortably
    dpi        : int — resolution for saved figures (default=150).
                 150 dpi: sharp enough for reports, not excessively large on disk
    save_path  : str or None — path to save figure (None = don't save)
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- Left: ROC Curve ---
    ax = axes[0]
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    ax.plot(fpr, tpr, color="#c4b7d4", linewidth=2,
            label=f"{model_name} (AUC={roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="red", linestyle="--", alpha=0.5,
            label="Random (AUC=0.5)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    # --- Right: Precision-Recall Curve ---
    ax = axes[1]
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    fraud_rate = np.mean(y_true)

    ax.plot(rec, prec, color="#b3cde3", linewidth=2,
            label=f"{model_name} (AP={pr_auc:.4f})")
    ax.axhline(y=fraud_rate, color="red", linestyle="--", alpha=0.5,
               label=f"Random ({fraud_rate:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    plt.suptitle(f"Model Evaluation: {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"   Saved plot to {save_path}")

    plt.show()


def plot_feature_importance(model, feature_names, top_n=30,
                            model_name="LightGBM v0",
                            figsize_width=10, figsize_height_per_feature=0.3,
                            figsize_min_height=6, dpi=150, save_path=None):
    """
    Plot horizontal bar chart of top-N most important features.

    Parameters
    ----------
    model                    : fitted model — LightGBM (.feature_importances_)
                               or XGBoost Booster (.get_score()) — both supported
    feature_names            : list of str — column names used in training
    top_n                    : int — number of top features to display (default=30).
                               30 gives a readable chart without information overload
    model_name               : str — title label
    figsize_width            : int — figure width in inches (default=10)
    figsize_height_per_feature: float — height per feature row (default=0.3).
                               Scales figure height dynamically with top_n so labels
                               never overlap regardless of how many features are shown
    figsize_min_height       : int — minimum figure height in inches (default=6).
                               Prevents tiny charts when top_n is small
    dpi                      : int — resolution for saved figures (default=150)
    save_path                : str or None — path to save figure (None = don't save)

    Returns
    -------
    pd.DataFrame — full feature importance table (all features, sorted descending)
    """
    # Build importance DataFrame
    # WHY three-way check: each model exposes feature importance via a different API.
    #   LightGBM : model.feature_importances_  — sklearn API, returns np.ndarray
    #   XGBoost  : model.get_score(importance_type="gain") — returns dict {feature: score}
    #              "gain" chosen over "weight": gain measures actual improvement in loss,
    #              not just how often a feature is used — more informative ranking.
    import xgboost as xgb_lib
    if isinstance(model, xgb_lib.Booster):
        # XGBoost Booster: get_score() returns only features with non-zero importance
        score_dict  = model.get_score(importance_type="gain")
        importances = np.array([score_dict.get(f, 0.0) for f in feature_names])
    else:
        # LightGBM (sklearn API)
        importances = model.feature_importances_

    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    # Print top features
    print(f">> Top {top_n} features by importance:")
    for i, row in imp_df.head(top_n).iterrows():
        print(f"   {i+1:3d}. {row['feature']:30s}  importance: {row['importance']}")

    # Zero-importance features
    zero_imp = (imp_df["importance"] == 0).sum()
    print(f"\n   Features with zero importance: {zero_imp} / {len(imp_df)}")

    # Plot — dynamic height scales with number of features shown
    top = imp_df.head(top_n).sort_values("importance", ascending=True)
    fig_height = max(figsize_min_height, top_n * figsize_height_per_feature)

    fig, ax = plt.subplots(figsize=(figsize_width, fig_height))

    ax.barh(range(len(top)), top["importance"], color="#b3cde3", alpha=0.9)

    # Value labels
    for i, (_, row) in enumerate(top.iterrows()):
        ax.text(row["importance"] + max(top["importance"]) * 0.01, i,
                f"{row['importance']:.0f}", va="center", fontsize=9,
                color="#333333", fontweight="bold")

    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"], fontsize=9)
    ax.set_xlabel("Importance (split count)")
    ax.set_title(f"Top {top_n} Feature Importance — {model_name}")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"   Saved plot to {save_path}")

    plt.show()

    return imp_df