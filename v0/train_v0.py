"""
train_v0.py — Model training for baseline v0
═════════════════════════════════════════════════
Train LightGBM with early stopping on validation set.
Default parameters — no hyperparameter tuning.

Functions:
    train_lgbm() — train LightGBM classifier, return model + predictions
"""

import lightgbm as lgb
import numpy as np
import pandas as pd


# Default LightGBM parameters for baseline
# Every value is justified below — no magic numbers.
DEFAULT_LGBM_PARAMS = dict(
    objective="binary",         # Binary classification (fraud / not fraud)
    metric="auc",               # Optimize ROC AUC — robust to class imbalance (3.5% fraud rate)
    boosting_type="gbdt",       # Gradient Boosted Decision Trees — standard, well-proven choice
    n_estimators=3000,          # High ceiling — actual stopping controlled by early_stopping,
                                # so this is a safe upper bound, not a tuned value
    learning_rate=0.05,         # Moderate LR: balances training speed and generalization.
                                # Lower than default (0.1) to allow early stopping to find
                                # a better optimum with more trees
    num_leaves=128,             # Controls tree complexity. Default 31 is too shallow for
                                # ~500k rows with 400+ features. 128 allows richer patterns
                                # without severe overfitting at this data scale
    max_depth=-1,               # No depth limit — num_leaves already controls complexity
                                # more directly than depth for LightGBM (leaf-wise growth)
    min_child_samples=50,       # Minimum samples per leaf. Higher than default (20) to reduce
                                # overfitting on rare fraud patterns in a 3.5% imbalanced dataset
    subsample=0.8,              # Row subsampling per tree: reduces overfitting, speeds training.
                                # 0.8 = standard starting point (retains 80% of rows per tree)
    colsample_bytree=0.4,       # Feature subsampling per tree: with 400+ features, using all
                                # columns leads to correlated trees. 0.4 forces diversity
    reg_alpha=0.1,              # L1 regularization: encourages sparse feature usage,
                                # helps with the many near-zero-importance features in this dataset
    reg_lambda=1.0,             # L2 regularization: standard smoothing, prevents large weights
                                # on noisy features
    random_state=42,            # Fixed seed for reproducibility across all runs
    n_jobs=-1,                  # Use all available CPU cores
    verbose=-1,                 # Suppress LightGBM internal logs (controlled via callbacks)
)


def train_lgbm(X_train, y_train, X_val, y_val,
               params=None, early_stopping=200, log_period=100, verbose=True):
    """
    Train LightGBM binary classifier with early stopping.

    Parameters
    ----------
    X_train        : pd.DataFrame — training features
    y_train        : pd.Series    — training target
    X_val          : pd.DataFrame — validation features for early stopping
    y_val          : pd.Series    — validation target
    params         : dict or None — LightGBM parameters (None = use DEFAULT_LGBM_PARAMS)
    early_stopping : int          — stop if no AUC improvement for this many rounds.
                     Default=200: large enough to escape local plateaus, small enough
                     to avoid wasting compute time
    log_period     : int          — print validation score every N rounds (default=100)
    verbose        : bool         — print training summary

    Returns
    -------
    tuple (model, y_pred_val)
        model      — fitted LGBMClassifier
        y_pred_val — predicted probabilities on validation set
    """
    params = params or DEFAULT_LGBM_PARAMS.copy()

    if verbose:
        print(">> Training LightGBM...")
        print(f"   Train: {X_train.shape[0]:,} samples x {X_train.shape[1]} features")
        print(f"   Val:   {X_val.shape[0]:,} samples x {X_val.shape[1]} features")
        print(f"   Early stopping: {early_stopping} rounds")
        print(f"   Key params: lr={params.get('learning_rate')}, "
              f"leaves={params.get('num_leaves')}, "
              f"estimators={params.get('n_estimators')}")

    model = lgb.LGBMClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(early_stopping, verbose=True),
            lgb.log_evaluation(period=log_period),
        ],
    )

    y_pred_val = model.predict_proba(X_val)[:, 1]

    if verbose:
        print(f"\n   Best iteration: {model.best_iteration_}")
        print(f"   Best AUC on val: {model.best_score_['valid_0']['auc']:.6f}")

    return model, y_pred_val