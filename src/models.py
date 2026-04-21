"""
models.py
=========
Trains and evaluates the three-model pipeline for BTC Kalshi mispricing.

Model 1 — Logistic Regression   (baseline classifier)
Model 2 — Random Forest          (non-linear classifier)
Model 3 — XGBoost                (final classifier + feature importance)

Also runs a regression model to predict gap closure size (position sizing).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import (classification_report, confusion_matrix,
                                     roc_auc_score, roc_curve,
                                     mean_absolute_error, r2_score)
from xgboost                 import XGBClassifier, XGBRegressor
import os

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BG    = "#0F1117"
PANEL = "#1A1D27"
TEXT  = "#E8E8E8"
GRID  = "#2A2D3A"
GREEN = "#00C48C"
BLUE  = "#4A90D9"
ORG   = "#F5A623"
RED   = "#E74C3C"


# ── Time-series safe train/test split ─────────────────────────────────────────
def train_test_split_temporal(X, y, test_size: float = 0.2):
    """
    Split data chronologically — NEVER shuffle prediction market data.
    Training on future data to predict the past = data leakage.
    """
    split = int(len(X) * (1 - test_size))
    return X[:split], X[split:], y[:split], y[split:]


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 1 — LOGISTIC REGRESSION (baseline)
# ══════════════════════════════════════════════════════════════════════════════
def train_logistic(X_train, y_train, X_test, y_test, feature_cols):
    print("\n" + "="*50)
    print("MODEL 1: Logistic Regression (Baseline)")
    print("="*50)

    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train)
    X_te_sc  = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_tr_sc, y_train)

    y_pred = model.predict(X_te_sc)
    y_prob = model.predict_proba(X_te_sc)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)

    print(f"AUC-ROC:  {auc:.3f}")
    print(f"Accuracy: {(y_pred == y_test).mean():.1%}")
    print(classification_report(y_test, y_pred,
                                 target_names=["Not Profitable","Profitable"]))

    # Top coefficients
    try:
        coef_df = pd.DataFrame(
            list(zip(feature_cols, model.coef_[0])),
            columns=["feature", "coefficient"]
        ).sort_values("coefficient", key=abs, ascending=False)
        print("\nTop 5 features (Logistic Regression):")
        print(coef_df.head(5).to_string(index=False))
    except Exception:
        pass

    return model, scaler, auc, y_prob


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 2 — RANDOM FOREST
# ══════════════════════════════════════════════════════════════════════════════
def train_random_forest(X_train, y_train, X_test, y_test, feature_cols):
    print("\n" + "="*50)
    print("MODEL 2: Random Forest")
    print("="*50)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)

    print(f"AUC-ROC:  {auc:.3f}")
    print(f"Accuracy: {(y_pred == y_test).mean():.1%}")
    print(classification_report(y_test, y_pred,
                                 target_names=["Not Profitable","Profitable"]))

    # Feature importance
    imp_df = pd.DataFrame(
        list(zip(feature_cols, model.feature_importances_)),
        columns=["feature", "importance"]
    ).sort_values("importance", ascending=False)
    print("\nTop 10 features (Random Forest):")
    print(imp_df.head(10).to_string(index=False))

    return model, auc, y_prob, imp_df


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 3 — XGBOOST (final model)
# ══════════════════════════════════════════════════════════════════════════════
def train_xgboost(X_train, y_train, X_test, y_test, feature_cols):
    print("\n" + "="*50)
    print("MODEL 3: XGBoost (Final Model)")
    print("="*50)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)

    print(f"AUC-ROC:  {auc:.3f}")
    print(f"Accuracy: {(y_pred == y_test).mean():.1%}")
    print(classification_report(y_test, y_pred,
                                 target_names=["Not Profitable","Profitable"]))

    # Feature importance
    imp_df = pd.DataFrame(
        list(zip(feature_cols, model.feature_importances_)),
        columns=["feature", "importance"]
    ).sort_values("importance", ascending=False)
    print("\nTop 10 features (XGBoost):")
    print(imp_df.head(10).to_string(index=False))

    return model, auc, y_prob, imp_df


# ══════════════════════════════════════════════════════════════════════════════
# REGRESSION — Gap closure size (position sizing)
# ══════════════════════════════════════════════════════════════════════════════
def train_gap_regression(X_train, y_train_reg, X_test, y_test_reg, feature_cols):
    print("\n" + "="*50)
    print("REGRESSION: Gap Closure Size (Position Sizing)")
    print("="*50)

    model = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train_reg)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test_reg, y_pred)
    r2  = r2_score(y_test_reg, y_pred)

    print(f"MAE:  {mae:.2f} percentage points")
    print(f"R²:   {r2:.3f}")
    print(f"  Predicting how many pp the gap closes (used for position sizing)")

    return model, mae, r2


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION — Model comparison + feature importance
# ══════════════════════════════════════════════════════════════════════════════
def plot_model_results(y_test,
                       probs_lr, probs_rf, probs_xgb,
                       auc_lr,   auc_rf,   auc_xgb,
                       imp_xgb:  pd.DataFrame,
                       feature_cols: list):

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=BG)
    fig.suptitle("BTC Kalshi Mispricing — Model Results",
                 color=TEXT, fontsize=13, fontweight="bold")

    # ── Panel 1: ROC Curves ───────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(PANEL)
    ax.spines[:].set_color(GRID)

    for probs, auc, label, color in [
        (probs_lr,  auc_lr,  f"Logistic Reg  (AUC={auc_lr:.3f})",  BLUE),
        (probs_rf,  auc_rf,  f"Random Forest (AUC={auc_rf:.3f})",  ORG),
        (probs_xgb, auc_xgb, f"XGBoost       (AUC={auc_xgb:.3f})", GREEN),
    ]:
        fpr, tpr, _ = roc_curve(y_test, probs)
        ax.plot(fpr, tpr, color=color, lw=2, label=label)

    ax.plot([0,1],[0,1], color=GRID, lw=1, ls="--", label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate", color=TEXT, fontsize=9)
    ax.set_ylabel("True Positive Rate",  color=TEXT, fontsize=9)
    ax.set_title("ROC Curves — All Models", color=TEXT, fontsize=10)
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    ax.tick_params(colors=TEXT)

    # ── Panel 2: XGBoost Feature Importance ───────────────────────────────────
    ax = axes[1]
    ax.set_facecolor(PANEL)
    ax.spines[:].set_color(GRID)

    top10 = imp_xgb.head(10)
    bars  = ax.barh(top10["feature"][::-1], top10["importance"][::-1],
                    color=GREEN, alpha=0.85)
    ax.set_xlabel("Importance Score", color=TEXT, fontsize=9)
    ax.set_title("XGBoost Feature Importance\n(Top 10)", color=TEXT, fontsize=10)
    ax.tick_params(colors=TEXT, labelsize=8)

    # ── Panel 3: AUC comparison bar chart ─────────────────────────────────────
    ax = axes[2]
    ax.set_facecolor(PANEL)
    ax.spines[:].set_color(GRID)

    models   = ["Logistic\nRegression", "Random\nForest", "XGBoost"]
    aucs     = [auc_lr, auc_rf, auc_xgb]
    colors   = [BLUE, ORG, GREEN]
    bars     = ax.bar(models, aucs, color=colors, width=0.5, zorder=3)
    ax.axhline(0.5, color=RED, lw=1.5, ls="--", alpha=0.7, label="Random baseline")
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("AUC-ROC", color=TEXT, fontsize=9)
    ax.set_title("Model Performance Comparison", color=TEXT, fontsize=10)
    ax.tick_params(colors=TEXT)
    ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, auc + 0.005,
                f"{auc:.3f}", ha="center", va="bottom",
                color=TEXT, fontsize=9, fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "model_results.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=BG)
    print(f"\nSaved: {out_path}")
    return fig
