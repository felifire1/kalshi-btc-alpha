import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import (classification_report, roc_auc_score, roc_curve,
                                     mean_absolute_error, r2_score)
from xgboost                 import XGBClassifier, XGBRegressor
import os

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BG, PANEL, TEXT, GRID = "#0F1117", "#1A1D27", "#E8E8E8", "#2A2D3A"
GREEN, BLUE, ORG, RED = "#00C48C", "#4A90D9", "#F5A623", "#E74C3C"


def train_test_split_temporal(X, y, test_size: float = 0.2):
    """Chronological split — never shuffle time-series data."""
    split = int(len(X) * (1 - test_size))
    return X[:split], X[split:], y[:split], y[split:]


def train_logistic(X_train, y_train, X_test, y_test, feature_cols):
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    model  = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_tr_sc, y_train)

    y_pred = model.predict(X_te_sc)
    y_prob = model.predict_proba(X_te_sc)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)

    print(f"Logistic Regression  |  AUC: {auc:.3f}  |  Acc: {(y_pred == y_test).mean():.1%}")
    print(classification_report(y_test, y_pred, target_names=["Not Profitable", "Profitable"]))

    coef_df = pd.DataFrame(
        list(zip(feature_cols, model.coef_[0])),
        columns=["feature", "coefficient"]
    ).sort_values("coefficient", key=abs, ascending=False)
    print(coef_df.head(5).to_string(index=False))

    return model, scaler, auc, y_prob


def train_random_forest(X_train, y_train, X_test, y_test, feature_cols):
    model = RandomForestClassifier(
        n_estimators=200, max_depth=6,
        min_samples_leaf=5, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)

    print(f"Random Forest  |  AUC: {auc:.3f}  |  Acc: {(y_pred == y_test).mean():.1%}")
    print(classification_report(y_test, y_pred, target_names=["Not Profitable", "Profitable"]))

    imp_df = pd.DataFrame(
        list(zip(feature_cols, model.feature_importances_)),
        columns=["feature", "importance"]
    ).sort_values("importance", ascending=False)
    print(imp_df.head(10).to_string(index=False))

    return model, auc, y_prob, imp_df


def train_xgboost(X_train, y_train, X_test, y_test, feature_cols):
    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42, verbosity=0
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)

    print(f"XGBoost  |  AUC: {auc:.3f}  |  Acc: {(y_pred == y_test).mean():.1%}")
    print(classification_report(y_test, y_pred, target_names=["Not Profitable", "Profitable"]))

    imp_df = pd.DataFrame(
        list(zip(feature_cols, model.feature_importances_)),
        columns=["feature", "importance"]
    ).sort_values("importance", ascending=False)
    print(imp_df.head(10).to_string(index=False))

    return model, auc, y_prob, imp_df


def train_gap_regression(X_train, y_train_reg, X_test, y_test_reg, feature_cols):
    model = XGBRegressor(
        n_estimators=200, max_depth=4,
        learning_rate=0.05, subsample=0.8,
        random_state=42, verbosity=0
    )
    model.fit(X_train, y_train_reg)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test_reg, y_pred)
    r2  = r2_score(y_test_reg, y_pred)
    print(f"Gap Regression  |  MAE: {mae:.2f}pp  |  R2: {r2:.3f}")

    return model, mae, r2


def plot_model_results(y_test, probs_lr, probs_rf, probs_xgb,
                       auc_lr, auc_rf, auc_xgb, imp_xgb, feature_cols):

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=BG)
    fig.suptitle("BTC Kalshi — Model Results", color=TEXT, fontsize=13, fontweight="bold")

    def style(ax):
        ax.set_facecolor(PANEL)
        ax.spines[:].set_color(GRID)
        ax.tick_params(colors=TEXT)

    # ROC curves
    ax = axes[0]; style(ax)
    for probs, auc, label, color in [
        (probs_lr,  auc_lr,  f"Logistic Reg  (AUC={auc_lr:.3f})",  BLUE),
        (probs_rf,  auc_rf,  f"Random Forest (AUC={auc_rf:.3f})",  ORG),
        (probs_xgb, auc_xgb, f"XGBoost       (AUC={auc_xgb:.3f})", GREEN),
    ]:
        fpr, tpr, _ = roc_curve(y_test, probs)
        ax.plot(fpr, tpr, color=color, lw=2, label=label)
    ax.plot([0,1],[0,1], color=GRID, lw=1, ls="--", label="Random (0.500)")
    ax.set_xlabel("False Positive Rate", color=TEXT, fontsize=9)
    ax.set_ylabel("True Positive Rate",  color=TEXT, fontsize=9)
    ax.set_title("ROC Curves", color=TEXT, fontsize=10)
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

    # Feature importance (XGBoost)
    ax = axes[1]; style(ax)
    top10 = imp_xgb.head(10)
    ax.barh(top10["feature"][::-1], top10["importance"][::-1], color=GREEN, alpha=0.85)
    ax.set_xlabel("Importance", color=TEXT, fontsize=9)
    ax.set_title("XGBoost Feature Importance", color=TEXT, fontsize=10)
    ax.tick_params(labelsize=8)

    # AUC comparison
    ax = axes[2]; style(ax)
    bars = ax.bar(["Logistic\nRegression", "Random\nForest", "XGBoost"],
                  [auc_lr, auc_rf, auc_xgb],
                  color=[BLUE, ORG, GREEN], width=0.5, zorder=3)
    ax.axhline(0.5, color=RED, lw=1.5, ls="--", alpha=0.7, label="Random baseline")
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("AUC-ROC", color=TEXT, fontsize=9)
    ax.set_title("AUC Comparison", color=TEXT, fontsize=10)
    ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    for bar, auc in zip(bars, [auc_lr, auc_rf, auc_xgb]):
        ax.text(bar.get_x() + bar.get_width()/2, auc + 0.005,
                f"{auc:.3f}", ha="center", va="bottom", color=TEXT, fontsize=9, fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "model_results.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {out_path}")
    return fig
