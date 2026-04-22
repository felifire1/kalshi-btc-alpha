"""
make_model_chart.py  —  updated with accuracy scores on every bar
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BG    = "#0F1117"
PANEL = "#1A1D27"
GRID  = "#2A2D3A"
TEXT  = "#E8E8E8"
MUTED = "#888A99"
BLUE  = "#4A90D9"
ORG   = "#F5A623"
TEAL  = "#00C48C"
RED   = "#E74C3C"

models = ["Logistic\nRegression", "Random\nForest", "XGBoost"]
colors = [BLUE, ORG, TEAL]

# All metrics in one dict — AUC & Accuracy first, then classification
all_metrics = {
    "AUC-ROC":               [0.699, 0.755, 0.749],
    "Accuracy":              [0.663, 0.701, 0.690],
    "Precision\n(Trade)":    [0.69,  0.79,  0.69 ],
    "Recall\n(Trade)":       [0.56,  0.52,  0.66 ],
    "F1\n(Trade)":           [0.62,  0.63,  0.68 ],
    "Precision\n(No Trade)": [0.64,  0.66,  0.69 ],
    "Recall\n(No Trade)":    [0.76,  0.87,  0.71 ],
    "F1\n(No Trade)":        [0.70,  0.75,  0.70 ],
}

fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG,
                          gridspec_kw={"width_ratios": [2, 5]})
fig.suptitle("Model Performance — AUC, Accuracy & Full Classification Metrics (Test Set)",
             color=TEXT, fontsize=13, fontweight="bold", y=1.02)

def style_ax(ax):
    ax.set_facecolor(PANEL)
    ax.spines[:].set_color(GRID)
    ax.tick_params(colors=TEXT, labelsize=10)
    ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)

# ── Panel 1: AUC & Accuracy with labeled bars ─────────────────────────────────
ax1 = axes[0]; style_ax(ax1)

headline = {"AUC-ROC": [0.699, 0.755, 0.749],
            "Accuracy": [0.663, 0.701, 0.690]}
x  = np.arange(len(headline))
w  = 0.22
n_models = len(models)

for mi, (mname, vals) in enumerate(headline.items()):
    offsets = np.linspace(-(n_models-1)*w/2, (n_models-1)*w/2, n_models)
    for ci, (val, col) in enumerate(zip(vals, colors)):
        ax1.bar(x[mi] + offsets[ci], val, width=w, color=col, alpha=0.9, zorder=3)
        # Value label on bar
        ax1.text(x[mi] + offsets[ci], val + 0.006,
                 f"{val:.3f}" if mname == "AUC-ROC" else f"{val:.1%}",
                 ha="center", va="bottom", color=col, fontsize=9, fontweight="bold")

ax1.axhline(0.5, color=RED, lw=1.5, ls="--", alpha=0.6, label="Random baseline")
ax1.set_xticks(x)
ax1.set_xticklabels(["AUC-ROC", "Accuracy"], color=TEXT, fontsize=12, fontweight="bold")
ax1.set_ylim(0.45, 0.85)
ax1.set_ylabel("Score", color=TEXT, fontsize=10)
ax1.set_title("Headline Metrics", color=TEXT, fontsize=11, pad=10)

legend_handles = [mpatches.Patch(color=c, label=m.replace("\n"," "))
                  for m, c in zip(models, colors)]
legend_handles.append(plt.Line2D([0],[0], color=RED, ls="--", lw=1.5, label="Random (0.5)"))
ax1.legend(handles=legend_handles, facecolor=PANEL, edgecolor=GRID,
           labelcolor=TEXT, fontsize=9, loc="lower right")

# ── Panel 2: Precision / Recall / F1 with accuracy reference line ─────────────
ax2 = axes[1]; style_ax(ax2)

clf_metrics = {k: v for k, v in all_metrics.items()
               if k not in ("AUC-ROC", "Accuracy")}
clf_names = list(clf_metrics.keys())
acc_vals  = all_metrics["Accuracy"]
x2 = np.arange(len(clf_names))
w2 = 0.24

for ci, col in enumerate(colors):
    vals   = [clf_metrics[m][ci] for m in clf_names]
    offset = (ci - 1) * w2
    bars   = ax2.bar(x2 + offset, vals, width=w2, color=col, alpha=0.9, zorder=3,
                     label=models[ci].replace("\n"," "))
    for bar, val in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.008,
                 f"{val:.2f}", ha="center", va="bottom",
                 color=col, fontsize=8.5, fontweight="bold")

# Accuracy reference lines — one per model
line_styles = ["--", "-.", ":"]
for ci, (col, acc) in enumerate(zip(colors, acc_vals)):
    ax2.axhline(acc, color=col, lw=1.2, ls=line_styles[ci], alpha=0.55,
                label=f"Accuracy ({models[ci].replace(chr(10),' ')}: {acc:.1%})")

# Shade trade vs no-trade groups
ax2.axvspan(-0.5, 2.5, alpha=0.04, color=TEAL,  zorder=0)
ax2.axvspan(2.5,  5.5, alpha=0.04, color=BLUE,   zorder=0)
ax2.text(1.0, 0.44, "Profitable class",    ha="center", color=TEAL, fontsize=9, style="italic")
ax2.text(4.0, 0.44, "Not-profitable class", ha="center", color=BLUE, fontsize=9, style="italic")

ax2.axhline(0.5, color=RED, lw=1.2, ls="--", alpha=0.4)
ax2.set_xticks(x2)
ax2.set_xticklabels(clf_names, color=TEXT, fontsize=10)
ax2.set_ylim(0.42, 1.0)
ax2.set_ylabel("Score", color=TEXT, fontsize=10)
ax2.set_title("Precision  /  Recall  /  F1  by Class  (dashed lines = model accuracy)",
              color=TEXT, fontsize=11, pad=10)
ax2.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8, ncol=2)

plt.tight_layout()
out = "/Users/felipequiroz/Desktop/FINA_4390/btc_kalshi/output/model_comparison.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"Saved: {out}")
