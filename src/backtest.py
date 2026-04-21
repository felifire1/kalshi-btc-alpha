"""
backtest.py
===========
Simulates trading on historical Kalshi BTC markets using model signals.

Strategy rules:
  - Enter a trade only when Model 1 (XGBoost) predicts probability > threshold
  - Position size proportional to Model 2 (gap closure regression) prediction
  - Exit at convergence (or resolution if no convergence)
  - Track P&L, win rate, Sharpe ratio
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import os

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
BG, PANEL, TEXT, GRID = "#0F1117", "#1A1D27", "#E8E8E8", "#2A2D3A"
GREEN, RED, ORG, BLUE  = "#00C48C", "#E74C3C", "#F5A623", "#4A90D9"


def run_backtest(df:            pd.DataFrame,
                 model_probs:   np.ndarray,
                 feature_cols:  list,
                 threshold:     float = 0.55,
                 position_size: float = 100.0,
                 transaction_cost: float = 0.01) -> pd.DataFrame:
    """
    Simulate trades on the test set.

    Parameters
    ----------
    df            : full feature dataframe (test portion)
    model_probs   : XGBoost predicted probabilities for each row
    threshold     : only trade when model_prob > threshold
    position_size : dollars per trade (base)
    transaction_cost : spread/fee per trade (in probability units, e.g. 0.01 = 1pp)

    Returns
    -------
    trades_df : one row per executed trade with P&L
    """
    test_df = df.iloc[-len(model_probs):].copy().reset_index(drop=True)
    # Remove any duplicate columns before iterating
    test_df = test_df.loc[:, ~test_df.columns.duplicated()]
    test_df["model_prob"] = model_probs

    trades = []
    for _, row in test_df.iterrows():
        # ── Signal filter ────────────────────────────────────────────────────
        if float(row["model_prob"]) < threshold:
            continue

        entry_prob = float(row["entry_prob"])

        # Only trade near-the-money markets (40¢–60¢)
        # These have symmetric payoffs — no need for >60% win rate to profit
        if not (0.38 <= entry_prob <= 0.62):
            continue

        # ── Direction: use momentum + BTC position (top ML features) ───────────
        # momentum_score > 0 AND btc above strike → BUY YES (trend confirms position)
        # momentum_score < 0 AND btc below strike → BUY NO  (trend confirms position)
        # Only trade when momentum confirms the btc_above_strike signal
        above  = float(row["btc_above_strike"]) == 1
        mom    = float(row["momentum_score"])

        if above and mom > 0:
            buy_yes = True     # BTC above strike + upward momentum → YES likely holds
        elif not above and mom < 0:
            buy_yes = False    # BTC below strike + downward momentum → NO likely holds
        else:
            continue           # signals conflict — skip trade

        if buy_yes:
            entry_price       = entry_prob + transaction_cost
            exit_price        = float(row["exit_prob"]) - transaction_cost
            resolution_payout = float(row["resolved_yes"])
        else:
            entry_price       = (1 - entry_prob) + transaction_cost
            exit_price        = (1 - float(row["exit_prob"])) - transaction_cost
            resolution_payout = 1 - float(row["resolved_yes"])

        if entry_price >= 1.0 or entry_price <= 0:
            continue

        # ── P&L: hold to resolution ───────────────────────────────────────────
        # Near-the-money + symmetric payoff → resolution P&L is meaningful
        pnl       = (resolution_payout - entry_price) * position_size
        exit_type = "resolution"

        trades.append({
            "open_time":      row["open_time"],
            "close_time": row["close_time"],
            "ticker":          row.get("ticker", ""),
            "strike":    row["strike"],
            "btc_price": row["btc_price"],
            "entry_price":     entry_price,
            "exit_price":      exit_price,
            "model_prob":      row["model_prob"],
            "gap_at_entry":    row["gap"],
            "resolved_yes":    row["resolved_yes"],
            "pnl":             pnl,
            "pnl_pct":         pnl / position_size * 100,
            "profitable":      int(pnl > 0),
            "exit_type":       exit_type,
        })

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        print("No trades executed — lower the threshold")
        return trades_df

    trades_df["cumulative_pnl"]    = trades_df["pnl"].cumsum()
    trades_df["cumulative_return"] = trades_df["cumulative_pnl"] / position_size

    _print_stats(trades_df, position_size)
    return trades_df


def _print_stats(t: pd.DataFrame, position_size: float):
    n          = len(t)
    wins       = t["profitable"].sum()
    total_pnl  = t["pnl"].sum()
    avg_pnl    = t["pnl"].mean()
    avg_return = t["pnl_pct"].mean()

    # Sharpe ratio (annualized, assuming weekly trades)
    weekly_ret = t["pnl_pct"] / 100
    sharpe     = (weekly_ret.mean() / weekly_ret.std()) * np.sqrt(52) \
                 if weekly_ret.std() > 0 else 0

    # Max drawdown
    cum = t["cumulative_pnl"]
    peak = cum.cummax()
    dd   = (cum - peak)
    max_dd = dd.min()

    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)
    print(f"  Total trades:        {n}")
    print(f"  Win rate:            {wins/n:.1%}")
    print(f"  Total P&L:           ${total_pnl:,.2f}")
    print(f"  Avg P&L per trade:   ${avg_pnl:.2f}")
    print(f"  Avg return/trade:    {avg_return:.1f}%")
    print(f"  Annualized Sharpe:   {sharpe:.2f}")
    print(f"  Max drawdown:        ${max_dd:,.2f}")
    print(f"  Convergence exits:   {(t['exit_type']=='convergence').sum()}")
    print(f"  Resolution exits:    {(t['exit_type']=='resolution').sum()}")


def plot_backtest(trades_df: pd.DataFrame):
    if trades_df.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor=BG)
    fig.suptitle("BTC Kalshi Strategy — Backtest Results",
                 color=TEXT, fontsize=13, fontweight="bold")

    def style(ax):
        ax.set_facecolor(PANEL)
        ax.spines[:].set_color(GRID)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.yaxis.grid(True, color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)

    # ── Panel 1: Cumulative P&L ────────────────────────────────────────────
    ax = axes[0, 0]; style(ax)
    ax.plot(range(len(trades_df)), trades_df["cumulative_pnl"],
            color=GREEN, lw=2.5)
    ax.fill_between(range(len(trades_df)), 0, trades_df["cumulative_pnl"],
                    alpha=0.15, color=GREEN)
    ax.axhline(0, color=RED, lw=1, ls="--", alpha=0.6)
    ax.set_xlabel("Trade #", color=TEXT, fontsize=9)
    ax.set_ylabel("Cumulative P&L ($)", color=TEXT, fontsize=9)
    ax.set_title("Cumulative P&L", color=TEXT, fontsize=10)

    # ── Panel 2: P&L per trade (bar) ───────────────────────────────────────
    ax = axes[0, 1]; style(ax)
    colors = [GREEN if p > 0 else RED for p in trades_df["pnl"]]
    ax.bar(range(len(trades_df)), trades_df["pnl"], color=colors, width=0.7)
    ax.axhline(0, color=TEXT, lw=0.8, alpha=0.4)
    ax.set_xlabel("Trade #", color=TEXT, fontsize=9)
    ax.set_ylabel("P&L ($)", color=TEXT, fontsize=9)
    ax.set_title("P&L per Trade", color=TEXT, fontsize=10)

    # ── Panel 3: Return distribution ──────────────────────────────────────
    ax = axes[1, 0]; style(ax)
    ax.hist(trades_df["pnl_pct"], bins=20, color=BLUE, alpha=0.8, edgecolor=GRID)
    ax.axvline(trades_df["pnl_pct"].mean(), color=ORG, lw=2,
               label=f"Mean: {trades_df['pnl_pct'].mean():.1f}%")
    ax.axvline(0, color=RED, lw=1.5, ls="--", alpha=0.7)
    ax.set_xlabel("Return per Trade (%)", color=TEXT, fontsize=9)
    ax.set_ylabel("Frequency", color=TEXT, fontsize=9)
    ax.set_title("Return Distribution", color=TEXT, fontsize=10)
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

    # ── Panel 4: Win rate by gap size ─────────────────────────────────────
    ax = axes[1, 1]; style(ax)
    trades_df["gap_bucket"] = pd.cut(
        trades_df["gap_at_entry"],
        bins=[0, 0.05, 0.08, 0.12, 0.20, 1.0],
        labels=["2-5pp", "5-8pp", "8-12pp", "12-20pp", "20pp+"]
    )
    win_by_gap = trades_df.groupby("gap_bucket", observed=True)["profitable"].mean()
    count_by_gap = trades_df.groupby("gap_bucket", observed=True).size()
    bars = ax.bar(win_by_gap.index.astype(str), win_by_gap.values,
                  color=ORG, alpha=0.85)
    ax.axhline(0.5, color=RED, lw=1.5, ls="--", alpha=0.7, label="50% baseline")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Gap Size at Entry", color=TEXT, fontsize=9)
    ax.set_ylabel("Win Rate", color=TEXT, fontsize=9)
    ax.set_title("Win Rate by Gap Size", color=TEXT, fontsize=10)
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    for bar, count in zip(bars, count_by_gap):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.02,
                f"n={count}", ha="center", color=TEXT, fontsize=7.5)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "backtest_results.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {out_path}")
    return fig
