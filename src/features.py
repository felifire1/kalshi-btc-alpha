"""
features.py  —  REAL DATA VERSION
===================================
Builds the X feature matrix by joining:
  - Kalshi BTC daily markets (entry price, strike, resolution)
  - BTC hourly prices (momentum, volatility, fair probability)

Key insight: the 'fair probability' is computed analytically using
Black-Scholes for binary options. The gap between Kalshi's price
and this fair probability IS our signal.
"""

import pandas as pd
import numpy as np
from data_loader import compute_fair_probability


def build_features(markets: pd.DataFrame,
                   btc:     pd.DataFrame) -> tuple:
    """
    Parameters
    ----------
    markets : from data_loader.load_btc_daily_markets()
    btc     : from data_loader.load_btc_hourly_prices()

    Returns
    -------
    df           : full feature dataframe
    feature_cols : list of X column names
    """
    df = markets.copy()

    # ── Join BTC price at market open time ────────────────────────────────────
    print("Joining BTC price data to markets...")
    btc_at_open = (btc[["close", "ret_24h", "ret_72h", "ret_168h",
                          "vol_24h", "vol_168h", "rsi"]]
                      .copy())
    btc_at_open.columns = ["btc_price", "ret_1d", "ret_3d", "ret_7d",
                            "vol_1d", "vol_7d", "rsi"]

    # Align on nearest hourly timestamp
    df["open_time_utc"] = pd.to_datetime(df["open_time"], utc=True)
    btc_idx             = btc_at_open.index

    def get_btc_at(ts):
        idx = btc_idx.searchsorted(ts)
        if idx >= len(btc_idx): idx = len(btc_idx) - 1
        return btc_at_open.iloc[idx]

    btc_joined = pd.DataFrame(
        [get_btc_at(ts) for ts in df["open_time_utc"]],
        index=df.index
    )
    df = pd.concat([df, btc_joined], axis=1)

    # ── Fair probability (Black-Scholes binary) ───────────────────────────────
    # This is our analytical benchmark — replaces CME FedWatch from FOMC strategy
    # vol_7d is hourly std; annualize to daily for fair prob calculation
    df["fair_prob"] = df.apply(
        lambda r: compute_fair_probability(
            btc_price = r["btc_price"],
            strike    = r["strike"],
            vol_hourly = r["vol_7d"] if pd.notna(r["vol_7d"]) else 0.035 / np.sqrt(24),
            hours     = 25
        ),
        axis=1
    )

    # ── Core signal: Kalshi vs Fair Probability ───────────────────────────────
    df["gap"]     = df["fair_prob"] - df["entry_prob"]   # + means Kalshi underpriced YES
    df["gap_abs"] = df["gap"].abs()
    df["gap_pct"] = df["gap"] / (df["fair_prob"] + 1e-6)

    # ── Strike proximity features ─────────────────────────────────────────────
    df["dist_to_strike"]     = df["btc_price"] - df["strike"]
    df["dist_pct"]           = df["dist_to_strike"] / df["strike"]
    df["abs_dist_pct"]       = df["dist_pct"].abs()
    df["btc_above_strike"]   = (df["btc_price"] > df["strike"]).astype(int)

    # Near-the-money flag (within 2% of strike) — most mispricing here
    df["near_money"] = (df["abs_dist_pct"] < 0.02).astype(int)

    # ── Momentum & volatility regime ──────────────────────────────────────────
    df["momentum_score"] = (
        df["ret_1d"].fillna(0) * 0.5 +
        df["ret_3d"].fillna(0) * 0.3 +
        df["ret_7d"].fillna(0) * 0.2
    )
    df["high_vol"]       = (df["vol_7d"] > 0.04).astype(int)
    df["overbought"]     = (df["rsi"] > 70).astype(int)
    df["oversold"]       = (df["rsi"] < 30).astype(int)
    df["strong_trend"]   = (df["momentum_score"].abs() > 0.03).astype(int)

    # ── Time features ─────────────────────────────────────────────────────────
    df["hour_of_day"]  = df["open_time_utc"].dt.hour
    df["day_of_week"]  = df["open_time_utc"].dt.dayofweek
    df["month"]        = df["open_time_utc"].dt.month
    df["year"]         = df["open_time_utc"].dt.year

    # ── Historical rolling features ───────────────────────────────────────────
    df = df.sort_values("open_time").reset_index(drop=True)

    # Rolling win rate for this gap direction (last 50 trades, no lookahead)
    df["rolling_win_rate"] = (
        df["resolved_yes"]
        .shift(1)
        .rolling(50, min_periods=10)
        .mean()
        .fillna(0.5)
    )

    # Rolling avg gap (is the current gap large relative to recent gaps?)
    df["rolling_avg_gap"] = (
        df["gap_abs"]
        .shift(1)
        .rolling(50, min_periods=10)
        .mean()
        .fillna(df["gap_abs"].median())
    )
    df["gap_vs_avg"] = df["gap_abs"] / (df["rolling_avg_gap"] + 1e-6)

    # ── Target variables ──────────────────────────────────────────────────────
    # Y1 (Classification): did buying YES when gap>0 (Kalshi underpriced) profit?
    # i.e., we buy YES when fair_prob > entry_prob
    # Profit if it resolves YES
    df["y_trade_profitable"] = (
        ((df["gap"] > 0) & (df["resolved_yes"] == 1)) |   # bought YES, resolved YES
        ((df["gap"] < 0) & (df["resolved_yes"] == 0))     # bought NO,  resolved NO
    ).astype(int)

    # Y2 (Regression): how much did the price actually converge?
    # exit_prob - entry_prob (positive = price moved toward fair value)
    # This is NOT derivable from features — no leakage
    df["y_convergence_pp"] = (df["exit_prob"] - df["entry_prob"]) * 100

    # ── Feature list ─────────────────────────────────────────────────────────
    feature_cols = [
        # Core signal
        "gap", "gap_abs", "gap_pct", "gap_vs_avg",
        # BTC price signals
        "ret_1d", "ret_3d", "ret_7d",
        "vol_1d", "vol_7d", "rsi",
        # Strike proximity
        "dist_pct", "abs_dist_pct",
        "btc_above_strike", "near_money",
        # Regime
        "momentum_score", "high_vol",
        "overbought", "oversold", "strong_trend",
        # Timing
        "hour_of_day", "day_of_week", "month",
        # Historical
        "rolling_win_rate",
    ]

    clean = df[feature_cols + ["y_trade_profitable", "y_convergence_pp",
                                "entry_prob", "exit_prob", "fair_prob", "gap",
                                "resolved_yes", "strike", "btc_price",
                                "open_time", "close_time", "ticker"]].dropna()

    print(f"\nFeature matrix: {len(clean):,} rows × {len(feature_cols)} features")
    print(f"  YES resolution rate:     {clean['resolved_yes'].mean():.1%}")
    print(f"  Trade profitable rate:   {clean['y_trade_profitable'].mean():.1%}")
    print(f"  Avg gap (abs):           {clean['gap_abs'].mean():.3f} ({clean['gap_abs'].mean()*100:.1f}pp)")
    print(f"  Near-the-money markets:  {clean['near_money'].mean():.1%}")

    return clean, feature_cols


def get_X_y(df: pd.DataFrame,
            feature_cols: list,
            target: str = "y_trade_profitable") -> tuple:
    keep = df[feature_cols + [target]].dropna()
    X    = keep[feature_cols].values
    y    = keep[target].values
    return X, y, feature_cols
