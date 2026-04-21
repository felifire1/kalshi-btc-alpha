"""
data_loader.py  —  REAL DATA VERSION v2
=========================================
Correctly sources entry/exit prices from the TRADES table,
not the markets table (which only has the price at fetch time).

Strategy:
  - Entry price = first trade in the market (shortly after open)
  - Exit price  = last trade before close
  - Fair prob   = Black-Scholes binary using BTC spot + realized vol
  - Gap         = fair_prob - entry_price  → our signal
"""

import os
import pandas as pd
import numpy as np

REPO_DATA    = os.path.expanduser("~/Desktop/prediction-market-analysis/data/kalshi")
MARKETS_GLOB = os.path.join(REPO_DATA, "markets", "*.parquet")
TRADES_GLOB  = os.path.join(REPO_DATA, "trades",  "*.parquet")


def load_btc_dataset() -> pd.DataFrame:
    """
    Builds the core dataset by joining:
      - Resolved 25-hour BTC daily markets
      - First + last trade prices per market (from trades table)
      - Filtered to markets with enough trading activity

    Returns one row per market.
    """
    import duckdb
    print("Building BTC dataset from real trades...")
    con = duckdb.connect()

    # ── Step 1: All resolved daily BTC markets ────────────────────────────────
    print("  Loading markets...")
    markets = con.execute(f"""
        SELECT
            ticker,
            result,
            open_time,
            close_time,
            volume,
            CAST(REGEXP_EXTRACT(ticker, '-T([0-9.]+)$', 1) AS DOUBLE) AS strike
        FROM read_parquet('{MARKETS_GLOB}')
        WHERE ticker LIKE 'KXBTCD-%'
          AND result IN ('yes', 'no')
          AND DATEDIFF('hour', open_time, close_time) = 25
          AND volume >= 100
        ORDER BY open_time ASC
    """).df()
    print(f"  Found {len(markets):,} resolved daily BTC markets")

    # ── Step 2: First + last trade per market ─────────────────────────────────
    print("  Pulling entry/exit prices from trades (this takes ~1 min)...")
    tickers_str = "', '".join(markets["ticker"].dropna().tolist())

    trades = con.execute(f"""
        SELECT
            ticker,
            yes_price,
            created_time,
            ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY created_time ASC)  AS rn_first,
            ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY created_time DESC) AS rn_last
        FROM read_parquet('{TRADES_GLOB}')
        WHERE ticker IN ('{tickers_str}')
    """).df()

    trades["created_time"] = pd.to_datetime(trades["created_time"], utc=True)
    trades["yes_prob"]     = trades["yes_price"] / 100.0

    entry_prices = (trades[trades["rn_first"] <= 3]        # avg first 3 trades
                    .groupby("ticker")["yes_prob"]
                    .mean()
                    .rename("entry_prob"))

    exit_prices  = (trades[trades["rn_last"] <= 3]         # avg last 3 trades
                    .groupby("ticker")["yes_prob"]
                    .mean()
                    .rename("exit_prob"))

    # ── Step 3: Merge ─────────────────────────────────────────────────────────
    df = (markets
          .merge(entry_prices.reset_index(), on="ticker", how="inner")
          .merge(exit_prices.reset_index(),  on="ticker", how="inner"))

    df["open_time"]    = pd.to_datetime(df["open_time"],  utc=True)
    df["close_time"]   = pd.to_datetime(df["close_time"], utc=True)
    df["open_date"]    = df["open_time"].dt.date
    df["resolved_yes"] = (df["result"] == "yes").astype(int)

    # Filter to uncertain markets (not deep ITM/OTM at entry)
    df = df[(df["entry_prob"] >= 0.10) & (df["entry_prob"] <= 0.90)]

    print(f"  Final dataset: {len(df):,} markets")
    print(f"  Date range:   {df['open_date'].min()} → {df['open_date'].max()}")
    print(f"  YES rate:     {df['resolved_yes'].mean():.1%}")
    print(f"  Avg entry:    {df['entry_prob'].mean():.2f}")
    print(f"  Avg exit:     {df['exit_prob'].mean():.2f}")

    return df


def load_btc_hourly_prices(start: str = "2024-01-01",
                            end:   str = "2025-11-25") -> pd.DataFrame:
    """BTC-USD hourly OHLCV + derived features from Yahoo Finance."""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("pip install yfinance")

    print("\nFetching BTC hourly prices...")
    btc = yf.download("BTC-USD", start=start, end=end,
                      interval="1h", progress=False)
    btc.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                   for c in btc.columns]
    btc.index   = pd.to_datetime(btc.index, utc=True)

    btc["ret_1h"]      = btc["close"].pct_change(1)
    btc["ret_24h"]     = btc["close"].pct_change(24)
    btc["ret_72h"]     = btc["close"].pct_change(72)
    btc["ret_168h"]    = btc["close"].pct_change(168)
    btc["vol_24h"]     = btc["ret_1h"].rolling(24).std()
    btc["vol_168h"]    = btc["ret_1h"].rolling(168).std()
    btc["rsi"]         = _rsi(btc["close"], 14 * 24)

    print(f"  {len(btc):,} hourly rows from {btc.index[0].date()} to {btc.index[-1].date()}")
    return btc


def _rsi(series, window=336):
    d    = series.diff()
    gain = d.clip(lower=0).rolling(window).mean()
    loss = (-d.clip(upper=0)).rolling(window).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def compute_fair_probability(btc_price, strike, vol_hourly, hours=25):
    """
    Probability that BTC finishes above `strike` in `hours` hours.
    Uses log-normal assumption (Black-Scholes binary call).
    This is our analytical benchmark.
    """
    from scipy.stats import norm
    if vol_hourly <= 0 or btc_price <= 0 or strike <= 0:
        return 0.5
    d = np.log(btc_price / strike) / (vol_hourly * np.sqrt(hours))
    return float(norm.cdf(d))


if __name__ == "__main__":
    df = load_btc_dataset()
    print("\nSample:")
    print(df[["ticker", "strike", "entry_prob", "exit_prob",
              "resolved_yes", "open_date"]].head(10).to_string())
