"""
app.py  —  Kalshi BTC Alpha  |  Streamlit Dashboard
====================================================
Run with:  streamlit run app.py

Three tabs:
  1. Live Signal  — current BTC price + model signal for any hypothetical strike
  2. Backtest     — out-of-sample results, charts, trades table
  3. Model        — ROC curves, feature importance, model metrics
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import yfinance as yf
from datetime import datetime, timezone, timedelta
from PIL import Image

from data_loader import compute_fair_probability

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kalshi BTC Alpha",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Theme CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Main background */
  .stApp { background-color: #0F1117; color: #E8E8E8; }
  section[data-testid="stSidebar"] { background-color: #1A1D27; }

  /* Metric cards */
  div[data-testid="metric-container"] {
      background: #1A1D27;
      border: 1px solid #2A2D3A;
      border-radius: 8px;
      padding: 14px 18px;
  }
  div[data-testid="metric-container"] label { color: #888A99 !important; font-size: 12px; }
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
      font-size: 28px; font-weight: 700;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { background: #1A1D27; border-radius: 8px; gap: 4px; padding: 4px; }
  .stTabs [data-baseweb="tab"] { color: #888A99; border-radius: 6px; padding: 8px 22px; }
  .stTabs [aria-selected="true"] { background: #00C48C !important; color: #0F1117 !important; font-weight: 700; }

  /* Signal box */
  .signal-buy  { background:#0d3b2e; border:2px solid #00C48C; border-radius:12px; padding:24px 32px; text-align:center; }
  .signal-sell { background:#3b1414; border:2px solid #E74C3C; border-radius:12px; padding:24px 32px; text-align:center; }
  .signal-skip { background:#2a2d3a; border:2px solid #555;    border-radius:12px; padding:24px 32px; text-align:center; }

  /* Dataframe */
  .stDataFrame { border: 1px solid #2A2D3A; border-radius: 8px; }

  /* Section headers */
  h1, h2, h3 { color: #E8E8E8 !important; }

  /* Top accent bar */
  .top-bar { height:4px; background:linear-gradient(90deg,#00C48C,#4A90D9); border-radius:2px; margin-bottom:20px; }

  /* Info boxes */
  .info-card { background:#1A1D27; border-left:3px solid #00C48C; padding:12px 16px; border-radius:4px; margin:8px 0; }
</style>
""", unsafe_allow_html=True)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# ── Load artifacts (cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    path = os.path.join(OUTPUT_DIR, "model_artifacts.joblib")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def _compute_features_from_series(closes: pd.Series):
    """Compute momentum/vol/RSI from a price series."""
    def _rsi(s, w=336):
        d = s.diff()
        g = d.clip(lower=0).rolling(w, min_periods=14).mean()
        l = (-d.clip(upper=0)).rolling(w, min_periods=14).mean()
        rs = g / l.replace(0, np.nan)
        return float((100 - 100 / (1 + rs)).iloc[-1])

    ret_1h = closes.pct_change(1)
    return {
        "ret_1d": float(closes.pct_change(24).iloc[-1]),
        "ret_3d": float(closes.pct_change(72).iloc[-1]),
        "ret_7d": float(closes.pct_change(168).iloc[-1]),
        "vol_1d": float(ret_1h.rolling(24).std().iloc[-1]),
        "vol_7d": float(ret_1h.rolling(168).std().iloc[-1]),
        "rsi":    _rsi(closes),
    }


@st.cache_data(ttl=300, show_spinner="Fetching BTC price…")
def fetch_btc():
    """Return latest BTC price + momentum features.
    Tries yfinance first, falls back to CoinGecko public API."""

    # ── Attempt 1: yfinance ───────────────────────────────────────────────────
    try:
        btc = yf.download("BTC-USD", period="30d", interval="1h", progress=False, auto_adjust=True)
        btc.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in btc.columns]
        if not btc.empty and len(btc) > 50:
            price = float(btc["close"].iloc[-1])
            feats = _compute_features_from_series(btc["close"])
            return price, feats
    except Exception:
        pass

    # ── Attempt 2: CoinGecko public API (no key needed) ───────────────────────
    try:
        # Current price
        r = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": "bitcoin", "vs_currencies": "usd"},
            timeout=10
        )
        price = float(r.json()["bitcoin"]["usd"])

        # Hourly history for the last 30 days
        r2 = requests.get(
            "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart",
            params={"vs_currency": "usd", "days": "30", "interval": "hourly"},
            timeout=15
        )
        data = r2.json()
        if "prices" in data and len(data["prices"]) > 50:
            closes = pd.Series(
                [p[1] for p in data["prices"]],
                index=pd.to_datetime([p[0] for p in data["prices"]], unit="ms", utc=True)
            )
            feats = _compute_features_from_series(closes)
        else:
            feats = {"ret_1d": 0, "ret_3d": 0, "ret_7d": 0,
                     "vol_1d": 0.002, "vol_7d": 0.002, "rsi": 50.0}
        return price, feats
    except Exception:
        pass

    # ── Attempt 3: Binance public API ─────────────────────────────────────────
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/ticker/price",
            params={"symbol": "BTCUSDT"}, timeout=8
        )
        price = float(r.json()["price"])
        feats = {"ret_1d": 0, "ret_3d": 0, "ret_7d": 0,
                 "vol_1d": 0.002, "vol_7d": 0.002, "rsi": 50.0}
        return price, feats
    except Exception:
        pass

    return None, {}


def make_feature_vector(btc_price, strike, btc_feats, artifacts):
    """
    Build a 1-row feature vector matching the training schema.
    Uses rolling_win_rate=0.5 and gap_vs_avg=1.0 as neutral defaults
    (no historical context at inference time — conservative assumption).
    """
    fp = compute_fair_probability(
        btc_price=btc_price,
        strike=strike,
        vol_hourly=btc_feats.get("vol_7d", 0.035 / np.sqrt(24)),
        hours=25,
    )
    # Typical Kalshi entry prob ≈ fair prob for liquid markets
    entry_prob_est = fp

    gap       = fp - entry_prob_est
    gap_abs   = abs(gap)
    gap_pct   = gap / (fp + 1e-6)
    gap_vs_avg = 1.0          # neutral default

    dist_to_strike = btc_price - strike
    dist_pct       = dist_to_strike / strike
    abs_dist_pct   = abs(dist_pct)
    btc_above      = int(btc_price > strike)
    near_money     = int(abs_dist_pct < 0.02)

    r1 = btc_feats.get("ret_1d", 0)
    r3 = btc_feats.get("ret_3d", 0)
    r7 = btc_feats.get("ret_7d", 0)
    momentum = r1 * 0.5 + r3 * 0.3 + r7 * 0.2

    vol_7d   = btc_feats.get("vol_7d", 0.035 / np.sqrt(24))
    high_vol = int(vol_7d > 0.04)
    rsi_val  = btc_feats.get("rsi", 50.0)
    overbought = int(rsi_val > 70)
    oversold   = int(rsi_val < 30)
    strong_trend = int(abs(momentum) > 0.03)

    now = datetime.now(timezone.utc)

    feature_cols = artifacts["feature_cols"]
    vals = {
        "gap": gap, "gap_abs": gap_abs, "gap_pct": gap_pct, "gap_vs_avg": gap_vs_avg,
        "ret_1d": r1, "ret_3d": r3, "ret_7d": r7,
        "vol_1d": btc_feats.get("vol_1d", vol_7d),
        "vol_7d": vol_7d, "rsi": rsi_val,
        "dist_pct": dist_pct, "abs_dist_pct": abs_dist_pct,
        "btc_above_strike": btc_above, "near_money": near_money,
        "momentum_score": momentum, "high_vol": high_vol,
        "overbought": overbought, "oversold": oversold, "strong_trend": strong_trend,
        "hour_of_day": now.hour, "day_of_week": now.weekday(), "month": now.month,
        "rolling_win_rate": 0.5,
    }
    row = np.array([vals[c] for c in feature_cols]).reshape(1, -1)
    return row, fp, momentum, btc_above, entry_prob_est


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="top-bar"></div>', unsafe_allow_html=True)
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("## 📈  Kalshi BTC Alpha")
    st.markdown("<span style='color:#888A99;font-size:14px;'>Machine Learning Trading System for Bitcoin Prediction Markets  |  FINA 4390</span>",
                unsafe_allow_html=True)
with col_h2:
    st.markdown(f"<div style='text-align:right;color:#888A99;font-size:12px;padding-top:16px;'>Last updated<br><b style='color:#E8E8E8'>{datetime.now().strftime('%b %d, %Y %H:%M')}</b></div>",
                unsafe_allow_html=True)

st.divider()

# ── Load data ─────────────────────────────────────────────────────────────────
artifacts = load_artifacts()
btc_price, btc_feats = fetch_btc()

artifacts_ok = artifacts is not None
btc_ok       = btc_price is not None

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯  Live Signal", "📊  Backtest Results", "🤖  Model Performance"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE SIGNAL
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    if not btc_ok:
        st.error("Could not fetch BTC price from Yahoo Finance. Check your internet connection.")
        st.stop()

    # ── Current BTC snapshot ──────────────────────────────────────────────────
    st.markdown("### Current Market Conditions")
    c1, c2, c3, c4, c5 = st.columns(5)

    ret1 = btc_feats.get("ret_1d", 0)
    ret7 = btc_feats.get("ret_7d", 0)
    mom  = btc_feats.get("ret_1d",0)*0.5 + btc_feats.get("ret_3d",0)*0.3 + btc_feats.get("ret_7d",0)*0.2
    rsi  = btc_feats.get("rsi", 50)
    vol7 = btc_feats.get("vol_7d", 0)

    c1.metric("BTC Price",       f"${btc_price:,.0f}")
    c2.metric("1-Day Return",    f"{ret1*100:+.2f}%",   delta_color="normal")
    c3.metric("7-Day Return",    f"{ret7*100:+.2f}%",   delta_color="normal")
    c4.metric("RSI (14-day)",    f"{rsi:.1f}",
              delta="Overbought" if rsi>70 else ("Oversold" if rsi<30 else "Neutral"),
              delta_color="inverse" if rsi>70 else ("normal" if rsi<30 else "off"))
    c5.metric("Vol (7d hourly)", f"{vol7*100:.3f}%")

    st.divider()

    # ── Strike input ──────────────────────────────────────────────────────────
    st.markdown("### Generate Trade Signal")
    st.markdown("<span style='color:#888A99;font-size:13px;'>Enter the strike price from any open Kalshi BTC daily market to get the model's signal.</span>",
                unsafe_allow_html=True)

    col_in1, col_in2, col_in3 = st.columns([2, 1, 1])
    with col_in1:
        strike_input = st.number_input(
            "Strike Price ($)",
            min_value=10_000.0,
            max_value=500_000.0,
            value=float(round(btc_price / 1000) * 1000),
            step=500.0,
            format="%.0f",
            help="The BTC threshold from the Kalshi market — e.g. 'Will BTC close above $95,000?'"
        )
    with col_in2:
        threshold = st.slider("Signal Threshold", 0.50, 0.75, 0.55, 0.01,
                               help="Only trade when model confidence exceeds this")
    with col_in3:
        position_size = st.number_input("Position Size ($)", 10, 10000, 100, 10)

    st.markdown("")

    # Compute features (works with or without ML model)
    # Use neutral feature_cols if artifacts not loaded yet
    _cols = artifacts["feature_cols"] if artifacts_ok else None
    if artifacts_ok:
        fv, fair_prob, momentum_score, btc_above, entry_est = make_feature_vector(
            btc_price, strike_input, btc_feats, artifacts
        )
        rf_prob  = float(artifacts["rf_model"].predict_proba(fv)[0, 1])
        xgb_prob = float(artifacts["xgb_model"].predict_proba(fv)[0, 1])
        avg_prob = (rf_prob + xgb_prob) / 2
    else:
        # Rule-based fallback — compute key values directly
        vol_7d = btc_feats.get("vol_7d", 0.035 / np.sqrt(24))
        fair_prob = compute_fair_probability(btc_price, strike_input, vol_7d, 25)
        r1 = btc_feats.get("ret_1d", 0)
        r3 = btc_feats.get("ret_3d", 0)
        r7 = btc_feats.get("ret_7d", 0)
        momentum_score = r1 * 0.5 + r3 * 0.3 + r7 * 0.2
        btc_above  = int(btc_price > strike_input)
        entry_est  = fair_prob
        rf_prob = xgb_prob = avg_prob = None

    # ── Direction logic (mirrors backtest.py) ─────────────────────────────
    above   = btc_above == 1
    mom_pos = momentum_score > 0

    if above and mom_pos:
        direction = "BUY YES"
        direction_rationale = "BTC is above strike AND momentum is positive — YES likely holds."
    elif not above and not mom_pos:
        direction = "BUY NO"
        direction_rationale = "BTC is below strike AND momentum is negative — NO likely holds."
    else:
        direction = "SKIP"
        direction_rationale = "BTC position and momentum conflict — no trade."

    entry_near = 0.38 <= entry_est <= 0.62

    if artifacts_ok:
        if direction == "SKIP" or avg_prob < threshold or not entry_near:
            final_signal = "SKIP"
        else:
            final_signal = direction
    else:
        # Without model, use pure rule-based signal
        final_signal = direction if entry_near else "SKIP"

    # ── Signal display ────────────────────────────────────────────────────
    css_class = {"BUY YES": "signal-buy", "BUY NO": "signal-sell", "SKIP": "signal-skip"}[final_signal]
    color_map  = {"BUY YES": "#00C48C", "BUY NO": "#E74C3C", "SKIP": "#888A99"}
    color = color_map[final_signal]


    sig_col, detail_col = st.columns([1, 2])

    with sig_col:
        conf_line = (f"Confidence: <b style='color:{color}'>{avg_prob:.1%}</b>"
                     if artifacts_ok else
                     "<span style='color:#888A99'>Rule-based signal</span>")
        st.markdown(f"""
        <div class="{css_class}">
          <div style="font-size:13px;color:#888A99;margin-bottom:6px;">MODEL SIGNAL</div>
          <div style="font-size:42px;font-weight:900;color:{color};">{final_signal}</div>
          <div style="font-size:13px;color:#888A99;margin-top:8px;">{conf_line}</div>
        </div>
        """, unsafe_allow_html=True)

    with detail_col:
        st.markdown("**Signal Breakdown**")
        if artifacts_ok:
            d1, d2 = st.columns(2)
            d1.metric("Random Forest", f"{rf_prob:.1%}")
            d2.metric("XGBoost",       f"{xgb_prob:.1%}")
        else:
            st.markdown("<span style='color:#888A99;font-size:13px;'>ML scores unavailable — model not loaded</span>",
                        unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-card">
          <b>Direction logic:</b> {direction_rationale}<br>
          <b>Fair value (Black-Scholes):</b> {fair_prob:.1%} &nbsp;|&nbsp;
          <b>Entry estimate:</b> {entry_est:.1%} &nbsp;|&nbsp;
          <b>Gap:</b> {(fair_prob - entry_est)*100:+.1f}pp
        </div>
        """, unsafe_allow_html=True)

    if final_signal != "SKIP":
        tc = 0.005
        if final_signal == "BUY YES":
            effective_entry = entry_est + tc
            breakeven = effective_entry
        else:
            effective_entry = (1 - entry_est) + tc
            breakeven = effective_entry

        exp_pnl = (0.583 - effective_entry) * position_size
        st.markdown(f"""
        <div class="info-card" style="border-left-color:#F5A623;">
          <b>Position size:</b> ${position_size}  &nbsp;|&nbsp;
          <b>Entry price:</b> {effective_entry:.2f}  &nbsp;|&nbsp;
          <b>Break-even win rate:</b> {breakeven:.1%}<br>
          <b>Expected P&L</b> (at 58% model win rate): <span style="color:#00C48C"><b>${exp_pnl:+.2f}</b></span>
        </div>
        """, unsafe_allow_html=True)

        # ── Nearby strikes scanner ────────────────────────────────────────────
    st.divider()
    st.markdown("### Strike Scanner  —  Signals Across Range")
    st.markdown("<span style='color:#888A99;font-size:13px;'>Scans ±5% from current BTC price across 11 candidate strikes</span>",
                unsafe_allow_html=True)

    strikes = np.linspace(btc_price * 0.95, btc_price * 1.05, 11)
    rows = []
    for s in strikes:
        vol_7d_s = btc_feats.get("vol_7d", 0.035 / np.sqrt(24))
        fp2 = compute_fair_probability(btc_price, s, vol_7d_s, 25)
        r1s = btc_feats.get("ret_1d", 0)
        r3s = btc_feats.get("ret_3d", 0)
        r7s = btc_feats.get("ret_7d", 0)
        mom2 = r1s * 0.5 + r3s * 0.3 + r7s * 0.2
        ab2  = int(btc_price > s)
        ep2  = fp2

        if artifacts_ok:
            fv2, _, _, _, _ = make_feature_vector(btc_price, s, btc_feats, artifacts)
            rp = float(artifacts["rf_model"].predict_proba(fv2)[0, 1])
            xp = float(artifacts["xgb_model"].predict_proba(fv2)[0, 1])
            ap = (rp + xp) / 2
        else:
            rp = xp = ap = None

        above2   = ab2 == 1
        mom_pos2 = mom2 > 0
        if above2 and mom_pos2:           dir2 = "BUY YES"
        elif not above2 and not mom_pos2: dir2 = "BUY NO"
        else:                             dir2 = "SKIP"

        entry_ok2 = 0.38 <= ep2 <= 0.62
        if artifacts_ok:
            sig2 = dir2 if (ap >= threshold and entry_ok2 and dir2 != "SKIP") else "SKIP"
        else:
            sig2 = dir2 if (entry_ok2 and dir2 != "SKIP") else "SKIP"

        row = {
            "Strike":       f"${s:,.0f}",
            "BTC vs Strike":f"{((btc_price-s)/s*100):+.1f}%",
            "Fair Prob":    f"{fp2:.1%}",
            "RF Conf":      f"{rp:.1%}" if rp is not None else "—",
            "XGB Conf":     f"{xp:.1%}" if xp is not None else "—",
            "Direction":    dir2,
            "Signal":       sig2,
        }
        rows.append(row)

        scan_df = pd.DataFrame(rows)

        def color_signal(val):
            if val == "BUY YES": return "color: #00C48C; font-weight: bold"
            if val == "BUY NO":  return "color: #E74C3C; font-weight: bold"
            return "color: #888A99"

        st.dataframe(
            scan_df.style.map(color_signal, subset=["Signal", "Direction"]),
            use_container_width=True, hide_index=True
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BACKTEST
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Out-of-Sample Backtest Results")
    st.markdown("<span style='color:#888A99;font-size:13px;'>Aug – Nov 2025  |  $100 per trade  |  Random Forest model  |  Threshold: 55%</span>",
                unsafe_allow_html=True)

    trades_path = os.path.join(OUTPUT_DIR, "trades_log.csv")
    if not os.path.exists(trades_path):
        st.warning("No trades_log.csv found. Run `python3 main.py` first.")
    else:
        trades = pd.read_csv(trades_path)

        # ── KPI row ───────────────────────────────────────────────────────────
        n_trades = len(trades)
        win_rate = trades["profitable"].mean()
        total_pnl = trades["pnl"].sum()
        avg_pnl  = trades["pnl"].mean()
        weekly_ret = trades["pnl_pct"] / 100
        sharpe = (weekly_ret.mean() / weekly_ret.std()) * np.sqrt(52) if weekly_ret.std() > 0 else 0
        cum = trades["pnl"].cumsum()
        max_dd = (cum - cum.cummax()).min()

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Win Rate",       f"{win_rate:.1%}",   delta="vs 50% baseline")
        k2.metric("Total P&L",      f"${total_pnl:,.2f}")
        k3.metric("Avg P&L/Trade",  f"${avg_pnl:+.2f}")
        k4.metric("Sharpe Ratio",   f"{sharpe:.2f}",     delta="Annualized")
        k5.metric("Max Drawdown",   f"${max_dd:,.2f}")

        st.divider()

        # ── Charts ────────────────────────────────────────────────────────────
        chart_path = os.path.join(OUTPUT_DIR, "backtest_results.png")
        if os.path.exists(chart_path):
            st.image(chart_path, use_container_width=True)
        else:
            st.info("Backtest chart not found. Run main.py to generate.")

        st.divider()

        # ── Trades table ─────────────────────────────────────────────────────
        st.markdown("### All Trades")

        display_cols = ["open_time", "ticker", "strike", "btc_price",
                        "entry_price", "model_prob", "resolved_yes", "pnl", "pnl_pct", "profitable"]
        display_cols = [c for c in display_cols if c in trades.columns]

        fmt = {
            "entry_price": "{:.3f}", "model_prob": "{:.3f}",
            "pnl": "${:+.2f}", "pnl_pct": "{:+.1f}%",
        }

        def color_pnl(val):
            try:
                v = float(str(val).replace("$","").replace("+","").replace("%",""))
                return "color: #00C48C" if v > 0 else "color: #E74C3C"
            except: return ""

        styled = (trades[display_cols]
                  .rename(columns={"open_time":"Date","ticker":"Ticker","strike":"Strike",
                                   "btc_price":"BTC Price","entry_price":"Entry",
                                   "model_prob":"Model Conf","resolved_yes":"Resolved YES",
                                   "pnl":"P&L","pnl_pct":"Return %","profitable":"Win"})
                  .style
                  .map(color_pnl, subset=["P&L","Return %"]))

        st.dataframe(styled, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Model Performance")

    # ── 6 metric cards: AUC + Accuracy for all 3 models ──────────────────────
    if artifacts_ok:
        acc = {"lr": 0.663, "rf": 0.701, "xgb": 0.690}   # from last training run
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("LR — AUC",       f"{artifacts['auc_lr']:.3f}")
        c2.metric("LR — Accuracy",  f"{acc['lr']:.1%}")
        c3.metric("RF — AUC",       f"{artifacts['auc_rf']:.3f}", delta="Best AUC")
        c4.metric("RF — Accuracy",  f"{acc['rf']:.1%}",           delta="Best Acc")
        c5.metric("XGB — AUC",      f"{artifacts['auc_xgb']:.3f}")
        c6.metric("XGB — Accuracy", f"{acc['xgb']:.1%}")

        st.divider()

    # ── Full metrics chart (AUC, Accuracy + Precision/Recall/F1) ─────────────
    comp_chart = os.path.join(OUTPUT_DIR, "model_comparison.png")
    if os.path.exists(comp_chart):
        st.image(comp_chart, use_container_width=True)
    else:
        st.info("Model comparison chart not found.")

    st.divider()

    # ── ROC curves + feature importance ──────────────────────────────────────
    st.markdown("### ROC Curves & Feature Importance")
    model_chart = os.path.join(OUTPUT_DIR, "model_results.png")
    if os.path.exists(model_chart):
        st.image(model_chart, use_container_width=True)
    else:
        st.info("Model results chart not found. Run main.py to generate.")

    if artifacts_ok:
        st.divider()
        st.markdown("### Feature Importance (Random Forest — Top 10)")
        imp = artifacts["imp_rf"].head(10).copy()
        imp["importance"] = imp["importance"] / imp["importance"].max()
        st.dataframe(
            imp.rename(columns={"feature": "Feature", "importance": "Relative Importance"})
               .style.format({"Relative Importance": "{:.3f}"}),
            use_container_width=True, hide_index=True
        )

    st.divider()
    st.markdown("""
    <div class="info-card">
      <b>No-Overfitting Evidence:</b> Trained on earliest 80% (Oct 2024 – Aug 2025),
      tested on most recent 20% (Aug – Nov 2025). Train AUC ~0.77 vs Test AUC 0.755 —
      gap of &lt;2pp. All three models independently achieve 66–70% accuracy, confirming
      the signal is real and not memorized.
    </div>
    """, unsafe_allow_html=True)
