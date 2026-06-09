"""
AlphaTrend Backtest Dashboard — Streamlit
==========================================
Run locally:
    pip install streamlit plotly yfinance pandas numpy
    streamlit run backtest_dashboard.py

Opens at http://localhost:8501

Same strategy as alphatrend_backtest_per_ticker.ipynb:
  - Configurable timeframe: 1H (native), 4H (time-aligned resample), 1D (native).
    For 4H we resample yfinance 1h bars onto a proper 4H clock grid so bars
    line up with TradingView's 4H bars (especially for 24/7 instruments).
  - Long-only. BUY/SELL on AlphaTrend crossover, fills at next bar's open.
  - Shared portfolio cash balance, compounding.
  - Max 6 concurrent positions, total notional <= 1.8 x balance.
  - Each new entry sized at balance * (MAX_LEVERAGE / MAX_POSITIONS) of the
    current balance, clipped if the leverage cap would be breached.
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")
for _log in ("yfinance", "peewee", "urllib3", "requests"):
    logging.getLogger(_log).setLevel(logging.CRITICAL)

st.set_page_config(page_title="AlphaTrend Backtest", layout="wide")


# ──────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_bars(ticker: str, days: int, timeframe: str) -> pd.DataFrame:
    """Fetch OHLCV and return bars at the requested timeframe.

    - timeframe="1h" or "4h": fetch yfinance 1h native. For 4h, resample onto
      a proper 4-hour clock grid (00:00, 04:00, 08:00 UTC, ...). yfinance caps
      1h history at ~730 days.
    - timeframe="1d": fetch yfinance 1d native, no resample.
    """
    end = pd.Timestamp.utcnow().tz_localize(None)
    start = end - pd.Timedelta(days=days)
    native = "1h" if timeframe in ("1h", "4h") else timeframe
    df = yf.download(
        ticker, start=start, end=end, interval=native,
        auto_adjust=True, progress=False, threads=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    if timeframe == "4h":
        # Time-based resample. label/closed="left" → bar 09:00 contains
        # the hourly bars from 09:00..12:00 inclusive.
        df = df.resample("4h", label="left", closed="left").agg({
            "Open":   "first",
            "High":   "max",
            "Low":    "min",
            "Close":  "last",
            "Volume": "sum",
        }).dropna(subset=["Open", "High", "Low", "Close"])
    return df


# ──────────────────────────────────────────────────────────────────────────
# AlphaTrend (faithful Pine v5 port)
# ──────────────────────────────────────────────────────────────────────────
def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    tr.iloc[0] = np.nan
    return tr


def mfi(df: pd.DataFrame, period: int) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    raw_mf = tp * df["Volume"]
    delta = tp.diff()
    pos_mf = raw_mf.where(delta > 0, 0.0)
    neg_mf = raw_mf.where(delta < 0, 0.0)
    mfr = pos_mf.rolling(period).sum() / neg_mf.rolling(period).sum().replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + mfr))


def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    dn = (-delta).clip(lower=0)
    avg_up = up.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_dn = dn.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_up / avg_dn.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def alphatrend(df: pd.DataFrame, coeff: float, ap: int, use_mfi: bool) -> pd.Series:
    tr  = true_range(df)
    atr = tr.rolling(ap).mean()
    up_t   = df["Low"]  - atr * coeff
    down_t = df["High"] + atr * coeff
    osc = mfi(df, ap) if use_mfi else rsi(df["Close"], ap)

    out  = np.full(len(df), np.nan)
    prev = 0.0
    for i in range(len(df)):
        ut, dt = up_t.iloc[i], down_t.iloc[i]
        if np.isnan(ut) or np.isnan(dt):
            continue
        o = osc.iloc[i]
        bullish = (not np.isnan(o)) and (o >= 50)
        val = (prev if ut < prev else ut) if bullish else (prev if dt > prev else dt)
        out[i] = val
        prev = val
    return pd.Series(out, index=df.index, name="AT")


def compute_signals(df: pd.DataFrame, coeff: float, ap: int, use_mfi: bool) -> pd.DataFrame:
    at = alphatrend(df, coeff, ap, use_mfi)
    df = df.copy()
    df["AT"] = at
    a0, a2 = at, at.shift(2)
    a1, a3 = at.shift(1), at.shift(3)
    df["BuySig"]  = ((a0 > a2) & (a1 <= a3)).fillna(False)
    df["SellSig"] = ((a0 < a2) & (a1 >= a3)).fillna(False)
    return df


# ──────────────────────────────────────────────────────────────────────────
# Portfolio simulator
# ──────────────────────────────────────────────────────────────────────────
def build_events(per_ticker: Dict[str, pd.DataFrame]) -> List[Tuple]:
    events = []
    for ticker, df in per_ticker.items():
        opens = df["Open"].values
        idx   = df.index
        bs    = df["BuySig"].values
        ss    = df["SellSig"].values
        for i in range(len(df) - 1):
            px = float(opens[i + 1])
            if np.isnan(px):
                continue
            if bs[i]:
                events.append((idx[i + 1], "BUY",  ticker, px))
            if ss[i]:
                events.append((idx[i + 1], "SELL", ticker, px))
    events.sort(key=lambda e: (e[0], 0 if e[1] == "SELL" else 1, e[2]))
    return events


def portfolio_simulate(events, per_ticker, *,
                       start_balance, max_leverage, max_positions,
                       comm, slippage):
    balance      = start_balance
    open_pos     = {}
    trades       = []
    skipped      = []
    per_pos_frac = max_leverage / max_positions

    first_time = events[0][0] if events else None
    equity_curve = [(first_time, balance, 0)]

    def total_notional():
        return sum(p["notional"] for p in open_pos.values())

    for time, kind, ticker, price in events:
        if kind == "SELL":
            if ticker not in open_pos:
                continue
            pos        = open_pos.pop(ticker)
            eff_exit   = price * (1.0 - slippage)
            ret        = eff_exit / pos["entry_price"] - 1.0
            gross_pnl  = ret * pos["notional"]
            exit_comm  = comm * pos["notional"] * (eff_exit / pos["entry_price"])
            total_comm = pos["entry_comm"] + exit_comm
            net_pnl    = gross_pnl - total_comm
            balance   += net_pnl

            trades.append({
                "ticker":           ticker,
                "entry_time":       pos["entry_time"],
                "entry_price":      pos["entry_price"],
                "exit_time":        time,
                "exit_price":       eff_exit,
                "notional":         pos["notional"],
                "balance_at_entry": pos["balance_at_entry"],
                "return_pct":       ret * 100.0,
                "gross_pnl":        gross_pnl,
                "commission":       total_comm,
                "net_pnl":          net_pnl,
                "balance_after":    balance,
            })
            equity_curve.append((time, balance, len(open_pos)))

        elif kind == "BUY":
            if ticker in open_pos:
                continue
            if balance <= 0:
                skipped.append((time, ticker, "balance_depleted"))
                continue
            if len(open_pos) >= max_positions:
                skipped.append((time, ticker, "max_positions"))
                continue
            headroom = balance * max_leverage - total_notional()
            full     = balance * per_pos_frac
            notional = min(full, headroom)
            if notional <= 0:
                skipped.append((time, ticker, "leverage_cap"))
                continue

            eff_entry  = price * (1.0 + slippage)
            entry_comm = comm * notional
            open_pos[ticker] = {
                "entry_time":       time,
                "entry_price":      eff_entry,
                "notional":         notional,
                "entry_comm":       entry_comm,
                "balance_at_entry": balance,
            }
            equity_curve.append((time, balance, len(open_pos)))

    open_trades = []
    for ticker, pos in open_pos.items():
        last_close = float(per_ticker[ticker]["Close"].iloc[-1])
        ret        = last_close / pos["entry_price"] - 1.0
        exit_comm  = comm * pos["notional"] * (last_close / pos["entry_price"])
        unreal_net = ret * pos["notional"] - pos["entry_comm"] - exit_comm
        open_trades.append({
            "ticker":             ticker,
            **pos,
            "last_price":         last_close,
            "unrealized_pct":     ret * 100.0,
            "unrealized_net_pnl": unreal_net,
        })
    projected_balance = balance + sum(t["unrealized_net_pnl"] for t in open_trades)
    return trades, skipped, equity_curve, open_trades, projected_balance


# ──────────────────────────────────────────────────────────────────────────
# Sidebar — inputs
# ──────────────────────────────────────────────────────────────────────────
st.title("AlphaTrend Backtest")
st.caption("Configurable timeframe • long-only • shared portfolio cash • compounding • max 6 positions / 1.8× leverage")

with st.sidebar:
    st.header("Tickers")
    default_tickers = "AAPL\nMSFT\nNVDA\nTSLA\nAMD\nMETA\nGOOGL\nBTC-USD"
    tickers_text = st.text_area(
        "One ticker per line", value=default_tickers, height=200,
        help="yfinance symbols: AAPL, BTC-USD, EURUSD=X, ^GSPC, THYAO.IS",
    )

    st.header("Backtest window")
    timeframe = st.selectbox(
        "Timeframe", ["1d", "1h", "4h"], index=0,
        help="1D and 1H use yfinance native intervals and match TradingView "
             "bars exactly. 4H uses time-aligned resampling of 1H data — "
             "matches TradingView for 24/7 crypto, but US stocks will differ "
             "because of session vs. clock alignment.",
    )
    history_d = st.slider("History (days)", min_value=30, max_value=720, value=365, step=30)

    st.header("Portfolio")
    start_bal     = st.number_input("Starting balance ($)", value=10_000.0, min_value=100.0, step=1000.0)
    max_leverage  = st.number_input("Max leverage", value=1.8, min_value=0.1, max_value=10.0, step=0.1)
    max_positions = st.number_input("Max concurrent positions", value=6, min_value=1, max_value=50, step=1)
    st.caption(f"Per-entry sizing: balance × {max_leverage / max_positions:.3f}")

    st.header("AlphaTrend")
    coeff   = st.number_input("Multiplier (coeff)", value=1.0, min_value=0.1, max_value=10.0, step=0.1)
    ap      = st.number_input("Common Period (AP)", value=14, min_value=2, max_value=200, step=1)
    use_mfi = st.checkbox("Use MFI (volume mode)", value=True,
                          help="Uncheck for RSI mode — better for FX/indexes without reliable volume.")

    st.header("Costs")
    comm_pct     = st.number_input("Commission per side (%)", value=0.15, min_value=0.0, max_value=1.0, step=0.01)
    slippage_pct = st.number_input("Slippage per side (%)",   value=0.00, min_value=0.0, max_value=1.0, step=0.01)
    comm     = comm_pct / 100
    slippage = slippage_pct / 100

    run_btn = st.button("Run backtest", type="primary", use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────
# Main panel
# ──────────────────────────────────────────────────────────────────────────
if not run_btn:
    st.info("Configure on the left, then click **Run backtest**.")
    st.stop()

tickers = [t.strip().upper() for t in tickers_text.splitlines() if t.strip()]
if not tickers:
    st.error("No tickers provided.")
    st.stop()

# Fetch + compute signals
per_ticker: Dict[str, pd.DataFrame] = {}
fetch_status = []
prog = st.progress(0.0, text=f"Fetching {timeframe} data…")
for i, t in enumerate(tickers, start=1):
    bars = fetch_bars(t, history_d, timeframe)
    if bars.empty:
        fetch_status.append((t, 0, 0, 0, "no data"))
    elif len(bars) < ap + 5:
        fetch_status.append((t, len(bars), 0, 0, f"only {len(bars)} bars"))
    else:
        sig = compute_signals(bars, coeff, ap, use_mfi)
        per_ticker[t] = sig
        fetch_status.append((t, len(bars), int(sig.BuySig.sum()), int(sig.SellSig.sum()), "ok"))
    prog.progress(i / len(tickers), text=f"Fetching {timeframe} data… {t} ({i}/{len(tickers)})")
prog.empty()

if not per_ticker:
    st.error("No usable data for any ticker.")
    st.stop()

events = build_events(per_ticker)
trades, skipped, equity_curve, open_trades, projected_bal = portfolio_simulate(
    events, per_ticker,
    start_balance=start_bal, max_leverage=max_leverage, max_positions=int(max_positions),
    comm=comm, slippage=slippage,
)
trades_df = pd.DataFrame(trades)

final_bal     = equity_curve[-1][1]
realized_pct  = (final_bal / start_bal - 1.0) * 100.0
projected_pct = (projected_bal / start_bal - 1.0) * 100.0

# ── KPI row ─────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Starting balance", f"${start_bal:,.2f}")
c2.metric("Final cash",        f"${final_bal:,.2f}", f"{realized_pct:+.2f}%")
c3.metric("Projected balance", f"${projected_bal:,.2f}", f"{projected_pct:+.2f}%",
          help="Cash + unrealized P&L on still-open positions")
c4.metric("Closed trades",     f"{len(trades_df)}")
c5.metric("Skipped BUYs",      f"{len(skipped)}",
          help="Signals dropped because portfolio was at max positions or leverage cap")

# ── P&L chart ───────────────────────────────────────────────────────────
st.subheader("P&L — portfolio balance over time")
if len(equity_curve) <= 1:
    st.info("No closed trades — balance unchanged from starting value.")
else:
    times = [t for t, _, _ in equity_curve]
    bals  = [b for _, b, _ in equity_curve]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=bals, mode="lines+markers", name="Cash balance",
        line=dict(color="#00874c", width=2),
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>$%{y:,.2f}<extra></extra>",
    ))
    fig.add_hline(y=start_bal, line=dict(color="gray", dash="dash"),
                  annotation_text=f"Start ${start_bal:,.0f}", annotation_position="top left")
    if not trades_df.empty:
        m = trades_df[["exit_time", "ticker", "balance_after"]].copy()
        fig.add_trace(go.Scatter(
            x=m["exit_time"], y=m["balance_after"], mode="markers", name="Trade close",
            text=m["ticker"], hovertemplate="%{text} → $%{y:,.2f}<extra></extra>",
            marker=dict(symbol="circle", size=8, color="#0022FC",
                        line=dict(color="white", width=1)),
        ))
    fig.update_layout(
        height=460, margin=dict(l=40, r=20, t=20, b=30),
        xaxis_title="Time", yaxis_title="Balance ($)",
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Tabs: details ───────────────────────────────────────────────────────
tab_summary, tab_trades, tab_open, tab_diag, tab_data = st.tabs(
    ["Per-ticker summary", "Trade log", "Open positions", "Signal diagnostic", "Data status"]
)

with tab_summary:
    if trades_df.empty:
        st.info("No closed trades.")
    else:
        def summarize(g):
            n      = len(g)
            wins   = g[g["net_pnl"] > 0]
            losses = g[g["net_pnl"] <= 0]
            return pd.Series({
                "trades":       n,
                "wins":         len(wins),
                "losses":       len(losses),
                "win_rate_%":   round(100.0 * len(wins) / n, 1) if n else 0.0,
                "avg_win_$":    round(wins["net_pnl"].mean(), 2)   if len(wins) else 0.0,
                "avg_loss_$":   round(losses["net_pnl"].mean(), 2) if len(losses) else 0.0,
                "best_$":       round(g["net_pnl"].max(), 2),
                "worst_$":      round(g["net_pnl"].min(), 2),
                "total_net_$":  round(g["net_pnl"].sum(), 2),
                "total_comm_$": round(g["commission"].sum(), 2),
            })
        by_ticker = trades_df.groupby("ticker").apply(summarize)
        overall   = pd.DataFrame({"ALL": summarize(trades_df)}).T
        st.dataframe(pd.concat([by_ticker, overall]), use_container_width=True)

        if skipped:
            st.markdown("**Skipped BUY reasons**")
            reasons = pd.Series([r for _, _, r in skipped]).value_counts().rename("count")
            st.dataframe(reasons.to_frame(), use_container_width=False)

with tab_trades:
    if trades_df.empty:
        st.info("No closed trades.")
    else:
        cols = ["ticker", "entry_time", "entry_price", "exit_time", "exit_price",
                "notional", "balance_at_entry", "return_pct",
                "gross_pnl", "commission", "net_pnl", "balance_after"]
        disp = trades_df[cols].copy()
        disp["entry_time"] = pd.to_datetime(disp["entry_time"]).dt.strftime("%Y-%m-%d %H:%M")
        disp["exit_time"]  = pd.to_datetime(disp["exit_time"]).dt.strftime("%Y-%m-%d %H:%M")
        for c in ("entry_price", "exit_price", "notional", "balance_at_entry",
                  "return_pct", "gross_pnl", "commission", "net_pnl", "balance_after"):
            disp[c] = disp[c].round(2)
        st.dataframe(disp, use_container_width=True, hide_index=True)
        st.download_button(
            "Download CSV", data=disp.to_csv(index=False).encode("utf-8"),
            file_name="alphatrend_trades.csv", mime="text/csv",
        )

with tab_open:
    if not open_trades:
        st.info("No open positions at end of data.")
    else:
        rows = []
        for ot in open_trades:
            rows.append({
                "ticker":            ot["ticker"],
                "entry_time":        pd.to_datetime(ot["entry_time"]).strftime("%Y-%m-%d %H:%M"),
                "entry_price":       round(ot["entry_price"], 2),
                "last_price":        round(ot["last_price"], 2),
                "notional":          round(ot["notional"], 2),
                "balance_at_entry":  round(ot["balance_at_entry"], 2),
                "unrealized_%":      round(ot["unrealized_pct"], 2),
                "unrealized_$":      round(ot["unrealized_net_pnl"], 2),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with tab_diag:
    if timeframe == "4h":
        diag_note = (
            "These **4H bars** are time-aligned (`pandas.resample('4h')`), so for "
            "24/7 instruments like BTC-USD they match TradingView's 4H bars exactly. "
            "For US stocks, gaps in the trading day mean some 4H windows contain "
            "partial sessions — minor mismatches with TradingView's session-aware 4H "
            "bars are still possible."
        )
    elif timeframe == "1h":
        diag_note = "These **1H bars** are yfinance native and should match TradingView's 1H bars closely."
    else:
        diag_note = "These **1D bars** are yfinance native (daily close)."

    st.markdown(
        f"Pick a ticker to see every raw BUY/SELL signal our {timeframe.upper()} "
        f"series produced — plus the trades the simulator actually executed. "
        f"\n\n{diag_note}"
    )
    diag_ticker = st.selectbox("Ticker", list(per_ticker.keys()))
    if diag_ticker:
        d = per_ticker[diag_ticker]
        sig_buy  = d[d["BuySig"]]
        sig_sell = d[d["SellSig"]]

        c_a, c_b, c_c = st.columns(3)
        c_a.metric(f"{timeframe.upper()} bars",       f"{len(d)}")
        c_b.metric("Raw BUY signals",  f"{len(sig_buy)}")
        c_c.metric("Raw SELL signals", f"{len(sig_sell)}")

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
            name="Price", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=d.index, y=d["AT"], name="AlphaTrend",
            line=dict(color="#0022FC", width=2),
        ))
        # Raw signals (every crossover — what the simulator processes)
        if len(sig_buy):
            fig.add_trace(go.Scatter(
                x=sig_buy.index, y=sig_buy["Low"] * 0.995,
                mode="markers", name="Raw BUY signal",
                marker=dict(symbol="triangle-up", color="#0022FC", size=11,
                            line=dict(color="white", width=1)),
            ))
        if len(sig_sell):
            fig.add_trace(go.Scatter(
                x=sig_sell.index, y=sig_sell["High"] * 1.005,
                mode="markers", name="Raw SELL signal",
                marker=dict(symbol="triangle-down", color="#80000B", size=11,
                            line=dict(color="white", width=1)),
            ))
        # Executed trades (subset — signals that the portfolio actually acted on)
        if not trades_df.empty:
            tt = trades_df[trades_df["ticker"] == diag_ticker]
            if not tt.empty:
                fig.add_trace(go.Scatter(
                    x=tt["entry_time"], y=tt["entry_price"],
                    mode="markers", name="Executed BUY",
                    marker=dict(symbol="star", color="#00E60F", size=16,
                                line=dict(color="black", width=1)),
                ))
                fig.add_trace(go.Scatter(
                    x=tt["exit_time"], y=tt["exit_price"],
                    mode="markers", name="Executed SELL",
                    marker=dict(symbol="x", color="#FC0400", size=14,
                                line=dict(color="black", width=2)),
                ))
        # Open position (entered but not yet exited)
        for ot in open_trades:
            if ot["ticker"] == diag_ticker:
                fig.add_trace(go.Scatter(
                    x=[ot["entry_time"]], y=[ot["entry_price"]],
                    mode="markers", name="Open position (no exit yet)",
                    marker=dict(symbol="star-open", color="#00874c", size=18,
                                line=dict(color="black", width=2)),
                ))

        fig.update_layout(
            height=560, xaxis_rangeslider_visible=False,
            margin=dict(l=40, r=20, t=20, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
        st.plotly_chart(fig, use_container_width=True)

        # Why-was-this-signal-skipped table — useful for understanding gaps
        st.markdown("**All raw signals (chronological)** — green star = executed, otherwise reason it was skipped.")
        sig_rows = []
        executed_buy_times  = set(pd.to_datetime(trades_df[trades_df["ticker"] == diag_ticker]["entry_time"])) if not trades_df.empty else set()
        executed_sell_times = set(pd.to_datetime(trades_df[trades_df["ticker"] == diag_ticker]["exit_time"]))  if not trades_df.empty else set()
        open_entry_times = {pd.to_datetime(ot["entry_time"]) for ot in open_trades if ot["ticker"] == diag_ticker}

        # Signal on bar i fires at bar i+1 open. Mirror that mapping here.
        for i in range(len(d) - 1):
            t_fill = d.index[i + 1]
            if d["BuySig"].iloc[i]:
                t_fill_ts = pd.to_datetime(t_fill)
                if t_fill_ts in executed_buy_times:
                    note = "executed ✓"
                elif t_fill_ts in open_entry_times:
                    note = "executed (still open)"
                else:
                    # Match against skipped reasons recorded by the simulator
                    skip_reason = next(
                        (r for tt, tk, r in skipped if tk == diag_ticker and pd.to_datetime(tt) == t_fill_ts),
                        "already in position",
                    )
                    note = f"skipped: {skip_reason}"
                sig_rows.append({"fill_time": t_fill, "kind": "BUY",  "fill_price": float(d["Open"].iloc[i + 1]), "outcome": note})
            if d["SellSig"].iloc[i]:
                t_fill_ts = pd.to_datetime(t_fill)
                if t_fill_ts in executed_sell_times:
                    note = "executed ✓"
                else:
                    note = "skipped: no open position on this ticker"
                sig_rows.append({"fill_time": t_fill, "kind": "SELL", "fill_price": float(d["Open"].iloc[i + 1]), "outcome": note})

        if sig_rows:
            sdf = pd.DataFrame(sig_rows).sort_values("fill_time")
            sdf["fill_time"]  = pd.to_datetime(sdf["fill_time"]).dt.strftime("%Y-%m-%d %H:%M")
            sdf["fill_price"] = sdf["fill_price"].round(2)
            st.dataframe(sdf, use_container_width=True, hide_index=True)
        else:
            st.info("No raw signals on this ticker in the chosen window.")

with tab_data:
    bars_col = f"{timeframe.upper()}_bars"
    status_df = pd.DataFrame(
        fetch_status, columns=["ticker", bars_col, "raw_BUYs", "raw_SELLs", "status"],
    )
    st.dataframe(status_df, use_container_width=True, hide_index=True)
    st.caption(
        "yfinance caps 1H/4H history at ~730 days; 1D goes back several years. "
        "If a ticker shows '0 bars' or 'no data', try a shorter history window or check the symbol."
    )
