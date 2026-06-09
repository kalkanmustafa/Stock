"""
Stock Screener Dashboard — Streamlit
=====================================
Run locally:
    pip install streamlit plotly yfinance requests pandas numpy
    streamlit run dashboard.py

Run in Google Colab:
    !pip install streamlit pyngrok plotly yfinance -q
    !streamlit run dashboard.py &
    from pyngrok import ngrok
    public_url = ngrok.connect(8501)
    print(public_url)
"""

from __future__ import annotations

import io
import logging
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

# Silence yfinance and related loggers so their stderr output
# doesn't render as error paragraphs inside Streamlit.
for _log in ("yfinance", "peewee", "urllib3", "requests", "charset_normalizer"):
    logging.getLogger(_log).setLevel(logging.CRITICAL)

import json as _json
import os as _os
import uuid as _uuid

_NOTES_FILE = _os.path.join(_os.path.dirname(__file__), "screener_notes.json")

def _load_notes() -> list:
    try:
        with open(_NOTES_FILE, "r", encoding="utf-8") as f:
            return _json.load(f)
    except Exception:
        return []

def _save_notes(notes: list) -> None:
    try:
        with open(_NOTES_FILE, "w", encoding="utf-8") as f:
            _json.dump(notes, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Stock Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

def _inject_css():
    _css_path = _os.path.join(_os.path.dirname(__file__), "style.css")
    with open(_css_path, "r", encoding="utf-8") as _f:
        _css = _f.read()
    st.markdown(f"<style>{_css}</style>", unsafe_allow_html=True)

_inject_css()


# =============================================================================
# DEFAULT CONFIG
# =============================================================================
DEFAULT_CONFIG: dict = {
    "universe":          "both",
    "benchmarks":        {"SPY": "S&P 500", "QQQ": "Nasdaq 100"},
    "primary_benchmark": "SPY",
    "timeframes":        {"1M": 21, "3M": 63, "6M": 126, "12M": 252},
    "short_term_timeframes": {"2W": 10, "4W": 20},
    "timeframe_weights": {"1M": 0.10, "3M": 0.25, "6M": 0.30, "12M": 0.35},
    "factor_weights":    {"rs_rank": 0.35, "outperf_rank": 0.20, "trend_rank": 0.25, "momentum_rank": 0.20},
    "min_market_cap":    10_000_000_000,
    "ma_stack":          {"fast1": 21, "fast2": 30, "slow": 200},
    "require_ma_stack":  True,
    "min_price":         0,
    "min_avg_volume":    500_000,
    "must_beat_both_on": "6M",
    "resilience_windows": ["2W", "4W"],
    "resilience_top_n":  25,
    "sector_analysis":   True,
    "top_per_sector":    5,
    "sector_filter":     None,
    "top_n":             None,  # show all passing stocks
    "save_csv":          False,
    "csv_filename":      "strong_stocks_report.csv",
}


# =============================================================================
# SCREENER FUNCTIONS
# =============================================================================
def _clean_ticker_col(s: pd.Series) -> pd.Series:
    return (s.astype(str)
             .str.replace(r"\[.*?\]", "", regex=True)
             .str.strip()
             .str.replace(".", "-", regex=False))

def _pick_col(cols, *candidates):
    lower = {str(c).lower().strip(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None

def get_sp500_constituents() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    df = pd.read_html(io.StringIO(requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).text))[0]
    return pd.DataFrame({
        "ticker":   _clean_ticker_col(df[_pick_col(df.columns, "Symbol", "Ticker")]),
        "name":     df[_pick_col(df.columns, "Security", "Company")] if _pick_col(df.columns, "Security", "Company") else "",
        "sector":   df[_pick_col(df.columns, "GICS Sector", "Sector")] if _pick_col(df.columns, "GICS Sector", "Sector") else "Unknown",
        "industry": df[_pick_col(df.columns, "GICS Sub-Industry", "Industry")] if _pick_col(df.columns, "GICS Sub-Industry", "Industry") else "Unknown",
    })

def get_nasdaq100_constituents() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    tables = pd.read_html(io.StringIO(requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).text))
    best = next((t for t in tables if _pick_col(t.columns, "Ticker", "Symbol")), None)
    if best is None:
        raise RuntimeError("Could not locate Nasdaq-100 ticker table.")
    return pd.DataFrame({
        "ticker":   _clean_ticker_col(best[_pick_col(best.columns, "Ticker", "Symbol")]),
        "name":     best[_pick_col(best.columns, "Company", "Security")] if _pick_col(best.columns, "Company", "Security") else "",
        "sector":   best[_pick_col(best.columns, "GICS Sector", "Sector")] if _pick_col(best.columns, "GICS Sector", "Sector") else "Unknown",
        "industry": best[_pick_col(best.columns, "GICS Sub-Industry", "Industry")] if _pick_col(best.columns, "GICS Sub-Industry", "Industry") else "Unknown",
    })

def get_russell2000_constituents() -> pd.DataFrame:
    """
    Fetch all US-listed stocks from the NASDAQ public screener API.
    This covers NYSE + NASDAQ + NYSE American, ~7 000 stocks.
    Combined with the $5B market-cap filter it approximates the larger
    Russell 2000 / small-mid cap universe without needing any API key.
    Stocks already in S&P 500 / Nasdaq 100 are excluded to avoid overlap
    when this is combined with the 'both' universe.
    """
    url = (
        "https://api.nasdaq.com/api/screener/stocks"
        "?tableonly=true&limit=10000&offset=0&download=true"
    )
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.nasdaq.com/",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        rows = resp.json()["data"]["rows"]
        df = pd.DataFrame(rows)

        # Keep only US-domiciled common stocks with a valid ticker
        df = df[df["country"] == "United States"].copy()
        df["symbol"] = df["symbol"].astype(str).str.strip()
        df = df[df["symbol"].str.match(r"^[A-Z]{1,5}$", na=False)]

        # Map market cap string → float for pre-filtering
        df["marketCap"] = pd.to_numeric(df["marketCap"], errors="coerce").fillna(0)

        # Exclude micro-caps (< $300 M) — definitely not Russell 2000 territory
        df = df[df["marketCap"] >= 300_000_000]

        # Exclude stocks already in S&P 500 / Nasdaq 100 (avoid overlap with 'both')
        try:
            sp_tickers  = set(get_sp500_constituents()["ticker"])
            nd_tickers  = set(get_nasdaq100_constituents()["ticker"])
            big_idx     = sp_tickers | nd_tickers
            df = df[~df["symbol"].isin(big_idx)]
        except Exception:
            pass  # if Wikipedia is down just keep all

        df["sector"]   = df["sector"].fillna("Unknown")
        df["industry"] = df["industry"].fillna("Unknown")
        df["name"]     = df["name"].fillna("")

        return pd.DataFrame({
            "ticker":   df["symbol"].values,
            "name":     df["name"].values,
            "sector":   df["sector"].values,
            "industry": df["industry"].values,
        })
    except Exception as e:
        st.warning(f"Could not load small/mid-cap universe: {e}")
        return pd.DataFrame(columns=["ticker", "name", "sector", "industry"])

def resolve_universe(universe) -> pd.DataFrame:
    if isinstance(universe, (list, tuple, set)):
        tickers = list(dict.fromkeys(universe))
        return pd.DataFrame({"ticker": tickers, "name": "", "sector": "Unknown", "industry": "Unknown"})
    key = str(universe).lower()
    if key == "sp500":       return get_sp500_constituents()
    if key == "nasdaq100":   return get_nasdaq100_constituents()
    if key == "russell2000": return get_russell2000_constituents()
    if key == "both":
        sp = get_sp500_constituents()
        nd = get_nasdaq100_constituents()
        return pd.concat([sp, nd], ignore_index=True).drop_duplicates(subset=["ticker"], keep="first").reset_index(drop=True)
    if key == "all":
        sp  = get_sp500_constituents()
        nd  = get_nasdaq100_constituents()
        r2k = get_russell2000_constituents()
        return pd.concat([sp, nd, r2k], ignore_index=True).drop_duplicates(subset=["ticker"], keep="first").reset_index(drop=True)
    raise ValueError(f"Unknown universe: {universe!r}")

def _extract_ticker_ohlcv(data, ticker, batch_len):
    """Extract Close/Volume for one ticker from a yf.download result.
    Handles both old (ticker, field) and new (field, ticker) MultiIndex layouts,
    as well as single-ticker flat DataFrames.
    """
    try:
        cols = data.columns
        # Flat columns — single ticker download
        if not isinstance(cols, pd.MultiIndex):
            return data[["Close", "Volume"]].copy()

        top = [str(c).strip() for c in cols.get_level_values(0)]
        # New yfinance layout: top level = field name (Close, Open, …)
        if "Close" in top:
            close  = data["Close"][ticker] if ticker in data["Close"].columns else data["Close"].iloc[:, 0]
            volume = data["Volume"][ticker] if ticker in data["Volume"].columns else data["Volume"].iloc[:, 0]
            return pd.DataFrame({"Close": close, "Volume": volume})
        # Old yfinance layout: top level = ticker symbol
        return data[ticker][["Close", "Volume"]].copy()
    except Exception:
        return None

def fetch_history(tickers, lookback_days=500, batch_size=50, progress_cb=None):
    end = datetime.today()
    start = end - timedelta(days=lookback_days)
    out = {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i: i + batch_size]
        if progress_cb:
            progress_cb(i, len(tickers), f"Fetching prices {i+1}–{i+len(batch)} / {len(tickers)}")
        try:
            data = yf.download(batch, start=start, end=end, auto_adjust=True,
                               progress=False, group_by="ticker", threads=True)
        except Exception:
            continue
        for t in batch:
            try:
                df = _extract_ticker_ohlcv(data, t, len(batch))
                if df is None:
                    continue
                df = df.dropna()
                if len(df) >= 60:
                    out[t] = df
            except Exception:
                continue
    return out

def fetch_market_caps(tickers, max_workers=20, progress_cb=None):
    def _one(t):
        try:
            tk = yf.Ticker(t)
            for key in ("market_cap", "marketCap"):
                try:
                    val = tk.fast_info[key]
                    if val: return t, float(val)
                except Exception: pass
            try:
                val = tk.fast_info.market_cap
                if val: return t, float(val)
            except Exception: pass
            try:
                val = tk.info.get("marketCap")
                if val: return t, float(val)
            except Exception: pass
            return t, None
        except Exception:
            return t, None

    out = {}
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_one, t) for t in tickers]
        for f in as_completed(futures):
            t, mc = f.result()
            out[t] = mc
            done += 1
            if progress_cb and (done % 50 == 0 or done == len(tickers)):
                progress_cb(done, len(tickers), f"Market caps {done}/{len(tickers)}")
    return out

def rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False, min_periods=period).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False, min_periods=period).mean()
    return 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

_TF_OFFSETS = {
    "1M":  pd.DateOffset(months=1),
    "3M":  pd.DateOffset(months=3),
    "6M":  pd.DateOffset(months=6),
    "12M": pd.DateOffset(months=12),
    "2W":  pd.DateOffset(weeks=2),
    "4W":  pd.DateOffset(weeks=4),
}

def _base_price(series, target_date):
    pos = series.index.searchsorted(target_date, side="right") - 1
    return float(series.iloc[pos]) if pos >= 0 else np.nan

def compute_ticker_factors(df, benchmarks_df, timeframes, ma_stack=None):
    ma_stack = ma_stack or {"fast1": 21, "fast2": 30, "slow": 200}
    p1, p2, ps = int(ma_stack["fast1"]), int(ma_stack["fast2"]), int(ma_stack["slow"])
    close, volume = df["Close"], df["Volume"]
    last = float(close.iloc[-1])

    out = {"price": last, "avg_vol_50": float(volume.tail(50).mean()), "n_bars": int(len(close))}

    sma1 = close.rolling(p1).mean().iloc[-1] if len(close) >= p1 else np.nan
    sma2 = close.rolling(p2).mean().iloc[-1] if len(close) >= p2 else np.nan
    smas = close.rolling(ps).mean().iloc[-1] if len(close) >= ps else np.nan

    trend = 0
    if not np.isnan(sma1): trend += int(last > sma1)
    if not np.isnan(sma2): trend += int(last > sma2)
    if not np.isnan(smas): trend += int(last > smas)
    if not np.isnan(smas) and not np.isnan(sma1): trend += int(sma1 > smas)
    if not np.isnan(smas) and not np.isnan(sma2): trend += int(sma2 > smas)
    out["trend_score_raw"] = trend
    out["ma_stack_ok"] = bool(not np.isnan(smas) and not np.isnan(sma1) and not np.isnan(sma2) and sma1 > smas and sma2 > smas)
    out[f"sma_{p1}"] = float(sma1) if not np.isnan(sma1) else np.nan
    out[f"sma_{p2}"] = float(sma2) if not np.isnan(sma2) else np.nan
    out[f"sma_{ps}"] = float(smas) if not np.isnan(smas) else np.nan

    rsi14 = float(rsi(close, 14).iloc[-1])
    hi_52w = float(close.tail(252).max())
    pct_from_high = last / hi_52w - 1.0
    mom = 0
    if not np.isnan(rsi14):
        if 50 <= rsi14 <= 70: mom += 1
        if 55 <= rsi14 <= 70: mom += 1
    if pct_from_high >= -0.10: mom += 1
    out["rsi_14"] = rsi14
    out["pct_from_52w_hi"] = pct_from_high
    out["momentum_score_raw"] = mom

    last_date = close.index[-1]

    for label, n in timeframes.items():
        offset = _TF_OFFSETS.get(label)
        if offset is not None:
            target = last_date - offset
            base_s = _base_price(close, target)
            sr = (last / base_s - 1.0) if (not np.isnan(base_s) and base_s != 0) else np.nan
        else:
            if len(close) < n + 1:
                sr = np.nan
            else:
                sr = last / float(close.iloc[-n - 1]) - 1.0

        out[f"ret_{label}"] = sr

        for b in benchmarks_df.columns:
            bs = benchmarks_df[b].dropna()
            if offset is not None:
                base_b = _base_price(bs, target)
                br = (float(bs.iloc[-1]) / base_b - 1.0) if (not np.isnan(base_b) and base_b != 0) else np.nan
            else:
                if len(bs) < n + 1:
                    out[f"outperf_{b}_{label}"] = np.nan
                    out[f"rs_ratio_{b}_{label}"] = np.nan
                    continue
                br = float(bs.iloc[-1]) / float(bs.iloc[-n - 1]) - 1.0

            if np.isnan(sr) or np.isnan(br):
                out[f"outperf_{b}_{label}"] = np.nan
                out[f"rs_ratio_{b}_{label}"] = np.nan
            else:
                out[f"outperf_{b}_{label}"] = sr - br
                out[f"rs_ratio_{b}_{label}"] = (1 + sr) / (1 + br)
    return out

def _pct_rank(s): return s.rank(pct=True) * 100

def build_scores(factor_df, cfg):
    df = factor_df.copy()
    primary = cfg["primary_benchmark"]
    tfs = list(cfg["timeframes"].keys())
    fw_raw = cfg["factor_weights"]
    fw = {k: v / sum(fw_raw.values()) for k, v in fw_raw.items()}

    df["trend_rank"]    = _pct_rank(df["trend_score_raw"])
    df["momentum_rank"] = _pct_rank(df["momentum_score_raw"])

    tf_composites = []
    for tf in tfs:
        df[f"rs_rank_{tf}"]      = _pct_rank(df[f"rs_ratio_{primary}_{tf}"])
        df[f"outperf_rank_{tf}"] = _pct_rank(df[f"outperf_{primary}_{tf}"])
        df[f"composite_{tf}"] = (
            fw["rs_rank"]      * df[f"rs_rank_{tf}"]
          + fw["outperf_rank"] * df[f"outperf_rank_{tf}"]
          + fw["trend_rank"]   * df["trend_rank"]
          + fw["momentum_rank"]* df["momentum_rank"]
        )
        tf_composites.append(f"composite_{tf}")

    tw = cfg["timeframe_weights"]
    w = np.array([tw[tf] for tf in tfs], dtype=float); w /= w.sum()
    comp = df[tf_composites].to_numpy()
    mask = ~np.isnan(comp)
    ws = np.nansum(comp * w, axis=1)
    wt = (mask * w).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        df["overall_strength"] = np.where(wt > 0, ws / wt, np.nan)

    benches = list(cfg["benchmarks"].keys())
    for tf in tfs:
        df[f"beats_all_{tf}"] = (df[[f"outperf_{b}_{tf}" for b in benches]] > 0).all(axis=1)
    return df

def add_sector_rank(df):
    df = df.copy()
    if "sector" in df.columns:
        df["sector_rank"] = df.groupby("sector")["overall_strength"].rank(pct=True) * 100
    return df


# =============================================================================
# MINERVINI / RS / VCP FUNCTIONS
# =============================================================================
def minervini_trend_template(close):
    last = float(close.iloc[-1])
    def ma(n): return float(close.rolling(n).mean().iloc[-1]) if len(close) >= n else np.nan
    ma50, ma150, ma200 = ma(50), ma(150), ma(200)
    ma200s = close.rolling(200).mean().dropna()
    ma200_slope = (float(ma200s.iloc[-1]) - float(ma200s.iloc[-21])) if len(ma200s) >= 21 else np.nan
    hi_52w = float(close.tail(252).max())
    lo_52w = float(close.tail(252).min())
    pct_above_low = last / lo_52w - 1.0
    pct_from_high = last / hi_52w - 1.0
    def sg(a, b): return bool(a > b) if not (np.isnan(a) or np.isnan(b)) else False
    c = {
        "tt_c1_price_gt_ma50":        sg(last, ma50),
        "tt_c2_ma50_gt_ma150":        sg(ma50, ma150),
        "tt_c3_ma150_gt_ma200":       sg(ma150, ma200),
        "tt_c4_price_gt_ma150":       sg(last, ma150),
        "tt_c5_price_gt_ma200":       sg(last, ma200),
        "tt_c6_ma200_slope_positive": bool(ma200_slope > 0) if not np.isnan(ma200_slope) else False,
        "tt_c7_30pct_above_52w_low":  pct_above_low >= 0.30,
        "tt_c8_within_25pct_52w_hi":  pct_from_high >= -0.25,
    }
    passed = sum(c.values())
    return {**c, "tt_criteria_passed": passed, "tt_pass": passed == 8,
            "tt_ma50": ma50, "tt_ma150": ma150, "tt_ma200": ma200,
            "tt_ma200_slope": ma200_slope,
            "tt_pct_above_52w_low": pct_above_low, "tt_pct_from_52w_hi": pct_from_high}

def compute_rs_line(stock_close, bench_close):
    aligned = pd.concat([stock_close.rename("s"), bench_close.rename("b")], axis=1).dropna()
    return aligned["s"] / aligned["b"]

def rs_line_metrics(rs):
    if len(rs) < 10:
        return {"rs_line_new_high_52w": False, "rs_line_new_high_63d": False,
                "rs_line_pct_from_hi": np.nan, "rs_line_slope_21d": np.nan}
    current = float(rs.iloc[-1])
    hi_52w  = float(rs.tail(252).max())
    hi_63d  = float(rs.tail(63).max())
    slope   = (current / float(rs.iloc[-21]) - 1.0) if len(rs) >= 21 else np.nan
    return {"rs_line_new_high_52w": current >= hi_52w * 0.999,
            "rs_line_new_high_63d": current >= hi_63d * 0.999,
            "rs_line_pct_from_hi":  current / hi_52w - 1.0,
            "rs_line_slope_21d":    slope}

def vcp_proxy(close, volume):
    def cv(n):
        s = close.tail(n); m = float(s.mean())
        return float(s.std()) / m if m > 0 else np.nan
    cv10, cv20, cv40, cv60 = cv(10), cv(20), cv(40), cv(60)
    avg10 = float(volume.tail(10).mean())
    avg40 = float(volume.tail(40).mean())
    score, signals = 0, []
    if not any(np.isnan(x) for x in [cv10, cv20]) and cv10 < cv20:  score += 1; signals.append("vol2w<4w")
    if not any(np.isnan(x) for x in [cv20, cv40]) and cv20 < cv40:  score += 1; signals.append("vol4w<8w")
    if not any(np.isnan(x) for x in [cv40, cv60]) and cv40 < cv60:  score += 1; signals.append("vol8w<12w")
    r10 = (float(close.tail(10).max()) - float(close.tail(10).min())) / float(close.tail(10).mean())
    r20 = (float(close.tail(20).max()) - float(close.tail(20).min())) / float(close.tail(20).mean())
    if r10 < r20: score += 1; signals.append("range_tight")
    if avg40 > 0 and avg10 < avg40 * 0.85: score += 1; signals.append("vol_dryup")
    return {"vcp_score": score, "vcp_signals": ",".join(signals) or "none",
            "vcp_vol_ratio": round(avg10 / avg40, 2) if avg40 > 0 else np.nan}


# =============================================================================
# BACKTEST — STEM MBA v3.3
# =============================================================================

def calc_mfi(high, low, close, volume, period=14):
    hlc3   = (high + low + close) / 3
    delta  = hlc3.diff()
    raw_mf = hlc3 * volume
    pos_mf = raw_mf.where(delta > 0, 0.0)
    neg_mf = raw_mf.where(delta < 0, 0.0)
    mfr    = pos_mf.rolling(period).sum() / neg_mf.rolling(period).sum().replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + mfr)


def calc_atr_bt(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False, min_periods=period).mean()


@st.cache_data(show_spinner=False)
def fetch_ohlcv_cached(tickers_key: str, lookback_days: int = 1100):
    import json
    tickers = json.loads(tickers_key)
    end   = datetime.today()
    start = end - timedelta(days=lookback_days)
    out   = {}
    for i in range(0, len(tickers), 50):
        batch = tickers[i:i + 50]
        try:
            data = yf.download(batch, start=start, end=end, auto_adjust=True,
                               progress=False, group_by="ticker", threads=True)
        except Exception:
            continue
        for t in batch:
            try:
                df = (data[["Open", "High", "Low", "Close", "Volume"]].copy()
                      if len(batch) == 1 else
                      data[t][["Open", "High", "Low", "Close", "Volume"]].copy())
                df = df.dropna()
                if len(df) >= 200:
                    out[t] = df
            except Exception:
                continue
    return out



def compute_entry_setups(ohlcv_dict: dict, p: dict) -> pd.DataFrame:
    """
    Evaluate the CURRENT bar of every ticker against STEM MBA entry conditions.
    Returns a DataFrame sorted by readiness score (0-6).
    """
    rows = []
    for ticker, df in ohlcv_dict.items():
        if len(df) < 220:
            continue
        close  = df["Close"]
        high   = df["High"]
        low    = df["Low"]
        volume = df["Volume"]
        n = len(df)
        c = float(close.iloc[-1])

        # ── Pre-compute indicator series ────────────────────────────────────
        sma21_s = close.rolling(21).mean()
        sma30_s = close.rolling(30).mean()
        sma50_s = close.rolling(50).mean()
        s21  = float(sma21_s.iloc[-1])
        s30  = float(sma30_s.iloc[-1])
        s50  = float(sma50_s.iloc[-1])
        s100 = float(close.rolling(100).mean().iloc[-1]) if len(close) >= 100 else np.nan
        s200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else np.nan
        if any(np.isnan(x) for x in [s21, s30, s50]):
            continue

        mfi_val = float(calc_mfi(high, low, close, volume, 14).iloc[-1])
        atr_val = float(calc_atr_bt(high, low, close, 14).iloc[-1])
        if np.isnan(atr_val) or atr_val <= 0:
            continue

        # ── 6 entry conditions (same logic as _stem_mba_one) ───────────────
        fm, fn = max(s21, s30), min(s21, s30)
        clust_pct  = (fm - fn) / fn * 100 if fn > 0 else np.nan
        cond_cluster = (not np.isnan(clust_pct)) and clust_pct <= p["cluster_input"]

        cond_above = c > s21 and c > s30 and c > s50

        cond_bars = True
        for j in range(p["bars_required"]):
            ij = n - 1 - j
            if ij < 0: cond_bars = False; break
            cj = float(close.iloc[ij])
            if not (cj > float(sma21_s.iloc[ij]) and
                    cj > float(sma30_s.iloc[ij]) and
                    cj > float(sma50_s.iloc[ij])):
                cond_bars = False; break

        cond_mfi = p["mfi_long_min"] < mfi_val < 70

        cond_trend = True
        if p.get("use_trend", True) and not np.isnan(s100) and not np.isnan(s200):
            cond_trend = c > s100 and c > s200 and s100 > s200

        cond_htf = True
        if p.get("use_htf", True):
            wc = close.resample("W-FRI").last().dropna()
            if len(wc) >= 20:
                ws = wc.rolling(20).mean()
                if not np.isnan(ws.iloc[-1]):
                    cond_htf = float(wc.iloc[-1]) > float(ws.iloc[-1])

        score = sum([cond_cluster, cond_above, cond_bars,
                     cond_mfi, cond_trend, cond_htf])
        ready = score == 6

        # ── Entry levels ────────────────────────────────────────────────────
        stop_p    = round(c - atr_val * p["atr_mult"], 2)
        risk_ps   = c - stop_p
        t1_p      = round(c + risk_ps * 2.0, 2)
        stop_pct  = round(risk_ps / c * 100, 2)

        # Position size from portfolio settings
        capital   = p.get("capital", 20_000)
        pos_pct   = p.get("pos_pct", 15)
        pos_size  = round(capital * pos_pct / 100, 2)
        shares    = max(1, int(pos_size / c)) if c > 0 else 0
        dollar_risk = round(shares * risk_ps, 2)

        # Volume
        vol20     = float(volume.tail(20).mean())
        vol_ratio = round(float(volume.iloc[-1]) / vol20, 2) if vol20 > 0 else np.nan

        rows.append({
            "ticker":       ticker,
            "score":        score,
            "ready":        ready,
            "close":        round(c, 2),
            "cluster_%":    round(clust_pct, 2) if not np.isnan(clust_pct) else np.nan,
            "✔ cluster":    cond_cluster,
            "✔ above MAs":  cond_above,
            "✔ bars":       cond_bars,
            "✔ MFI":        cond_mfi,
            "✔ trend":      cond_trend,
            "✔ HTF":        cond_htf,
            "MFI":          round(mfi_val, 1),
            "entry_$":      round(c, 2),
            "stop_$":       stop_p,
            "T1_$":         t1_p,
            "stop_%":       stop_pct,
            "pos_size_$":   pos_size,
            "shares":       shares,
            "risk_$":       dollar_risk,
            "vol_ratio":    vol_ratio,
        })

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).set_index("ticker")
    return out.sort_values(["score", "ready"], ascending=[False, False])


def compute_ma_cross_signals(stock_data: dict, lookback_bars: int = 5) -> pd.DataFrame:
    """
    For every stock, check whether price is currently above SMA21 AND SMA30.
    Also flags a 'fresh cross' if the crossover happened within the last
    `lookback_bars` bars (i.e. price was below at least one of them before).

    No SMA50 / SMA200 requirement — this is an early-stage signal.
    """
    rows = []
    for ticker, df in stock_data.items():
        close = df["Close"]
        volume = df["Volume"]
        if len(close) < 35:
            continue

        sma21_s = close.rolling(21).mean()
        sma30_s = close.rolling(30).mean()
        sma50_s = close.rolling(50).mean()
        sma200_s = close.rolling(200).mean()

        c    = float(close.iloc[-1])
        s21  = float(sma21_s.iloc[-1])
        s30  = float(sma30_s.iloc[-1])
        s50  = float(sma50_s.iloc[-1])  if len(close) >= 50  else np.nan
        s200 = float(sma200_s.iloc[-1]) if len(close) >= 200 else np.nan

        if np.isnan(s21) or np.isnan(s30):
            continue

        above_21 = c > s21
        above_30 = c > s30
        if not (above_21 and above_30):
            continue                          # only keep stocks above BOTH

        # RSI
        rsi_val = float(rsi(close, 14).iloc[-1])

        # Volume ratio vs 20-day avg
        vol20 = float(volume.tail(20).mean())
        vol_ratio = float(volume.iloc[-1]) / vol20 if vol20 > 0 else np.nan

        # Detect fresh crossover — was price below SMA21 OR SMA30
        # at any point in the last `lookback_bars` bars?
        fresh_bars = None
        for lag in range(1, lookback_bars + 1):
            idx = -(lag + 1)
            if abs(idx) > len(close):
                break
            c_prev   = float(close.iloc[idx])
            s21_prev = float(sma21_s.iloc[idx])
            s30_prev = float(sma30_s.iloc[idx])
            if c_prev < s21_prev or c_prev < s30_prev:
                fresh_bars = lag   # crossed `lag` bars ago
                break

        rows.append({
            "ticker":       ticker,
            "close":        round(c, 2),
            "sma21":        round(s21, 2),
            "sma30":        round(s30, 2),
            "above_21_%":   round((c / s21 - 1) * 100, 2),
            "above_30_%":   round((c / s30 - 1) * 100, 2),
            "vs_sma50_%":   round((c / s50  - 1) * 100, 2) if not np.isnan(s50)  else np.nan,
            "vs_sma200_%":  round((c / s200 - 1) * 100, 2) if not np.isnan(s200) else np.nan,
            "above_50":     bool(c > s50)  if not np.isnan(s50)  else None,
            "above_200":    bool(c > s200) if not np.isnan(s200) else None,
            "RSI":          round(rsi_val, 1),
            "vol_ratio":    round(vol_ratio, 2) if not np.isnan(vol_ratio) else np.nan,
            "fresh_cross":  fresh_bars,         # None = wasn't below recently
            "fresh":        fresh_bars is not None,
        })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).set_index("ticker")
    # Sort: fresh crosses first, then by how far above SMA21 (tighter = better)
    out["_sort"] = out["fresh"].astype(int) * 1000 - out["above_21_%"]
    return out.sort_values("_sort", ascending=False).drop(columns="_sort")



# =============================================================================
# MAIN SCREENER RUNNER  (cached so it only re-runs when config changes)
# =============================================================================
@st.cache_data(show_spinner=False)
def run_screener_cached(cfg_key: str, cfg_dict: str):
    import json
    cfg = json.loads(cfg_dict)

    status = st.status("Running screener...", expanded=True)

    def prog(done, total, msg):
        status.write(msg)

    # Universe
    status.write(f"Loading universe: {cfg['universe']}...")
    universe_df = resolve_universe(cfg["universe"])
    if cfg.get("sector_filter"):
        universe_df = universe_df[universe_df["sector"].isin(cfg["sector_filter"])].reset_index(drop=True)
    tickers = universe_df["ticker"].tolist()
    status.write(f"Universe: {len(tickers)} tickers")

    # Market cap filter
    mcap_map = {}
    min_mcap = cfg.get("min_market_cap") or 0
    if min_mcap > 0:
        status.write("Fetching market caps...")
        mcap_map = fetch_market_caps(tickers, progress_cb=prog)
        tickers = [t for t in tickers if (mcap_map.get(t) or 0) >= min_mcap]
        universe_df = universe_df[universe_df["ticker"].isin(tickers)].reset_index(drop=True)
        status.write(f"After market cap filter: {len(tickers)} tickers")

    # Benchmarks
    status.write("Fetching benchmarks...")
    bench_data = fetch_history(list(cfg["benchmarks"].keys()), lookback_days=500, batch_size=10)
    if not bench_data:
        st.error(
            "⚠️ Could not download benchmark data (SPY / QQQ). "
            "This is usually a temporary yfinance / network issue. "
            "Wait 30 seconds and click **🚀 Run Screener** again."
        )
        st.stop()
    benchmarks_df = pd.concat({t: d["Close"] for t, d in bench_data.items()}, axis=1).dropna(how="all")

    # Stock history
    status.write(f"Fetching price history for {len(tickers)} stocks...")
    stock_data = fetch_history(tickers, lookback_days=500, progress_cb=prog)
    status.write(f"Price data received for {len(stock_data)} tickers")

    # Factors
    status.write("Computing factors...")
    all_tf = {**cfg["timeframes"], **cfg.get("short_term_timeframes", {})}
    ma_stack_cfg = cfg.get("ma_stack", {"fast1": 21, "fast2": 30, "slow": 200})
    rows = {}
    for t, df in stock_data.items():
        try:
            rows[t] = compute_ticker_factors(df, benchmarks_df, all_tf, ma_stack=ma_stack_cfg)
        except Exception:
            continue
    factor_df = pd.DataFrame(rows).T
    factor_df.index.name = "ticker"
    meta = universe_df.set_index("ticker")[["name", "sector", "industry"]]
    factor_df = factor_df.join(meta, how="left")
    factor_df[["sector", "industry"]] = factor_df[["sector", "industry"]].fillna("Unknown")
    if mcap_map:
        factor_df["market_cap"] = pd.Series(mcap_map)
        factor_df["market_cap_b"] = factor_df["market_cap"] / 1e9

    # Hard filters
    mask = factor_df["avg_vol_50"] >= cfg["min_avg_volume"]
    if (cfg.get("min_price") or 0) > 0:
        mask &= factor_df["price"] >= cfg["min_price"]
    if cfg.get("require_ma_stack", True):
        mask &= factor_df["ma_stack_ok"].fillna(False).astype(bool)
    factor_df = factor_df[mask]

    # Score
    status.write("Scoring...")
    ranked = build_scores(factor_df, cfg)
    ranked = add_sector_rank(ranked)
    if cfg.get("must_beat_both_on"):
        col = f"beats_all_{cfg['must_beat_both_on']}"
        ranked = ranked[ranked[col]].copy()
    ranked = ranked.sort_values("overall_strength", ascending=False)

    # Minervini / RS / VCP
    status.write("Computing Minervini TT, RS Line, VCP...")
    spy = benchmarks_df[cfg["primary_benchmark"]].dropna()
    enh_rows = {}
    for ticker in ranked.index:
        if ticker not in stock_data: continue
        df_ = stock_data[ticker]
        row = {}
        row.update(minervini_trend_template(df_["Close"]))
        rs = compute_rs_line(df_["Close"], spy)
        row.update(rs_line_metrics(rs))
        row.update(vcp_proxy(df_["Close"], df_["Volume"]))
        enh_rows[ticker] = row
    enh_df = pd.DataFrame(enh_rows).T
    enh_df.index.name = "ticker"
    ranked_plus = ranked.join(enh_df, how="left")

    spy_ret_2w = float(spy.iloc[-1]) / float(spy.iloc[-11]) - 1.0
    spy_ret_4w = float(spy.iloc[-1]) / float(spy.iloc[-21]) - 1.0
    ranked_plus["sepa_signal"] = (
        ranked_plus["tt_pass"].fillna(False).astype(bool)
        & ranked_plus["rs_line_new_high_63d"].fillna(False).astype(bool)
        & (ranked_plus["vcp_score"].fillna(0) >= 2)
    )
    ranked_plus["diverged_2w"] = (pd.to_numeric(ranked_plus.get("ret_2W", np.nan), errors="coerce") > 0) & (spy_ret_2w < 0)
    ranked_plus["diverged_4w"] = (pd.to_numeric(ranked_plus.get("ret_4W", np.nan), errors="coerce") > 0) & (spy_ret_4w < 0)

    status.update(label="Screener complete!", state="complete")
    return ranked_plus, stock_data, benchmarks_df, spy_ret_2w, spy_ret_4w


# =============================================================================
# CHART HELPERS  (plotly)
# =============================================================================
COLORS = {
    "stock":  "#60a5fa",   # blue-400
    "SPY":    "#34d399",   # emerald-400
    "QQQ":    "#fb923c",   # orange-400
    "bg":     "#030912",   # app background
    "panel":  "#060f22",   # chart background
    "text":   "#94a3b8",   # slate-400
    "title":  "#e2e8f0",   # slate-200
    "grid":   "rgba(255,255,255,0.04)",
    "border": "rgba(255,255,255,0.05)",
}

def _plotly_layout(fig, title="", height=450):
    fig.update_layout(
        title=dict(text=title,
                   font=dict(color=COLORS["title"], size=12, family="Inter, sans-serif"),
                   x=0, xref="paper", pad=dict(l=4)),
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["panel"],
        font=dict(color=COLORS["text"], family="Inter, sans-serif", size=11),
        height=height,
        xaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"],
                   linecolor=COLORS["border"], tickfont=dict(size=10)),
        yaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"],
                   linecolor=COLORS["border"], tickfont=dict(size=10)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=COLORS["text"], size=10),
                    bordercolor=COLORS["border"], borderwidth=1),
        margin=dict(l=50, r=30, t=46, b=44),
        hoverlabel=dict(bgcolor="#0f172a", bordercolor=COLORS["border"],
                        font=dict(color="#e2e8f0", size=12, family="Inter")),
    )
    return fig

def chart_grouped_bar(ranked_plus, cfg, tf="6M", top_n=15):
    primary = cfg["primary_benchmark"]
    benches = list(cfg["benchmarks"].keys())
    top = ranked_plus.head(top_n)
    tickers = list(top.index)
    stock_rets = (pd.to_numeric(top[f"ret_{tf}"], errors="coerce") * 100).round(1)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Stock", x=tickers, y=stock_rets,
        marker_color=COLORS["stock"],
        text=[f"{v:+.0f}%" for v in stock_rets],
        textposition="outside", textfont=dict(size=9),
    ))
    for b in benches:
        outperf = pd.to_numeric(top[f"outperf_{b}_{tf}"], errors="coerce") * 100
        bench_ret = stock_rets - outperf
        fig.add_trace(go.Bar(
            name=b, x=tickers, y=bench_ret.round(1),
            marker_color=COLORS.get(b, "#aaa"),
        ))

    fig.update_layout(barmode="group")
    _plotly_layout(fig, f"Top {top_n} Stocks — {tf} Return vs Benchmarks", height=420)
    fig.update_yaxes(ticksuffix="%")
    return fig

def chart_heatmap(ranked_plus, cfg, top_n=25):
    primary = cfg["primary_benchmark"]
    tfs = list(cfg["timeframes"].keys())
    top = ranked_plus.head(top_n)
    data = pd.DataFrame(
        {tf: (pd.to_numeric(top[f"outperf_{primary}_{tf}"], errors="coerce") * 100).round(1) for tf in tfs},
        index=top.index
    )
    fig = go.Figure(go.Heatmap(
        z=data.values, x=tfs, y=list(data.index),
        colorscale="RdYlGn", zmid=0,
        text=[[f"{v:+.1f}%" if not np.isnan(v) else "—" for v in row] for row in data.values],
        texttemplate="%{text}", textfont=dict(size=9),
        colorbar=dict(title=dict(text=f"% vs {primary}", font=dict(color=COLORS["text"])),
                      tickfont=dict(color=COLORS["text"])),
    ))
    _plotly_layout(fig, f"Outperformance vs {primary} (%) — Top {top_n} Stocks", height=max(400, top_n * 22))
    return fig

def chart_bubble(ranked_plus, cfg, tf="6M", top_n=40):
    top = ranked_plus.head(top_n).copy()
    x = pd.to_numeric(top[f"ret_{tf}"], errors="coerce") * 100
    y = pd.to_numeric(top["overall_strength"], errors="coerce")
    sizes = (pd.to_numeric(top.get("market_cap_b", pd.Series(50, index=top.index)), errors="coerce")
               .fillna(50).clip(10, 3000) / 3000 * 60 + 8)

    fig = px.scatter(
        x=x, y=y, size=sizes, color=top["sector"].fillna("Unknown"),
        text=top.index, hover_name=top.index,
        hover_data={"sector": top["sector"].fillna("Unknown"),
                    "score": y.round(1), "ret": x.round(1)},
        color_discrete_sequence=px.colors.qualitative.Dark24,
    )
    fig.update_traces(textposition="top center", textfont=dict(size=7))
    _plotly_layout(fig, f"{tf} Return vs Overall Score  |  bubble ∝ market cap  |  color = sector", height=520)
    fig.update_xaxes(ticksuffix="%", title=f"{tf} Return (%)")
    fig.update_yaxes(title="Overall Strength Score (0–100)")
    return fig

def chart_divergence(ranked_plus, cfg, window="4W"):
    col = f"diverged_{window.lower()}"
    ret_col = f"ret_{window}"
    if col not in ranked_plus.columns:
        return None
    div = (ranked_plus[ranked_plus[col]]
           .dropna(subset=[ret_col])
           .sort_values(ret_col, ascending=False)
           .head(20))
    if div.empty:
        return None
    rets = (pd.to_numeric(div[ret_col], errors="coerce") * 100).round(1)
    sepa = div["sepa_signal"].fillna(False).astype(bool)
    colors = ["#3fb950" if s else "#58a6ff" for s in sepa]
    vcp = div["vcp_score"].fillna(0).astype(int) if "vcp_score" in div.columns else pd.Series(0, index=div.index)

    fig = go.Figure(go.Bar(
        x=list(div.index), y=rets,
        marker_color=colors,
        text=[f"VCP:{v}" for v in vcp],
        textposition="outside", textfont=dict(size=8),
    ))
    _plotly_layout(fig, f"Divergence — Stocks UP while SPY fell ({window})  |  🟢 SEPA signal  🔵 divergent only", height=400)
    fig.update_yaxes(ticksuffix="%")
    return fig

def chart_rs_line(ticker, stock_data, benchmarks_df, cfg, row_data):
    if ticker not in stock_data:
        return None
    primary = cfg["primary_benchmark"]
    spy = benchmarks_df[primary].dropna()
    close = stock_data[ticker]["Close"].tail(130)
    rs = compute_rs_line(close, spy).tail(130)
    dates = list(range(len(close)))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.04)

    for ma_n, col, dash in [(50, "#f0e68c", "dash"), (150, "#ffa07a", "dash"), (200, "#ff6347", "solid")]:
        ma = close.rolling(ma_n).mean()
        fig.add_trace(go.Scatter(x=dates, y=ma.values, name=f"{ma_n}MA",
                                 line=dict(color=col, width=1, dash=dash), opacity=0.8), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=close.values, name="Price",
                             line=dict(color=COLORS["stock"], width=1.5)), row=1, col=1)

    fig.add_trace(go.Scatter(x=list(range(len(rs))), y=rs.values, name="RS Line",
                             line=dict(color="#3fb950", width=1.5)), row=2, col=1)
    fig.add_hline(y=float(rs.tail(252).max()), line=dict(color="#ffd700", width=1, dash="dot"),
                  annotation_text="52W hi", annotation_font=dict(color="#ffd700", size=8), row=2, col=1)
    fig.add_hline(y=float(rs.tail(63).max()),  line=dict(color="#87ceeb", width=1, dash="dot"),
                  annotation_text="3M hi",  annotation_font=dict(color="#87ceeb", size=8),  row=2, col=1)

    tt  = int(row_data.get("tt_criteria_passed", 0))
    vcp = int(row_data.get("vcp_score", 0))
    sc  = row_data.get("overall_strength", 0)
    rs_hi = bool(row_data.get("rs_line_new_high_63d", False))
    title_color = "#3fb950" if rs_hi else COLORS["text"]
    star = " ★ RS 3M HIGH" if rs_hi else ""

    fig.update_layout(
        title=dict(text=f"{ticker}{star}  |  TT:{tt}/8  VCP:{vcp}  Score:{sc:.0f}",
                   font=dict(color=title_color, size=12)),
        paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["panel"],
        font=dict(color=COLORS["text"]), height=380,
        xaxis2=dict(showticklabels=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=8)),
        margin=dict(l=40, r=20, t=50, b=20),
    )
    for ax in ["xaxis", "xaxis2", "yaxis", "yaxis2"]:
        fig.update_layout(**{ax: dict(gridcolor=COLORS["grid"])})
    return fig

def chart_sector_bar(ranked_plus):
    if "sector" not in ranked_plus.columns or ranked_plus.empty:
        return None
    lb = (ranked_plus.groupby("sector")["overall_strength"]
          .agg(["mean", "count"]).sort_values("mean", ascending=True).reset_index())
    lb.columns = ["Sector", "Avg Score", "Count"]
    fig = go.Figure(go.Bar(
        x=lb["Avg Score"].round(1), y=lb["Sector"],
        orientation="h",
        marker_color=lb["Avg Score"],
        marker_colorscale="RdYlGn", marker_cmin=40, marker_cmax=90,
        text=[f"{s:.1f}  (n={n})" for s, n in zip(lb["Avg Score"], lb["Count"])],
        textposition="outside", textfont=dict(size=9),
    ))
    _plotly_layout(fig, "Sector Leaderboard — Average Overall Strength Score", height=400)
    return fig


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("""
    <div style="padding:8px 0 20px 0;border-bottom:1px solid rgba(255,255,255,0.04);margin-bottom:20px;">
      <div style="font-size:10px;font-weight:700;color:#1d4ed8;letter-spacing:1.4px;text-transform:uppercase;margin-bottom:5px;">
        Configuration
      </div>
      <div style="font-size:17px;font-weight:800;color:#e2e8f0;letter-spacing:-0.4px;">
        Screener Settings
      </div>
    </div>
    """, unsafe_allow_html=True)

    universe = st.selectbox(
        "Universe",
        ["both", "sp500", "nasdaq100", "russell2000", "all"],
        index=0,
        help=(
            "both = S&P 500 + Nasdaq 100  |  "
            "russell2000 = IWM constituents  |  "
            "all = S&P 500 + Nasdaq 100 + Russell 2000"
        ),
    )
    # Default to $5B for Russell 2000 universes to keep scan manageable
    _r2k_universe = universe in ("russell2000", "all")
    _mcap_options = ["$10B", "$5B", "$1B", "None"]
    _mcap_default = 1 if _r2k_universe else 0   # $5B index=1, $10B index=0
    min_mcap = st.selectbox(
        "Min Market Cap", _mcap_options, index=_mcap_default,
        help="Russell 2000 defaults to $5B to filter out micro-caps",
    )
    mcap_map_val = {"$10B": 10e9, "$5B": 5e9, "$1B": 1e9, "None": 0}

    st.markdown("---")
    st.markdown("**Filters**")
    require_ma    = st.checkbox("Require MA Stack (21&30 > 200)", value=True)
    must_beat     = st.selectbox("Must beat both benchmarks on", ["6M", "3M", "12M", "None"], index=0)
    min_vol       = st.number_input("Min Avg Volume", value=500_000, step=100_000)

    st.markdown("---")
    st.markdown("**Divergence Window**")
    div_window = st.radio("Show divergence for", ["4W", "2W"], index=0, horizontal=True)

    st.markdown("---")
    run_btn = st.button("🚀 Run Screener", type="primary", use_container_width=True)
    st.caption("First run ~3–5 min  |  Results cached until config changes")

    st.markdown("---")
    with st.expander("⚙️ Entry Setup Parameters", expanded=False):
        st.caption("Parameters used by the Entry Setups scanner")
        bt_cluster    = st.slider("MA Cluster %",          0.1, 5.0,  1.0, 0.1, key="bt_cluster")
        bt_bars_req   = st.slider("Confirming bars",       1,   5,    2,         key="bt_bars")
        bt_mfi_min    = st.slider("MFI long min",          20,  70,   45,        key="bt_mfi_min")
        bt_atr_mult   = st.slider("ATR stop mult",         0.5, 4.0,  1.5, 0.1, key="bt_atr")
        bt_use_trend  = st.checkbox("Trend filter (100/200 MA)", value=True,  key="bt_trend")
        bt_use_htf    = st.checkbox("Weekly HTF filter",         value=True,  key="bt_htf")

        st.markdown("---")
        st.markdown("**Position sizing (for entry cards)**")
        bt_capital  = st.number_input(
            "Account size ($)", min_value=1_000, max_value=10_000_000,
            value=20_000, step=1_000, key="bt_capital",
            help="Used to calculate position size and dollar risk on each setup card.",
        )
        bt_pos_pct  = st.slider(
            "Position size (% of account)", 5, 50, 15, step=1,
            key="bt_pos_pct",
            help="Each setup card shows this % of account as the position size.",
        )
        _pos_usd = bt_capital * bt_pos_pct / 100
        st.caption(f"Position size ≈ **${_pos_usd:,.0f}** per trade")


# =============================================================================
# BUILD CONFIG FROM SIDEBAR
# =============================================================================
import json

cfg = {**DEFAULT_CONFIG}
cfg["universe"]          = universe
cfg["min_market_cap"]    = mcap_map_val[min_mcap]
cfg["require_ma_stack"]  = require_ma
cfg["must_beat_both_on"] = None if must_beat == "None" else must_beat
cfg["min_avg_volume"]    = int(min_vol)

cfg_key  = json.dumps({k: v for k, v in cfg.items() if k not in ("benchmarks",)}, sort_keys=True)
cfg_json = json.dumps(cfg)


# =============================================================================
# MAIN LAYOUT
# =============================================================================
st.markdown(f"""
<div class="topnav">
  <div class="brand">
    <div class="brand-icon">📈</div>
    <div>
      <div class="brand-name">StockScreener</div>
      <div class="brand-tagline">Momentum · Minervini SEPA · Divergence · S&amp;P500 &amp; Nasdaq100</div>
    </div>
  </div>
  <div class="nav-right">
    <div class="nav-date">{datetime.now().strftime('%A, %B %d %Y')}</div>
    <div class="live-pill"><div class="live-dot"></div>Live Data</div>
  </div>
</div>
""", unsafe_allow_html=True)

if "results" not in st.session_state:
    st.session_state.results = None
if "entry_setups" not in st.session_state:
    st.session_state.entry_setups = None
if "notes" not in st.session_state:
    st.session_state.notes = _load_notes()

if run_btn:
    st.session_state.results = run_screener_cached(cfg_key, cfg_json)

if st.session_state.results is None:
    st.info("Configure settings in the sidebar and click **Run Screener** to begin.")
    st.stop()

ranked_plus, stock_data, benchmarks_df, spy_ret_2w, spy_ret_4w = st.session_state.results
primary = cfg["primary_benchmark"]
tfs     = list(cfg["timeframes"].keys())
spy     = benchmarks_df[primary].dropna()


# ── Top KPI metrics ──────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Stocks Ranked",      len(ranked_plus))
k2.metric("SEPA Signals",       int(ranked_plus["sepa_signal"].sum()))
k3.metric("Full TT Pass",       int(ranked_plus["tt_pass"].sum()))
k4.metric("RS Line 3M High",    int(ranked_plus["rs_line_new_high_63d"].sum()))
k5.metric("SPY 2W Return",      f"{spy_ret_2w*100:+.2f}%")
k6.metric("SPY 4W Return",      f"{spy_ret_4w*100:+.2f}%")

st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)


# =============================================================================
# TABS
# =============================================================================
tab_ov, tab_cmp, tab_sepa, tab_div, tab_sec, tab_charts, tab_entry, tab_notes = st.tabs([
    "📋 Overview", "📊 Comparison", "🎯 SEPA Watchlist",
    "⚡ Divergence", "🏭 Sectors", "📉 Charts", "📍 Entry Setups", "📝 Notes"
])


# ── TAB 1: OVERVIEW ──────────────────────────────────────────────────────────
with tab_ov:
    st.subheader("All Stocks by Overall Strength")

    disp_cols = {
        "sector": "Sector", "market_cap_b": "MCap $B", "sector_rank": "Sector Rank",
        "overall_strength": "Score",
        **{f"ret_{tf}": f"{tf} Ret" for tf in tfs},
        **{f"outperf_{primary}_{tf}": f"Edge {tf}" for tf in tfs},
        "trend_score_raw": "Trend", "momentum_score_raw": "Mom",
        "rsi_14": "RSI", "pct_from_52w_hi": "vs52WHi", "price": "Price",
    }
    avail = {k: v for k, v in disp_cols.items() if k in ranked_plus.columns}
    overview_df = ranked_plus[list(avail.keys())].rename(columns=avail).copy()

    pct_cols = [c for c in overview_df.columns if "Ret" in c or "Edge" in c or "vs52W" in c]
    fmt = {c: "{:+.1f}%" for c in pct_cols}
    fmt["Score"] = "{:.1f}"
    if "MCap $B" in overview_df.columns:
        fmt["MCap $B"] = "${:.0f}B"

    styled = (
        overview_df.style
        .format({c: lambda x, f=f: (f.format(x * 100) if "%" in f else f.format(x))
                 if pd.notna(x) else "—" for c, f in fmt.items()}, na_rep="—")
        .background_gradient(subset=["Score"] if "Score" in overview_df.columns else [], cmap="RdYlGn", vmin=50, vmax=100)
        .background_gradient(subset=[c for c in pct_cols if "Edge" in c], cmap="RdYlGn", vmin=-20, vmax=20)
    )
    st.dataframe(styled, use_container_width=True, height=600)


# ── TAB 2: COMPARISON TABLE ───────────────────────────────────────────────────
with tab_cmp:
    st.subheader("Stock Returns vs Benchmarks — Side by Side")
    st.caption("Edge = Stock % − Benchmark %  |  Positive = outperformed  |  Shown vs all benchmarks (SPY & QQQ)")

    benches = list(cfg["benchmarks"].keys())
    other   = [b for b in benches if b != primary]

    rows = []
    for ticker, row in ranked_plus.iterrows():
        entry = {"Ticker": ticker,
                 "Sector": str(row.get("sector", ""))[:18],
                 "Score":  round(float(row.get("overall_strength", 0)), 1)}
        for tf in tfs:
            try:
                sr = float(row.get(f"ret_{tf}", np.nan))
            except Exception:
                sr = np.nan
            entry[f"{tf} Stock%"] = round(sr * 100, 1) if not np.isnan(sr) else np.nan
            for b in benches:
                try:
                    op = float(row.get(f"outperf_{b}_{tf}", np.nan))
                    br = sr - op
                except Exception:
                    op = br = np.nan
                entry[f"{tf} {b}%"]     = round(br * 100, 1) if not np.isnan(br) else np.nan
                entry[f"{tf} {b} Edge"] = round(op * 100, 1) if not np.isnan(op) else np.nan
        rows.append(entry)

    cmp_df = pd.DataFrame(rows).set_index("Ticker")
    edge_cols = [c for c in cmp_df.columns if "Edge" in c]

    def color_edge(val):
        try:
            v = float(val)
            if v > 0:  return f"background-color: rgba(0,{min(int(100+v*5),220)},80,0.3); color:#00cc44"
            if v < 0:  return f"background-color: rgba({min(int(100+abs(v)*5),220)},40,40,0.3); color:#ff4444"
        except Exception: pass
        return ""

    styled_cmp = (
        cmp_df.style
        .map(color_edge, subset=edge_cols)
        .background_gradient(subset=["Score"], cmap="RdYlGn", vmin=50, vmax=100)
        .format(na_rep="—", precision=1)
    )
    st.dataframe(styled_cmp, use_container_width=True, height=600)


# ── TAB 3: SEPA WATCHLIST ─────────────────────────────────────────────────────
with tab_sepa:
    st.subheader("Minervini SEPA Watchlist")
    st.caption("TT✓ = all 8 Stage 2 criteria  |  RS 3M Hi = RS Line at 3-month high  |  VCP = volatility contraction (0–5)")

    sepa_cols = {
        "sector": "Sector", "overall_strength": "Score",
        "tt_criteria_passed": "TT/8", "tt_pass": "TT✓",
        "rs_line_new_high_63d": "RS 3M Hi", "rs_line_new_high_52w": "RS 52W Hi",
        "rs_line_slope_21d": "RS Slope 21d", "rs_line_pct_from_hi": "RS vs Hi%",
        "vcp_score": "VCP", "vcp_signals": "VCP Detail", "sepa_signal": "SEPA",
        "diverged_2w": "Div 2W", "diverged_4w": "Div 4W",
        "ret_6M": "6M Ret", "rsi_14": "RSI", "price": "Price",
    }
    avail_sepa = {k: v for k, v in sepa_cols.items() if k in ranked_plus.columns}
    sepa_df = ranked_plus[list(avail_sepa.keys())].rename(columns=avail_sepa).copy()

    for col in ["RS Slope 21d", "RS vs Hi%", "6M Ret"]:
        if col in sepa_df.columns:
            sepa_df[col] = pd.to_numeric(sepa_df[col], errors="coerce").map(
                lambda x: f"{x*100:+.1f}%" if pd.notna(x) else "—")

    def style_bool(val):
        if val is True  or val == True:  return "background-color:#1a4731; color:#3fb950; font-weight:bold"
        if val is False or val == False: return "background-color:#3d1f1f; color:#f85149"
        return ""

    bool_cols = [c for c in ["TT✓","RS 3M Hi","RS 52W Hi","SEPA","Div 2W","Div 4W"] if c in sepa_df.columns]
    styled_sepa = (
        sepa_df.style
        .map(style_bool, subset=bool_cols)
        .background_gradient(subset=["Score"] if "Score" in sepa_df.columns else [], cmap="RdYlGn", vmin=50, vmax=100)
        .background_gradient(subset=["TT/8"]  if "TT/8"  in sepa_df.columns else [], cmap="RdYlGn", vmin=0, vmax=8)
        .background_gradient(subset=["VCP"]   if "VCP"   in sepa_df.columns else [], cmap="Blues",  vmin=0, vmax=5)
        .format(na_rep="—", precision=1)
    )
    st.dataframe(styled_sepa, use_container_width=True, height=550)

    st.markdown("---")
    st.subheader("Minervini Criteria Heatmap")
    tt_cols = ["tt_c1_price_gt_ma50","tt_c2_ma50_gt_ma150","tt_c3_ma150_gt_ma200",
               "tt_c4_price_gt_ma150","tt_c5_price_gt_ma200","tt_c6_ma200_slope_positive",
               "tt_c7_30pct_above_52w_low","tt_c8_within_25pct_52w_hi"]
    tt_labels = ["P>50MA","50>150","150>200","P>150","P>200","200↑","+30%low","<25%hi"]
    avail_tt = [c for c in tt_cols if c in ranked_plus.columns]
    heat = ranked_plus[avail_tt].astype(float)
    labels_avail = [tt_labels[tt_cols.index(c)] for c in avail_tt]

    fig_tt = go.Figure(go.Heatmap(
        z=heat.values, x=labels_avail, y=list(heat.index),
        colorscale=[[0,"#3d1f1f"],[1,"#1a4731"]],
        zmin=0, zmax=1, showscale=False,
        text=[["✓" if v else "✗" for v in row] for row in heat.values],
        texttemplate="%{text}", textfont=dict(size=10),
    ))
    _plotly_layout(fig_tt, "Minervini Trend Template — Pass/Fail per Criterion", height=max(350, len(heat)*20))
    st.plotly_chart(fig_tt, use_container_width=True)

    st.markdown("---")
    st.subheader("RS Line Charts — Top SEPA Stocks")
    candidates = ranked_plus[ranked_plus["sepa_signal"]].head(6)
    if len(candidates) < 3:
        candidates = ranked_plus.head(6)

    cols_rs = st.columns(3)
    for idx, (ticker, row) in enumerate(candidates.iterrows()):
        fig_rs = chart_rs_line(ticker, stock_data, benchmarks_df, cfg, row)
        if fig_rs:
            cols_rs[idx % 3].plotly_chart(fig_rs, use_container_width=True)


# ── TAB 4: DIVERGENCE ────────────────────────────────────────────────────────
with tab_div:
    st.subheader("Divergence Screen")
    st.caption("Stocks that went UP while SPY went DOWN — the strongest leadership signal")

    spy_vals = {
        "2W": (spy_ret_2w, "diverged_2w", "ret_2W"),
        "4W": (spy_ret_4w, "diverged_4w", "ret_4W"),
    }
    c1, c2 = st.columns(2)
    for col_ui, (window, (spy_ret, div_col, ret_col)) in zip([c1, c2], spy_vals.items()):
        with col_ui:
            spy_delta = f"{spy_ret*100:+.2f}%"
            label_color = "🔴" if spy_ret < 0 else "🟢"
            st.markdown(f"**{window} window** — SPY: {label_color} {spy_delta}")
            if spy_ret >= 0:
                st.info(f"SPY was positive over the last {window}. Divergence screen applies when SPY is down.")
            elif div_col not in ranked_plus.columns:
                st.warning("No divergence data available.")
            else:
                div_df = (ranked_plus[ranked_plus[div_col]]
                          .dropna(subset=[ret_col])
                          .sort_values(ret_col, ascending=False)
                          .head(20))
                if div_df.empty:
                    st.info("No stocks diverged upward in this window.")
                else:
                    st.caption(f"{len(div_df)} stocks up while SPY fell")
                    show_cols = {
                        "sector": "Sector", ret_col: f"{window} Ret",
                        "overall_strength": "Score", "sepa_signal": "SEPA",
                        "vcp_score": "VCP", "tt_criteria_passed": "TT/8",
                        "rs_line_new_high_63d": "RS 3M Hi",
                    }
                    sc = {k: v for k, v in show_cols.items() if k in div_df.columns}
                    d = div_df[list(sc.keys())].rename(columns=sc)
                    if f"{window} Ret" in d.columns:
                        d[f"{window} Ret"] = pd.to_numeric(d[f"{window} Ret"], errors="coerce").map(
                            lambda x: f"{x*100:+.2f}%" if pd.notna(x) else "—")

                    bool_c = [c for c in ["SEPA","RS 3M Hi"] if c in d.columns]
                    styled_div = (d.style
                                   .map(style_bool, subset=bool_c)
                                   .background_gradient(subset=["Score"] if "Score" in d.columns else [], cmap="RdYlGn", vmin=50, vmax=100)
                                   .format(na_rep="—", precision=1))
                    st.dataframe(styled_div, use_container_width=True, height=400)

    st.markdown("---")
    fig_div = chart_divergence(ranked_plus, cfg, window=div_window)
    if fig_div:
        st.plotly_chart(fig_div, use_container_width=True)
    else:
        st.info(f"SPY {div_window} return was positive — divergence chart only shows when SPY is down.")


# ── TAB 5: SECTORS ────────────────────────────────────────────────────────────
with tab_sec:
    st.subheader("Sector Analysis")

    fig_sec = chart_sector_bar(ranked_plus)
    if fig_sec:
        st.plotly_chart(fig_sec, use_container_width=True)

    st.markdown("---")
    st.subheader(f"Top {cfg['top_per_sector']} Stocks per Sector")

    if "sector" in ranked_plus.columns:
        lb = (ranked_plus.groupby("sector")["overall_strength"]
              .mean().sort_values(ascending=False).index.tolist())

        sector_show = [
            "overall_strength", "sepa_signal", "tt_criteria_passed",
            "rs_line_new_high_63d", "vcp_score",
            "ret_1M", "ret_3M", "ret_6M", "ret_12M",
            "rsi_14", "pct_from_52w_hi", "price",
        ]
        sector_show = [c for c in sector_show if c in ranked_plus.columns]
        sector_rename = {
            "overall_strength": "Score", "sepa_signal": "SEPA",
            "tt_criteria_passed": "TT/8", "rs_line_new_high_63d": "RS 3M Hi",
            "vcp_score": "VCP", "ret_1M": "1M", "ret_3M": "3M",
            "ret_6M": "6M", "ret_12M": "12M",
            "rsi_14": "RSI", "pct_from_52w_hi": "vs52WHi", "price": "Price",
        }

        for sector in lb:
            sub = ranked_plus[ranked_plus["sector"] == sector][sector_show].head(cfg["top_per_sector"]).rename(columns=sector_rename)
            avg = ranked_plus[ranked_plus["sector"] == sector]["overall_strength"].mean()
            n   = len(ranked_plus[ranked_plus["sector"] == sector])
            with st.expander(f"**{sector}** — avg score: {avg:.1f}  |  {n} stocks", expanded=False):
                ret_cols = [c for c in ["1M","3M","6M","12M","vs52WHi"] if c in sub.columns]
                for c in ret_cols:
                    sub[c] = pd.to_numeric(sub[c], errors="coerce").map(lambda x: f"{x*100:+.2f}%" if pd.notna(x) else "—")
                bool_c = [c for c in ["SEPA","RS 3M Hi"] if c in sub.columns]
                styled_sec = (sub.style
                                 .map(style_bool, subset=bool_c)
                                 .background_gradient(subset=["Score"] if "Score" in sub.columns else [], cmap="RdYlGn", vmin=50, vmax=100)
                                 .format(na_rep="—", precision=1))
                st.dataframe(styled_sec, use_container_width=True)


# ── TAB 6: CHARTS ─────────────────────────────────────────────────────────────
with tab_charts:
    st.subheader("Performance Charts")

    tf_select = st.radio("Timeframe for bar chart", tfs, index=tfs.index("6M") if "6M" in tfs else 0, horizontal=True)

    st.plotly_chart(chart_grouped_bar(ranked_plus, cfg, tf=tf_select, top_n=15), use_container_width=True)
    st.plotly_chart(chart_heatmap(ranked_plus, cfg, top_n=25), use_container_width=True)
    st.plotly_chart(chart_bubble(ranked_plus, cfg, tf="6M", top_n=40), use_container_width=True)


# ── TAB 7: ENTRY SETUPS ──────────────────────────────────────────────────────
with tab_entry:
    st.subheader("📍 Entry Setups — Today's Actionable Stocks")
    st.caption("Checks every screened stock against STEM MBA entry conditions on the current bar.")

    # ── Market regime panel ──────────────────────────────────────────────────
    spy_close  = benchmarks_df["SPY"].dropna()
    spy_now    = float(spy_close.iloc[-1])
    spy_sma50  = float(spy_close.rolling(50).mean().iloc[-1])
    spy_sma200 = float(spy_close.rolling(200).mean().iloc[-1])
    spy_above50  = spy_now > spy_sma50
    spy_above200 = spy_now > spy_sma200
    golden_cross = spy_sma50 > spy_sma200

    regime_score = sum([spy_above50, spy_above200, golden_cross])
    regime_label = {3: ("🟢 BULL — full size entries OK",       "#3fb950"),
                    2: ("🟡 CAUTION — reduce size or wait",     "#ffd700"),
                    1: ("🔴 BEAR — long entries not recommended","#f85149"),
                    0: ("🔴 BEAR — long entries not recommended","#f85149")}[regime_score]

    st.markdown(
        f"<div style='background:#161b22;border-radius:8px;padding:12px 18px;"
        f"border-left:4px solid {regime_label[1]};margin-bottom:1rem;'>"
        f"<b style='color:{regime_label[1]};font-size:15px;'>{regime_label[0]}</b>"
        f"<span style='color:#8b949e;font-size:12px;margin-left:18px;'>"
        f"SPY ${spy_now:.2f}  ·  50MA ${spy_sma50:.2f} {'✅' if spy_above50 else '❌'}  "
        f"·  200MA ${spy_sma200:.2f} {'✅' if spy_above200 else '❌'}  "
        f"·  50>200 {'✅' if golden_cross else '❌'}</span></div>",
        unsafe_allow_html=True,
    )

    # ── Scan button ──────────────────────────────────────────────────────────
    scan_col, info_col = st.columns([1, 3])
    scan_btn = scan_col.button("🔍 Scan Entry Setups", type="primary",
                                use_container_width=True)
    info_col.caption(
        f"Fetches OHLCV for all screened stocks (≈30 sec first time).  \n"
        "Uses the same cluster / bars / MFI / trend / HTF parameters as the Backtest sidebar."
    )

    if scan_btn:
        _setup_tickers = sorted(list(ranked_plus.index))
        _setup_key     = json.dumps(_setup_tickers)
        with st.spinner("Fetching OHLCV data…"):
            _ohlcv = fetch_ohlcv_cached(_setup_key, lookback_days=350)
        _setup_params = {
            "cluster_input": st.session_state.get("bt_cluster",  1.0),
            "bars_required": st.session_state.get("bt_bars",     2),
            "mfi_long_min":  st.session_state.get("bt_mfi_min",  45),
            "atr_mult":      st.session_state.get("bt_atr",      1.5),
            "use_trend":     st.session_state.get("bt_trend",    True),
            "use_htf":       st.session_state.get("bt_htf",      True),
            "capital":       st.session_state.get("bt_capital",  20_000),
            "pos_pct":       st.session_state.get("bt_pos_pct",  15),
        }
        with st.spinner("Evaluating entry conditions…"):
            st.session_state.entry_setups = compute_entry_setups(_ohlcv, _setup_params)

    es = st.session_state.entry_setups

    if es is None:
        st.info("Click **🔍 Scan Entry Setups** to check today's conditions.")
    elif es.empty:
        st.warning("No data returned. Try running the screener first.")
    else:
        ready_df   = es[es["ready"] == True]
        near_df    = es[(es["score"] >= 4) & (es["ready"] == False)]
        total_scan = len(es)

        # ── Summary KPIs ────────────────────────────────────────────────────────
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Stocks scanned",   total_scan)
        k2.metric("✅ Ready to enter", len(ready_df),
                  delta="All 6 conditions met" if len(ready_df) else "0 today")
        k3.metric("🟡 Near setup (5/6)", len(es[es["score"] == 5]))
        k4.metric("🟠 Building (4/6)",   len(es[es["score"] == 4]))

        # ── Ready-to-enter cards ─────────────────────────────────────────────────
        if not ready_df.empty:
            st.markdown("---")
            st.markdown("### ✅ Ready to Enter Now")
            cols_per_row = 3
            tickers_ready = list(ready_df.index)
            for row_start in range(0, len(tickers_ready), cols_per_row):
                card_cols = st.columns(cols_per_row)
                for col_idx, ticker in enumerate(tickers_ready[row_start:row_start + cols_per_row]):
                    r = ready_df.loc[ticker]
                    overall = float(ranked_plus.loc[ticker, "overall_strength"]) \
                              if ticker in ranked_plus.index else 0.0
                    with card_cols[col_idx]:
                        st.markdown(
                            f"<div style='background:#1a4731;border:1px solid #3fb950;"
                            f"border-radius:10px;padding:14px 16px;'>"
                            f"<div style='font-size:18px;font-weight:700;color:#3fb950;'>{ticker}</div>"
                            f"<div style='color:#8b949e;font-size:11px;margin-bottom:8px;'>"
                            f"Score {overall:.0f}/100  ·  MFI {r['MFI']:.0f}  ·  Vol×{r['vol_ratio']:.1f}</div>"
                            f"<table style='width:100%;font-size:12px;color:#e6edf3;'>"
                            f"<tr><td>Entry</td><td style='text-align:right;font-weight:600;'>${r['entry_$']:.2f}</td></tr>"
                            f"<tr><td>Stop</td><td style='text-align:right;color:#f85149;'>${r['stop_$']:.2f} (-{r['stop_%']:.1f}%)</td></tr>"
                            f"<tr><td>T1 target</td><td style='text-align:right;color:#58a6ff;'>${r['T1_$']:.2f}</td></tr>"
                            f"<tr><td>Position size</td><td style='text-align:right;'>${r['pos_size_$']:,.0f}"
                            f" ({r['shares']} shares)</td></tr>"
                            f"<tr><td>Dollar risk</td><td style='text-align:right;color:#ffd700;'>${r['risk_$']:,.0f}</td></tr>"
                            f"</table></div>",
                            unsafe_allow_html=True,
                        )

        # ── Full condition table ─────────────────────────────────────────────────
        st.markdown("---")

        view_filter = st.radio("Show", ["Ready (6/6)", "Near (≥ 4/6)", "All scanned"],
                               index=1, horizontal=True, key="es_filter")
        if view_filter == "Ready (6/6)":
            show_es = ready_df
        elif view_filter == "Near (≥ 4/6)":
            show_es = es[es["score"] >= 4]
        else:
            show_es = es

        bool_cols = ["✔ cluster", "✔ above MAs", "✔ bars", "✔ MFI", "✔ trend", "✔ HTF"]
        disp_cols = ["score", "close", "cluster_%"] + bool_cols + \
                    ["MFI", "entry_$", "stop_$", "T1_$", "stop_%",
                     "pos_size_$", "shares", "risk_$", "vol_ratio"]
        disp_cols = [c for c in disp_cols if c in show_es.columns]

        def _style_bool_entry(val):
            if val is True:  return "background-color:#1a4731; color:#3fb950; font-weight:bold; text-align:center"
            if val is False: return "background-color:#3d1f1f; color:#f85149; text-align:center"
            return ""

        def _style_score(val):
            try:
                v = int(val)
                if v == 6: return "background-color:#1a4731; color:#3fb950; font-weight:bold"
                if v >= 4: return "background-color:#3a3000; color:#ffd700"
                return "color:#8b949e"
            except Exception:
                return ""

        if not show_es.empty:
            styled_es = (
                show_es[disp_cols].style
                .map(_style_bool_entry, subset=bool_cols)
                .map(_style_score, subset=["score"])
                .format(na_rep="—", precision=2)
            )
            st.dataframe(styled_es, use_container_width=True, height=500)
        else:
            st.info("No stocks match the selected filter.")

        # ── SMA21 / SMA30 Cross Signal ───────────────────────────────────────────
        st.markdown("---")
        st.subheader("📈 SMA21 & SMA30 Cross Signal")
        st.caption(
            "All screened stocks where price closed **above both SMA21 and SMA30** today.  "
            "No SMA50 / SMA200 requirement — catches early-stage and recovery moves.  "
            "🟢 **Fresh cross** = was below at least one of them within the last 5 bars."
        )

        _cross_df = compute_ma_cross_signals(stock_data)

        if _cross_df.empty:
            st.info("No stocks above both SMA21 and SMA30 in the current screened universe.")
        else:
            _fresh_only = st.checkbox("Show fresh crosses only", value=False, key="cross_fresh_only")
            _cross_show = _cross_df[_cross_df["fresh"] == True] if _fresh_only else _cross_df

            # Summary
            n_fresh = int(_cross_df["fresh"].sum())
            n_total = len(_cross_df)
            fa, fb, fc, fd = st.columns(4)
            fa.metric("Above SMA21 & 30",  n_total)
            fb.metric("🟢 Fresh crosses",   n_fresh,
                      delta="crossed in last 5 bars")
            fc.metric("Also above SMA50",
                      int(_cross_df["above_50"].sum()) if "above_50" in _cross_df else "—")
            fd.metric("Also above SMA200",
                      int(_cross_df["above_200"].sum()) if "above_200" in _cross_df else "—")

            # Style helpers
            def _style_cross_pct(val):
                try:
                    v = float(val)
                    if v > 0:  return "color:#3fb950"
                    if v < 0:  return "color:#f85149"
                except Exception: pass
                return ""

            def _style_fresh(val):
                if val is True:
                    return "background-color:#1a4731;color:#3fb950;font-weight:bold;text-align:center"
                if val is False:
                    return "color:#8b949e;text-align:center"
                return ""

            def _style_above(val):
                if val is True:  return "color:#3fb950;text-align:center"
                if val is False: return "color:#f85149;text-align:center"
                return "color:#8b949e;text-align:center"

            def _style_rsi(val):
                try:
                    v = float(val)
                    if 50 <= v <= 70: return "color:#3fb950;font-weight:bold"
                    if v > 70:        return "color:#f85149"
                except Exception: pass
                return ""

            _cross_cols = ["close", "sma21", "sma30", "above_21_%", "above_30_%",
                           "vs_sma50_%", "vs_sma200_%", "above_50", "above_200",
                           "RSI", "vol_ratio", "fresh", "fresh_cross"]
            _cross_cols = [c for c in _cross_cols if c in _cross_show.columns]

            if not _cross_show.empty:
                styled_cross = (
                    _cross_show[_cross_cols].style
                    .map(_style_fresh,    subset=["fresh"]  if "fresh"     in _cross_cols else [])
                    .map(_style_above,    subset=["above_50","above_200"] if "above_50"   in _cross_cols else [])
                    .map(_style_cross_pct,subset=["above_21_%","above_30_%",
                                                  "vs_sma50_%","vs_sma200_%"])
                    .map(_style_rsi,      subset=["RSI"] if "RSI" in _cross_cols else [])
                    .format(na_rep="—", precision=2)
                )
                st.dataframe(styled_cross, use_container_width=True, height=480)
            else:
                st.info("No fresh crosses found. Uncheck 'Show fresh crosses only' to see all.")


# ── TAB 8: NOTES ─────────────────────────────────────────────────────────────
_TAG_OPTIONS = ["💡 Idea", "👁️ Watchlist", "📊 Trade", "⚠️ Risk", "📌 General"]
_TAG_CLASSES = {
    "💡 Idea":      "tag-idea",
    "👁️ Watchlist": "tag-watch",
    "📊 Trade":     "tag-trade",
    "⚠️ Risk":      "tag-risk",
    "📌 General":   "tag-general",
}

with tab_notes:
    st.subheader("📝 Trading Journal & Notes")
    st.caption("Notes are saved locally and persist between sessions.")

    # ── New note composer ────────────────────────────────────────────────────
    with st.expander("✏️  Write a new note", expanded=True):
        n_tag  = st.selectbox("Category", _TAG_OPTIONS, index=4, key="note_tag_sel",
                              label_visibility="collapsed")
        n_col1, n_col2 = st.columns([5, 1])
        with n_col1:
            n_title = st.text_input("Title (optional)", placeholder="e.g. NVDA breakout setup",
                                    key="note_title_inp", label_visibility="collapsed")
        with n_col2:
            st.write("")  # spacer
        n_body = st.text_area(
            "Note",
            placeholder="Write your thoughts, trade plan, observations…",
            height=140,
            key="note_body_inp",
            label_visibility="collapsed",
        )
        save_col, _, clear_col = st.columns([2, 5, 1])
        save_note = save_col.button("💾  Save Note", type="primary", use_container_width=True, key="save_note_btn")
        clear_note = clear_col.button("🗑️", help="Clear the form", key="clear_note_btn")

        if save_note:
            body_stripped = (n_body or "").strip()
            if body_stripped:
                new_note = {
                    "id":        str(_uuid.uuid4()),
                    "tag":       n_tag,
                    "title":     (n_title or "").strip(),
                    "body":      body_stripped,
                    "created":   datetime.now().strftime("%Y-%m-%d %H:%M"),
                }
                st.session_state.notes.insert(0, new_note)
                _save_notes(st.session_state.notes)
                st.success("Note saved!", icon="✅")
                st.rerun()
            else:
                st.warning("Note body cannot be empty.", icon="⚠️")

        if clear_note:
            st.rerun()

    st.markdown("---")

    # ── Search / filter bar ──────────────────────────────────────────────────
    notes = st.session_state.notes
    f_col1, f_col2 = st.columns([3, 1])
    with f_col1:
        search_q = st.text_input("🔍  Search notes", placeholder="Filter by keyword…",
                                 key="notes_search", label_visibility="collapsed")
    with f_col2:
        tag_filter = st.selectbox("Filter by tag", ["All"] + _TAG_OPTIONS,
                                  key="notes_tag_filter", label_visibility="collapsed")

    # Apply filters
    filtered = notes
    if tag_filter != "All":
        filtered = [n for n in filtered if n.get("tag") == tag_filter]
    if search_q.strip():
        q = search_q.strip().lower()
        filtered = [n for n in filtered
                    if q in n.get("body", "").lower()
                    or q in n.get("title", "").lower()]

    # ── Stats strip ──────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="display:flex;gap:24px;margin:12px 0 18px 0;flex-wrap:wrap;">
      <div style="font-size:12px;color:#64748b;font-weight:500;">
        <span style="color:#e2e8f0;font-size:18px;font-weight:700;">{len(notes)}</span>
        &nbsp;total notes
      </div>
      <div style="font-size:12px;color:#64748b;font-weight:500;">
        <span style="color:#60a5fa;font-size:18px;font-weight:700;">
          {len([n for n in notes if n.get('tag')=='💡 Idea'])}
        </span>&nbsp;ideas
      </div>
      <div style="font-size:12px;color:#64748b;font-weight:500;">
        <span style="color:#fbbf24;font-size:18px;font-weight:700;">
          {len([n for n in notes if n.get('tag')=='👁️ Watchlist'])}
        </span>&nbsp;on watchlist
      </div>
      <div style="font-size:12px;color:#64748b;font-weight:500;">
        <span style="color:#34d399;font-size:18px;font-weight:700;">
          {len([n for n in notes if n.get('tag')=='📊 Trade'])}
        </span>&nbsp;trades
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not filtered:
        if notes:
            st.info("No notes match your search / filter.", icon="🔍")
        else:
            st.info("No notes yet — write your first one above!", icon="📝")
    else:
        for note in filtered:
            nid     = note.get("id", "")
            tag     = note.get("tag", "📌 General")
            title   = note.get("title", "")
            body    = note.get("body", "")
            created = note.get("created", "")
            tag_cls = _TAG_CLASSES.get(tag, "tag-general")

            title_html = (
                f"<span style='font-size:15px;font-weight:700;"
                f"color:#f1f5f9;'>{title}</span>" if title else ""
            )
            tag_html = f"<span class='note-tag {tag_cls}'>{tag}</span>"

            st.markdown(
                f"""<div class='note-card'>
                  <div class='note-meta'>
                    {created} &nbsp;{tag_html}
                  </div>
                  {title_html}
                  {"<br>" if title else ""}
                  <div class='note-body'>{body}</div>
                </div>""",
                unsafe_allow_html=True,
            )

            # Delete button aligned right
            _, del_col = st.columns([10, 1])
            if del_col.button("🗑️", key=f"del_{nid}", help="Delete this note"):
                st.session_state.notes = [n for n in st.session_state.notes
                                          if n.get("id") != nid]
                _save_notes(st.session_state.notes)
                st.rerun()

    # ── Style Editor ─────────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("🎨  Style Editor — edit CSS live", expanded=False):
        st.caption(
            "Edit `style.css` directly here. "
            "Click **Save & Apply** and the page reloads with your changes."
        )

        _css_path = _os.path.join(_os.path.dirname(__file__), "style.css")

        # Load current CSS into editor on first open
        if "css_editor_content" not in st.session_state:
            try:
                with open(_css_path, "r", encoding="utf-8") as _f:
                    st.session_state.css_editor_content = _f.read()
            except Exception:
                st.session_state.css_editor_content = ""

        edited_css = st.text_area(
            "style.css",
            value=st.session_state.css_editor_content,
            height=520,
            key="css_editor_area",
            label_visibility="collapsed",
        )

        col_save, col_reset, col_reload = st.columns([2, 2, 4])

        if col_save.button("💾  Save & Apply", type="primary", use_container_width=True):
            try:
                with open(_css_path, "w", encoding="utf-8") as _f:
                    _f.write(edited_css)
                st.session_state.css_editor_content = edited_css
                # Clear injected CSS cache so _inject_css re-reads the file
                st.cache_data.clear()
                st.success("Saved! Reloading styles…", icon="✅")
                st.rerun()
            except Exception as _e:
                st.error(f"Could not save: {_e}")

        if col_reset.button("↩️  Reset to saved", use_container_width=True):
            try:
                with open(_css_path, "r", encoding="utf-8") as _f:
                    st.session_state.css_editor_content = _f.read()
                st.rerun()
            except Exception as _e:
                st.error(f"Could not reload: {_e}")

        col_reload.caption(f"📄 File: `{_css_path}`")

