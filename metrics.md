# Metric Calculation Formulas

---

## Part 1 — General Screener

### Price & Volume Basics

| Metric | Formula |
|---|---|
| `price` | Last closing price |
| `avg_vol_50` | Average daily volume over the last 50 bars |
| `52w_high` | Max close over the last 252 bars |
| `52w_low` | Min close over the last 252 bars |

---

### Moving Averages

Simple moving averages computed on the closing price series.

| Metric | Formula |
|---|---|
| `sma_21` | Mean of last 21 closes |
| `sma_30` | Mean of last 30 closes |
| `sma_200` | Mean of last 200 closes |

> Default periods are 21 / 30 / 200. These are configurable via `ma_stack` in the config.

---

### MA Stack OK (Hard Filter)

```
ma_stack_ok = (sma_21 > sma_200) AND (sma_30 > sma_200)
```

This is a hard filter — stocks that fail it are removed before scoring. It ensures the fast MAs are both above the slow MA, confirming a basic uptrend structure.

---

### Trend Score (0 – 5)

Each condition adds 1 point:

| # | Condition |
|---|---|
| 1 | Price > SMA_21 |
| 2 | Price > SMA_30 |
| 3 | Price > SMA_200 |
| 4 | SMA_21 > SMA_200 |
| 5 | SMA_30 > SMA_200 |

```
trend_score_raw = sum of above conditions (0–5)
```

---

### RSI (14-period)

```
delta      = daily price change
gain       = EWM( max(delta, 0),  com=13, adjust=False )   # Wilder's smoothing: alpha = 1/14
loss       = EWM( max(-delta, 0), com=13, adjust=False )
RS         = gain / loss
RSI_14     = 100 - (100 / (1 + RS))
```

> Wilder's smoothed average (`com = period − 1`) is used, **not** a simple rolling mean. This matches TradingView, IBD, and all standard charting platforms.

---

### Momentum Score (0 – 3)

| # | Condition | Points |
|---|---|---|
| 1 | 50 ≤ RSI_14 ≤ 70 | +1 |
| 2 | 55 ≤ RSI_14 ≤ 70 | +1 |
| 3 | Price ≥ 52w_high × 0.90 (within 10% of high) | +1 |

```
pct_from_52w_hi    = (price / 52w_high) - 1
momentum_score_raw = sum of above conditions (0–3)
```

---

### Returns & Outperformance (per timeframe)

Timeframes anchor to the **exact calendar date** from the last available bar, then snap to the nearest prior trading day — matching what TradingView and other chart platforms display.

| Label | Lookback |
|---|---|
| 1M | 1 calendar month |
| 3M | 3 calendar months |
| 6M | 6 calendar months |
| 12M | 12 calendar months |
| 2W | 2 calendar weeks |
| 4W | 4 calendar weeks |

```
target_date            = last_bar_date - offset
base_price             = last close on or before target_date

stock_return_{TF}      = (price_today / base_price) - 1

benchmark_return_{TF}  = (bench_today / bench_base_price) - 1

outperf_{bench}_{TF}   = stock_return_{TF} - benchmark_return_{TF}

rs_ratio_{bench}_{TF}  = (1 + stock_return_{TF}) / (1 + benchmark_return_{TF})
```

> `rs_ratio` > 1.0 means the stock outperformed the benchmark multiplicatively over that period.

---

### Beats All Benchmarks (flag)

```
beats_all_{TF} = (outperf_SPY_{TF} > 0) AND (outperf_QQQ_{TF} > 0)
```

By default the screener only keeps stocks where `beats_all_6M = True`.

---

### Percentile Ranks

All raw scores are converted to percentile ranks (0–100) within the surviving universe before being combined:

```
rs_rank_{TF}      = percentile_rank( rs_ratio_{primary}_{TF} )    × 100
outperf_rank_{TF} = percentile_rank( outperf_{primary}_{TF} )     × 100
trend_rank        = percentile_rank( trend_score_raw )             × 100
momentum_rank     = percentile_rank( momentum_score_raw )          × 100
```

---

### Composite Score (per timeframe)

Factor weights (defaults, normalized to sum = 1):

| Factor | Default weight |
|---|---|
| rs_rank | 0.35 |
| outperf_rank | 0.25 |
| trend_rank | 0.20 |
| momentum_rank | 0.20 |

```
composite_{TF} = w_rs      × rs_rank_{TF}
               + w_outperf × outperf_rank_{TF}
               + w_trend   × trend_rank
               + w_mom     × momentum_rank
```

---

### Overall Strength Score (final score)

Timeframe weights (defaults, normalized to sum = 1):

| Timeframe | Default weight |
|---|---|
| 1M | 0.10 |
| 3M | 0.25 |
| 6M | 0.30 |
| 12M | 0.35 |

```
overall_strength = weighted_average( composite_{TF} across all TFs )
```

Missing timeframes (insufficient price history) are excluded and remaining weights are renormalized automatically.

---

### Sector Rank

```
sector_rank = percentile_rank( overall_strength ) within the stock's GICS sector × 100
```

---

## Part 2 — Minervini / SEPA

### Minervini Trend Template (8 criteria, all must pass)

| # | Field | Condition |
|---|---|---|
| C1 | `tt_c1` | Price > MA50 |
| C2 | `tt_c2` | MA50 > MA150 |
| C3 | `tt_c3` | MA150 > MA200 |
| C4 | `tt_c4` | Price > MA150 |
| C5 | `tt_c5` | Price > MA200 |
| C6 | `tt_c6` | MA200 slope is positive (MA200 today > MA200 21 bars ago) |
| C7 | `tt_c7` | Price ≥ 52w_low × 1.30 (at least 30% above 52-week low) |
| C8 | `tt_c8` | Price ≥ 52w_high × 0.75 (within 25% of 52-week high) |

```
pct_above_52w_low  = (price / 52w_low)  - 1      # must be ≥ 0.30
pct_from_52w_hi    = (price / 52w_high) - 1      # must be ≥ -0.25

ma200_slope        = MA200_today - MA200_21_bars_ago   # must be > 0

tt_criteria_passed = count of conditions that are True  (0–8)
tt_pass            = True only if tt_criteria_passed == 8
```

---

### RS Line

```
rs_line = stock_close / SPY_close   (aligned by date, daily ratio)
```

This is a raw price ratio — it rises when the stock outperforms SPY and falls when it underperforms.

#### RS Line Metrics

```
rs_line_new_high_52w  = rs_line_today >= max(rs_line, last 252 bars) × 0.999
rs_line_new_high_63d  = rs_line_today >= max(rs_line, last 63 bars)  × 0.999
rs_line_pct_from_hi   = (rs_line_today / 52w_rs_high) - 1
rs_line_slope_21d     = (rs_line_today / rs_line_21_bars_ago) - 1
```

> The 0.999 tolerance allows for floating-point noise at the exact high.

---

### VCP Score (Volatility Contraction Pattern, 0 – 5)

VCP is detected by checking whether price volatility and volume are contracting progressively. Each condition adds 1 point:

```
CV(n) = std(close, last n bars) / mean(close, last n bars)   # coefficient of variation
```

| # | Signal | Condition |
|---|---|---|
| 1 | `vol2w<4w` | CV(10) < CV(20) — 2-week volatility < 4-week volatility |
| 2 | `vol4w<8w` | CV(20) < CV(40) — 4-week volatility < 8-week volatility |
| 3 | `vol8w<12w` | CV(40) < CV(60) — 8-week volatility < 12-week volatility |
| 4 | `range_tight` | Price range (high−low)/mean over 10 bars < same over 20 bars |
| 5 | `vol_dryup` | avg_volume(10) < avg_volume(40) × 0.85 — recent volume < 85% of 40-day average |

```
range_10 = (max(close,10) - min(close,10)) / mean(close,10)
range_20 = (max(close,20) - min(close,20)) / mean(close,20)

vcp_vol_ratio = avg_volume(10) / avg_volume(40)

vcp_score = sum of conditions above  (0–5)
```

---

### SEPA Signal (combined flag)

```
sepa_signal = tt_pass
           AND rs_line_new_high_63d
           AND vcp_score >= 2
```

A stock must pass all 8 Minervini criteria, have its RS line making a 3-month high, and show at least 2 VCP contraction signals.

---

### Market Divergence Flags

These flag stocks that rose while the broad market fell — showing unusual relative strength.

```
spy_ret_2w = (SPY_today / SPY_10_bars_ago) - 1
spy_ret_4w = (SPY_today / SPY_21_bars_ago) - 1

diverged_2w = (stock ret_2W > 0) AND (spy_ret_2w < 0)
diverged_4w = (stock ret_4W > 0) AND (spy_ret_4w < 0)
```
