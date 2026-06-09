# Stock Screener Dashboard

A Streamlit dashboard for stock screening, trend analysis, relative strength ranking, and AlphaTrend-style backtesting. The app combines price/volume metrics, moving-average filters, RSI, benchmark outperformance, Minervini-style trend-template checks, VCP signals, and interactive Plotly visualizations.

## Features

- Stock screener with configurable technical filters
- Relative strength and benchmark outperformance analysis versus SPY/QQQ
- Moving average stack checks and trend scoring
- RSI and momentum scoring
- Minervini / SEPA trend-template signals
- VCP-style volatility contraction scoring
- Interactive Streamlit dashboard with Plotly charts
- Backtesting notebooks for AlphaTrend strategy experiments
- Local notes support for watchlist/research workflow

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- yfinance
- Requests

## Project Structure

```text
dashboard.py                         Main Streamlit screener dashboard
backtest_dashboard.py                Backtesting dashboard
alphatrend_backtest.ipynb            AlphaTrend backtest notebook
alphatrend_backtest_per_ticker.ipynb Per-ticker AlphaTrend backtest notebook
metrics.md                           Metric and formula documentation
requirements.txt                     Python dependencies
style.css                            Dashboard styling
run_backtest.bat                     Windows helper script
```

## Getting Started

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run dashboard.py
```

## Metrics

See [metrics.md](metrics.md) for detailed formulas covering moving averages, RSI, relative strength, composite scoring, Minervini trend-template criteria, RS line metrics, VCP scoring, and market divergence flags.

## Notes

This project is for research and educational use. It is not financial advice.
