# exp-0 — Adaptive Virtual Swing Crossover

A causal (no-lookahead) trend-reversal indicator in Python/Numba, a next-bar-open
backtester, and a study of risk overlays. Full spec, change history and findings:
[`CONTEXT.txt`](CONTEXT.txt).

## The indicator in one paragraph

In an uptrend it tracks the highest high (`ath`) and a **virtual swing low** that starts
at the last swing low and only ever moves up: when a bar's pullback from `ath` is
≤ `threshold` of the cushion (`ath − vsl`), the floor is reset to `ath − pullback / threshold`.
A close below the **reversal line** `vsl + (ath − vsl) × threshold` flips the trend down,
and `ath` becomes the new swing high. Downtrends are the mirror image. `threshold` is in
percent (default 23).

## Files

| File | What it is |
|---|---|
| `vswing.py` | Engine (`run`, `StrategyParams`, `SwingResult`). Optional: `rev_threshold`, `min_cushion_atr`, `confirm_atr` — all off by default |
| `backtest.py` | Signal on close → fill next open, costs, long/short or long-only, optional SMA regime filter and volatility sizing; `stats()` |
| `evaluate.py`, `evaluate_combo.py` | Compare each idea vs baseline across thresholds 15–35 and datasets (medians) |
| `run_aapl.py`, `plot_*.py` | AAPL backtest and charts |
| `tests/test_vswing.py` | Causality (prefix-equality), swing promotion, ratchet and level-ordering checks |
| `results/` | Charts and CSVs produced by the scripts |

## Run

```bash
pip install -r requirements.txt
python tests/test_vswing.py
python evaluate_combo.py          # GOOG, EURUSD, random walks (+ AAPL if present)
python plot_goog.py
```

**Data is not committed.** GOOG / EURUSD / BTCUSD come from the `backtesting` package's
bundled samples. For the AAPL scripts, put `data/AAPL_daily.csv`
(`Date,Open,High,Low,Close,Volume`) in this folder.

## Key findings (details in `CONTEXT.txt`)

- Volatility sizing and an SMA trend-regime filter cut drawdowns on every dataset
  tested, including random walks — they manage risk, not prediction.
- GOOG 2004–13, threshold 23, long/short with overlays: max drawdown −25% vs −65% for
  buy & hold, but growth ~3.3× vs ~7×.
- On random walks the raw signal shows no edge. Next step: 10+ years of NSE index data
  with walk-forward validation.
