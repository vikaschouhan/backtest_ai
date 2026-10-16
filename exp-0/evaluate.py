"""Compare each enhancement against the baseline at the SAME threshold, across a grid
of thresholds and several datasets. Reports medians so no single lucky setting wins."""
import numpy as np, pandas as pd, warnings
from backtest import backtest, stats
from vswing import StrategyParams
from datasets import AAPL, RESULTS, sample
warnings.filterwarnings("ignore")

def load(f, date_col=None):
    d = pd.read_csv(f, index_col=None if date_col else 0, parse_dates=[date_col] if date_col else True)
    return d.reset_index(drop=True) if date_col else d

def rw(seed, n=2000):
    g = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(g.normal(0, 0.015, n)))
    o = np.r_[100, c[:-1]] * (1 + g.normal(0, 0.003, n))
    h = np.maximum(o, c) * (1 + np.abs(g.normal(0, 0.006, n)))
    l = np.minimum(o, c) * (1 - np.abs(g.normal(0, 0.006, n)))
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c})

SETS = {   # name: (frames, periods/yr, cost bps, sma_len)
    "GOOG d 8.5y":  ([load(sample("GOOG"))], 252, 5, 100),
    "EURUSD 1h":    ([load(sample("EURUSD"))], 6240, 1, 100),
    "BTC monthly":  ([load(sample("BTCUSD"))], 12, 10, 10),
    "RandomWalk x20": ([rw(s) for s in range(20)], 252, 5, 100),
}
if AAPL.exists():
    SETS = {"AAPL d 1y": ([load(AAPL, "Date")], 252, 5, 50), **SETS}
VARIANTS = {
    "baseline":            dict(),
    "rev line 10%":        dict(sp=dict(rev_threshold=10.0)),
    "min cushion 2 ATR":   dict(sp=dict(min_cushion_atr=2.0)),
    "confirm 0.5 ATR":     dict(sp=dict(confirm_atr=0.5)),
    "SMA regime filter":   dict(bt=dict(sma=True)),
    "vol sizing":          dict(bt=dict(vol_size=True)),
}
GRID = (15, 20, 25, 30, 35)
INIT = 20

def evaluate(frames, ppy, cost, sma_len, var, mode):
    rows = []
    for th in GRID:
        spk = dict(threshold=th, init_bars=INIT, **var.get("sp", {}))
        btk = var.get("bt", {})
        for d in frames:
            _, pos, eq, tr = backtest(d, StrategyParams(**spk), mode, cost,
                                      sma_len=sma_len if btk.get("sma") else 0,
                                      vol_size=btk.get("vol_size", False))
            s = stats(eq, pos, tr, INIT, ppy); s["th"] = th
            rows.append(s)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    out = []
    for mode in ("long_short", "long_only"):
        for name, (frames, ppy, cost, sma_len) in SETS.items():
            base = evaluate(frames, ppy, cost, sma_len, VARIANTS["baseline"], mode)
            for vname, var in VARIANTS.items():
                r = base if vname == "baseline" else evaluate(frames, ppy, cost, sma_len, var, mode)
                beat = (r.sharpe_ann.values > base.sharpe_ann.values).mean() if vname != "baseline" else np.nan

                out.append(dict(mode=mode, data=name, variant=vname,
                                sharpe=r.sharpe_ann.median(), cagr=r["cagr_%"].median(),
                                maxdd=r["max_drawdown_%"].median(), pf=r.profit_factor.replace(np.inf, np.nan).median(),
                                trades=r.trades.median(),
                                beats_base=beat,
                                dd_better=(r["max_drawdown_%"].values > base["max_drawdown_%"].values).mean() if vname != "baseline" else np.nan))
    res = pd.DataFrame(out)
    res.to_csv(RESULTS / "eval_results.csv", index=False)
    pd.set_option("display.width", 200)
    for mode in ("long_short", "long_only"):
        print(f"\n===== {mode}: medians over thresholds {GRID} (random walk: over 20 series too) =====")
        print(res[res["mode"] == mode].drop(columns="mode").round(2).to_string(index=False))
