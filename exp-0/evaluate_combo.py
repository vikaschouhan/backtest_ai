import numpy as np, pandas as pd, warnings
from evaluate import SETS, GRID, INIT, evaluate
from datasets import RESULTS
warnings.filterwarnings("ignore")
COMBOS = {
    "baseline":              dict(),
    "vol sizing":            dict(bt=dict(vol_size=True)),
    "SMA + vol":             dict(bt=dict(sma=True, vol_size=True)),
    "SMA + vol + confirm":   dict(sp=dict(confirm_atr=0.5), bt=dict(sma=True, vol_size=True)),
}
def bh(frames, ppy, start=INIT):
    out = []
    for d in frames:
        c, o = d.Close.to_numpy(), d.Open.to_numpy()
        e = np.r_[1.0, c[start:] / o[start]]
        dr = np.diff(e) / e[:-1]; dd = (e / np.maximum.accumulate(e) - 1).min()
        yrs = (len(e) - 1) / ppy; cagr = e[-1] ** (1 / yrs) - 1
        out.append((np.sqrt(ppy) * dr.mean() / dr.std(), 100 * cagr, 100 * dd))
    return np.median(np.array(out), axis=0)
rows = []
for name in [k for k in ("AAPL d 1y", "GOOG d 8.5y", "EURUSD 1h", "RandomWalk x20") if k in SETS]:
    frames, ppy, cost, sma_len = SETS[name]
    s, c_, d_ = bh(frames, ppy)
    rows.append(dict(data=name, mode="-", variant="buy & hold", sharpe=s, cagr=c_, maxdd=d_))
    for mode in ("long_short", "long_only"):
        base = None
        for vn, var in COMBOS.items():
            r = evaluate(frames, ppy, cost, sma_len, var, mode)
            if base is None: base = r
            rows.append(dict(data=name, mode=mode, variant=vn, sharpe=r.sharpe_ann.median(),
                             cagr=r["cagr_%"].median(), maxdd=r["max_drawdown_%"].median(),
                             trades=r.trades.median(), exposure=r["exposure_%"].median(),
                             beats_base=(r.sharpe_ann.values > base.sharpe_ann.values).mean() if vn != "baseline" else np.nan,
                             dd_better=(r["max_drawdown_%"].values > base["max_drawdown_%"].values).mean() if vn != "baseline" else np.nan))
res = pd.DataFrame(rows); res.to_csv(RESULTS / "eval_combo.csv", index=False)
pd.set_option("display.width", 200); print(res.round(2).to_string(index=False))
