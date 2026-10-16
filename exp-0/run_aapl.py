import numpy as np, pandas as pd
from backtest import backtest, stats
from vswing import StrategyParams
from datasets import AAPL, RESULTS

df = pd.read_csv(AAPL, parse_dates=["Date"])
sp = StrategyParams(23.0, 20)
start = sp.init_bars
c, o = df.Close.to_numpy(), df.Open.to_numpy()

bh = np.r_[np.ones(start), c[start:] / o[start]]           # buy & hold from first tradable open
bh_dd = bh[start:] / np.maximum.accumulate(bh[start:]) - 1
dr = np.diff(bh[start:]) / bh[start:-1]
print(f"Period: {df.Date.iloc[start].date()} -> {df.Date.iloc[-1].date()}  ({len(df)-start} bars)")
print(f"Buy&hold: return {100*(bh[-1]-1):.1f}%  maxDD {100*bh_dd.min():.1f}%  sharpe {np.sqrt(252)*dr.mean()/dr.std():.2f}\n")

res = {}
for mode in ("long_short", "long_only"):
    r, pos, eq, tr = backtest(df, sp, mode, cost_bps=5)
    res[mode] = (r, pos, eq, tr)
    s = stats(eq, pos, tr, start)
    print(mode, " ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in s.items()))
print()
tr = res["long_short"][3]
print(tr.assign(entry=tr.entry.dt.date, exit=tr.exit.dt.date, ret=(100*tr.ret).round(2)).to_string(index=False))

print("\nThreshold sensitivity (long_short, init 20, 5 bps):")
rows = []
for th in (10, 15, 20, 23, 25, 30, 35, 40, 45):
    r_, p_, e_, t_ = backtest(df, StrategyParams(th, 20), "long_short", 5)
    s_ = stats(e_, p_, t_, start)
    rows.append((th, round(s_["total_return_%"], 1), round(s_["max_drawdown_%"], 1), s_["trades"], round(s_["win_rate_%"], 0)))
print(pd.DataFrame(rows, columns=["thr%", "ret%", "maxDD%", "trades", "win%"]).to_string(index=False))

tr.to_csv(RESULTS / "aapl_trades.csv", index=False)
