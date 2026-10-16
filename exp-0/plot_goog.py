import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.ticker
from backtest import backtest, stats
from vswing import StrategyParams
from datasets import RESULTS, sample
d = pd.read_csv(sample("GOOG"), index_col=0, parse_dates=True); S = 20
c, o = d.Close.to_numpy(), d.Open.to_numpy()
bh = np.r_[np.ones(S), c[S:] / o[S]]
runs = {
  "Buy & hold": (bh, None),
  "Baseline long/short": backtest(d, StrategyParams(23, S), "long_short", 5)[2:0:-1],
  "Long/short + SMA filter + vol sizing + confirm": backtest(d, StrategyParams(23, S, confirm_atr=0.5), "long_short", 5, sma_len=100, vol_size=True)[2:0:-1],
}
BG, INK, SEC, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
cols = {"Buy & hold": "#8a8984", "Baseline long/short": "#eb6834", "Long/short + SMA filter + vol sizing + confirm": "#2a78d6"}
fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 7.5), dpi=150, sharex=True, gridspec_kw={"height_ratios": [2, 1]})
fig.patch.set_facecolor(BG)
for a in (a1, a2):
    a.set_facecolor(BG); a.grid(axis="y", color=GRID, lw=.6); a.set_axisbelow(True)
    for s_ in ("top", "right"): a.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"): a.spines[s_].set_color("#c3c2b7")
    a.tick_params(colors=SEC)
x = d.index
for k, (e, _) in runs.items():
    e = e[S:] / e[S]; dd = 100 * (e / np.maximum.accumulate(e) - 1)
    a1.plot(x[S:], e, color=cols[k], lw=1.8 if "Buy" not in k else 1.4, label=f"{k}  (max DD {dd.min():.0f}%)")
    a2.plot(x[S:], dd, color=cols[k], lw=1.4)
a1.set_yscale("log"); a1.set_ylabel("Growth of 1 (log)", color=SEC)
a1.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); a1.set_yticks([0.5, 1, 2, 4, 8]); a1.set_yticklabels(["0.5", "1", "2", "4", "8"])
a2.set_ylabel("Drawdown (%)", color=SEC)
a1.set_title("GOOG daily 2004–2013 — threshold 23%, 5 bps/side: baseline vs risk overlays", loc="left", color=INK, fontsize=12)
a1.legend(loc="upper left", frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(RESULTS / "goog_overlays.png", facecolor=BG)
