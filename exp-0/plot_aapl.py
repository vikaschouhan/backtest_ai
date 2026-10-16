import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from backtest import backtest
from vswing import StrategyParams
from datasets import AAPL, RESULTS

df = pd.read_csv(AAPL, parse_dates=["Date"])
sp = StrategyParams(23.0, 20); start = sp.init_bars
o, h, l, c = (df[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
r, pos, eq_ls, tr = backtest(df, sp, "long_short", 5)
_, _, eq_lo, _ = backtest(df, sp, "long_only", 5)
bh = np.r_[np.ones(start), c[start:] / o[start]]

BLUE, ORANGE, AQUA, INK, SEC, MUTED, GRID, BG = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
x = np.arange(len(df))
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 9), dpi=150, sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
fig.patch.set_facecolor(BG)

for a in (ax, ax2):
    a.set_facecolor(BG); a.grid(axis="y", color=GRID, lw=0.6); a.set_axisbelow(True)
    for s in ("top", "right"): a.spines[s].set_visible(False)
    for s in ("left", "bottom"): a.spines[s].set_color("#c3c2b7")
    a.tick_params(colors=SEC)

# shade position
for i in range(start, len(df)):
    if pos[i] != 0:
        ax.axvspan(i - 0.5, i + 0.5, color=BLUE if pos[i] > 0 else ORANGE, alpha=0.06, lw=0)

up = c >= o
ax.vlines(x, l, h, color=MUTED, lw=0.7, zorder=1)
ax.bar(x[up], (c - o)[up], bottom=o[up], width=0.65, color="#cfcec9", edgecolor=MUTED, lw=0.5, zorder=2)
ax.bar(x[~up], (o - c)[~up], bottom=c[~up], width=0.65, color="#6f6e69", edgecolor="#6f6e69", lw=0.5, zorder=2)
ax.step(x, r.rev_dn, where="post", color=ORANGE, lw=1.3, ls=(0, (4, 2)), label="Reversal level (up trend)")
ax.step(x, r.rev_up, where="post", color=BLUE, lw=1.3, ls=(0, (4, 2)), label="Reversal level (down trend)")

L = np.where(~np.isnan(r.long_signals))[0]; S = np.where(~np.isnan(r.short_signals))[0]
ax.scatter(L, l[L] * 0.985, marker="^", s=70, color=BLUE, edgecolor=BG, lw=1.2, zorder=5, label="Long signal (fill next open)")
ax.scatter(S, h[S] * 1.015, marker="v", s=70, color=ORANGE, edgecolor=BG, lw=1.2, zorder=5, label="Short signal (fill next open)")
ax.axvspan(-0.5, start - 0.5, color=GRID, alpha=0.6, lw=0)
ax.text(start / 2, 242, "init", ha="center", va="bottom", color=MUTED, fontsize=9)
ax.set_ylabel("Price (USD)", color=SEC)
ax.set_title("AAPL daily — Adaptive Virtual Swing backtest (threshold 23%, 5 bps/side)", loc="left", color=INK, fontsize=12)
ax.legend(loc="upper left", ncol=2, fontsize=8.5, frameon=False)

def pct(e): return 100 * (e - 1)   # all curves start at 1.0 = first tradable open
ax2.plot(x[start:], pct(bh)[start:], color=MUTED, lw=1.6, label="Buy & hold")
ax2.plot(x[start:], pct(eq_ls)[start:], color=BLUE, lw=2, label="Long/short")
ax2.plot(x[start:], pct(eq_lo)[start:], color=AQUA, lw=2, label="Long only")
for e, col, lab in ((bh, MUTED, "Buy & hold"), (eq_ls, BLUE, "Long/short"), (eq_lo, AQUA, "Long only")):
    ax2.annotate(f"{lab} {pct(e)[-1]:+.1f}%", (x[-1], pct(e)[-1]), xytext=(6, 0), textcoords="offset points",
                 va="center", fontsize=8.5, color=INK)
ax2.axhline(0, color="#c3c2b7", lw=0.8)
ax2.set_ylabel("Return (%)", color=SEC)

ticks = [i for i in range(len(df)) if i == 0 or df.Date.iloc[i].month != df.Date.iloc[i - 1].month]
ax2.set_xticks(ticks); ax2.set_xticklabels([df.Date.iloc[i].strftime("%b %y") for i in ticks])
ax2.set_xlim(-1, len(df) + 38)
fig.tight_layout()
fig.savefig(RESULTS / "aapl_backtest.png", facecolor=BG)
