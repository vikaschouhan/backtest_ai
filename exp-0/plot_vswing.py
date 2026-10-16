import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from vswing import PriceData, StrategyParams, run
from datasets import RESULTS

# --- simulated OHLC: GBM with alternating drift regimes ---
rng = np.random.default_rng(42)
n = 250
drift = np.repeat(rng.choice([-0.004, 0.0, 0.004], n // 25 + 1), 25)[:n]
ret = drift + rng.normal(0, 0.012, n)
c = 1000 * np.exp(np.cumsum(ret))
o = np.r_[1000, c[:-1]] * (1 + rng.normal(0, 0.002, n))
h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.005, n)))
l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.005, n)))

sp = StrategyParams(threshold=23.0, init_bars=20)
r = run(PriceData(o, h, l, c), sp)

# --- locate the bar where each promoted swing was made (plotting only) ---
flips = np.where(~np.isnan(r.long_signals) | ~np.isnan(r.short_signals))[0]
sh_pts, sl_pts, start = [], [], sp.init_bars
for i in flips:
    seg = slice(start, i + 1)
    if not np.isnan(r.short_signals[i]):          # up leg ended -> swing high
        k = start + np.argmax(h[seg]); sh_pts.append((k, h[k]))
    else:                                          # down leg ended -> swing low
        k = start + np.argmin(l[seg]); sl_pts.append((k, l[k]))
    start = i

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#8a8984", "#e6e5e1"
x = np.arange(n)
fig, ax = plt.subplots(figsize=(13, 6.5), dpi=150)
fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")

# candles
up = c >= o
ax.vlines(x, l, h, color=MUTED, lw=0.7, zorder=1)
ax.bar(x[up], (c - o)[up], bottom=o[up], width=0.65, color="#cfcec9", edgecolor=MUTED, lw=0.5, zorder=2)
ax.bar(x[~up], (o - c)[~up], bottom=c[~up], width=0.65, color="#6f6e69", edgecolor="#6f6e69", lw=0.5, zorder=2)

# reference swing levels and virtual stops
ax.step(x, r.swing_high, where="post", color=ORANGE, lw=1.2, alpha=0.55, label="Swing high level")
ax.step(x, r.swing_low, where="post", color=BLUE, lw=1.2, alpha=0.55, label="Swing low level")
ax.step(x, r.v_sl, where="post", color=ORANGE, lw=0.9, ls=(0, (1, 2)), label="Virtual swing low")
ax.step(x, r.v_sh, where="post", color=BLUE, lw=0.9, ls=(0, (1, 2)), label="Virtual swing high")
ax.step(x, r.rev_dn, where="post", color=ORANGE, lw=1.3, ls=(0, (4, 2)), label="Reversal level (up trend)")
ax.step(x, r.rev_up, where="post", color=BLUE, lw=1.3, ls=(0, (4, 2)), label="Reversal level (down trend)")

# swing points
if sh_pts:
    k, v = zip(*sh_pts)
    ax.scatter(k, np.array(v) * 1.006, marker="v", s=55, color=ORANGE, edgecolor="#fcfcfb", lw=1.2, zorder=5, label="Swing high")
if sl_pts:
    k, v = zip(*sl_pts)
    ax.scatter(k, np.array(v) * 0.994, marker="^", s=55, color=BLUE, edgecolor="#fcfcfb", lw=1.2, zorder=5, label="Swing low")

ax.axvspan(-0.5, sp.init_bars - 0.5, color=GRID, alpha=0.6, lw=0)
ax.text(sp.init_bars / 2, ax.get_ylim()[1], "init", ha="center", va="top", color=MUTED, fontsize=9)

ax.set_title(f"Adaptive Virtual Swing — simulated data  (threshold {sp.threshold:g}%, "
             f"{len(sh_pts)} swing highs, {len(sl_pts)} swing lows)", loc="left", color=INK, fontsize=12)
ax.set_xlabel("Bar", color="#52514e"); ax.set_ylabel("Price", color="#52514e")
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color("#c3c2b7")
ax.tick_params(colors="#52514e")
ax.set_xlim(-1, n)
ax.legend(loc="upper left", ncol=4, fontsize=8.5, frameon=False, bbox_to_anchor=(0, -0.09))
fig.tight_layout()
fig.savefig(RESULTS / "vswing_swings.png", facecolor=fig.get_facecolor())
print(len(sh_pts), len(sl_pts))
