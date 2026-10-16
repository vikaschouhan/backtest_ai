"""
gate_tests.py — exercise regime_gate.RegimeGate on synthetic price processes.

Scenarios
    ou_strong     OU, theta=0.10  (half-life ~6.6 bars)      -> gate SHOULD be on
    ou_weak       OU, theta=0.02  (half-life ~34 bars)       -> borderline
    random_walk   pure random walk                           -> gate should be OFF
    trend         random walk + drift                        -> gate should be OFF
    regime_switch OU -> trend -> OU -> random walk           -> on/off by segment

For each scenario:
    * per-bar gate decisions (state, score)
    * a toy z-score mean-reversion strategy, run ungated and gated
    * one PNG figure: price + gate state, score, cumulative PnL
Plus a Monte-Carlo summary across seeds (fig_summary.png, summary.csv).
All outputs go to results/.  Usage: python gate_tests.py [n_seeds=10]
"""

from __future__ import annotations

import csv
import math
import os
import sys
from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from regime_gate import (RegimeGate, Features, Regime, hurst_exponent,
                         ou_half_life, adf_pvalue, adx)

# ---------------- palette (reference dataviz palette, light mode) ----------
SURF = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e0"
C_MR = "#2a78d6"       # slot 1 blue   -> mean-reverting
C_TR = "#eb6834"       # slot 2 orange -> trending
C_TX = "#b4b2a9"       # neutral gray  -> transitional
C_PRICE = "#3d3d3a"
C_GATED = "#2a78d6"
C_RAW = "#eb6834"
STATE_COLOR = {Regime.MEAN_REVERTING: C_MR, Regime.TRANSITIONAL: C_TX,
               Regime.TRENDING: C_TR}

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "normal", "legend.frameon": False,
})

WINDOW = 120
N = 1500
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ---------------- price generators ----------------
def ou(n, theta, sigma, mu, x0, rng):
    x = np.empty(n); x[0] = x0
    for i in range(1, n):
        x[i] = x[i-1] + theta * (mu - x[i-1]) + sigma * rng.standard_normal()
    return x


def rw(n, sigma, drift, x0, rng):
    return x0 + np.cumsum(drift + sigma * rng.standard_normal(n))


def gen(name, rng, n=N):
    """Return (close, truth) where truth[t] is True if the generating process
    at t is mean-reverting (None for scenarios without segments)."""
    if name == "ou_strong":
        return ou(n, 0.10, 0.6, 100, 100, rng), np.ones(n, bool)
    if name == "ou_weak":
        return ou(n, 0.02, 0.6, 100, 100, rng), np.ones(n, bool)
    if name == "random_walk":
        return rw(n, 0.6, 0.0, 100, rng), np.zeros(n, bool)
    if name == "trend":
        return rw(n, 0.6, 0.12, 100, rng), np.zeros(n, bool)
    if name == "regime_switch":
        seg = n // 4
        a = ou(seg, 0.10, 0.6, 100, 100, rng)
        b = rw(seg, 0.6, 0.20, a[-1], rng)
        c = ou(seg, 0.10, 0.6, b[-1], b[-1], rng)
        d = rw(n - 3 * seg, 0.6, 0.0, c[-1], rng)
        truth = np.r_[np.ones(seg, bool), np.zeros(seg, bool),
                      np.ones(seg, bool), np.zeros(n - 3 * seg, bool)]
        return np.r_[a, b, c, d], truth
    raise ValueError(name)


# ---------------- run gate ----------------
@dataclass
class Run:
    close: np.ndarray
    truth: np.ndarray
    regime: list
    score: np.ndarray
    allow: np.ndarray
    force: np.ndarray


def run_gate(close, truth, rng):
    hi = close + np.abs(rng.normal(0, 0.3, len(close)))
    lo = close - np.abs(rng.normal(0, 0.3, len(close)))
    adx_arr = adx(hi, lo, close)
    gate = RegimeGate()
    n = len(close)
    regime = [Regime.TRENDING] * n
    score = np.full(n, np.nan)
    allow = np.zeros(n, bool)
    force = np.zeros(n, bool)
    for t in range(WINDOW, n):
        w = close[t - WINDOW:t + 1]
        f = Features(hurst=hurst_exponent(w), adf_pvalue=adf_pvalue(w),
                     half_life=ou_half_life(w),
                     adx=None if np.isnan(adx_arr[t]) else float(adx_arr[t]))
        d = gate.on_bar(f)
        regime[t], score[t] = d.regime, d.score
        allow[t], force[t] = d.allow_entries, d.force_exit
    return Run(close, truth, regime, score, allow, force)


# ---------------- toy z-score strategy ----------------
def zscore_strategy(close, allow=None, force=None, lb=20, z_in=2.0,
                    time_stop=20, cost=0.02):
    """Long when z<-z_in, short when z>z_in, exit at z crossing 0 or
    time stop. Decisions on bar t close, filled at bar t close (toy).
    cost = per-side cost in price units. Returns per-bar PnL array."""
    n = len(close)
    pnl = np.zeros(n)
    pos, age = 0, 0
    for t in range(lb, n - 1):
        w = close[t - lb + 1:t + 1]
        sd = w.std()
        z = 0.0 if sd == 0 else (close[t] - w.mean()) / sd
        gated = allow is not None
        if pos != 0:
            age += 1
            exit_ = (pos > 0 and z >= 0) or (pos < 0 and z <= 0) \
                or age >= time_stop or (gated and force[t])
            if exit_:
                pnl[t] -= cost
                pos, age = 0, 0
        if pos == 0 and (not gated or allow[t]):
            if z < -z_in:
                pos, age = 1, 0; pnl[t] -= cost
            elif z > z_in:
                pos, age = -1, 0; pnl[t] -= cost
        pnl[t + 1] += pos * (close[t + 1] - close[t])
    return pnl


# ---------------- metrics ----------------
def metrics(r: Run):
    idx = np.arange(WINDOW, len(r.close))
    # explicit identity checks: numpy mangles str-Enum members in comparisons
    mr = np.array([r.regime[i] is Regime.MEAN_REVERTING for i in idx])
    tx = np.array([r.regime[i] is Regime.TRANSITIONAL for i in idx])
    tr = np.array([r.regime[i] is Regime.TRENDING for i in idx])
    truth = r.truth[idx]
    out = {"pct_mr": mr.mean() * 100, "pct_tx": tx.mean() * 100,
           "pct_tr": tr.mean() * 100}
    if truth.any() and (~truth).any():
        out["precision"] = (mr & truth).sum() / max(mr.sum(), 1) * 100
        out["recall"] = (mr & truth).sum() / max(truth.sum(), 1) * 100
    return out


def transition_lags(r: Run):
    """Bars from each truth change to the gate reacting
    (MR->not: leaves MR; not->MR: enters MR)."""
    lags = []
    tt = r.truth
    for t in range(1, len(tt)):
        if tt[t] != tt[t - 1] and t >= WINDOW:
            for k in range(t, len(tt)):
                inmr = r.regime[k] == Regime.MEAN_REVERTING
                if inmr == tt[t]:
                    lags.append((t, "enter MR" if tt[t] else "leave MR", k - t))
                    break
            else:
                lags.append((t, "enter MR" if tt[t] else "leave MR", None))
    return lags


# ---------------- plotting ----------------
def shade_states(ax, regime, start):
    cur, s = regime[start], start
    for t in range(start + 1, len(regime) + 1):
        if t == len(regime) or regime[t] != cur:
            ax.axvspan(s, t, color=STATE_COLOR[cur], alpha=0.16, lw=0)
            if t < len(regime):
                cur, s = regime[t], t


TITLES = {
    "ou_strong": "Strong mean reversion (OU, half-life ≈ 6.6 bars)",
    "ou_weak": "Weak mean reversion (OU, half-life ≈ 34 bars)",
    "random_walk": "Random walk (no edge)",
    "trend": "Trending (random walk + drift)",
    "regime_switch": "Regime switch: OU → trend → OU → random walk",
}


def plot_scenario(name, r: Run, pnl_raw, pnl_gate, m, path):
    switch = name == "regime_switch"
    ratios = ([0.28] if switch else []) + [3, 1.3, 1.6]
    fig, axes = plt.subplots(len(ratios), 1, figsize=(11, 8.6 if switch else 8.2),
                             sharex=True,
                             gridspec_kw={"height_ratios": ratios, "hspace": 0.14})
    x = np.arange(len(r.close))
    if switch:
        ax = axes[0]
        seg = len(r.close) // 4
        for i, (lab, mr) in enumerate([("OU", True), ("trend", False),
                                        ("OU", True), ("random walk", False)]):
            ax.axvspan(i * seg, (i + 1) * seg, color=C_MR if mr else C_TR,
                       alpha=0.55, lw=0)
            ax.axvline(i * seg, color=SURF, lw=2)
            ax.text(i * seg + seg / 2, 0.5, f"truth: {lab}", ha="center",
                    va="center", fontsize=9, color=INK)
        ax.set_yticks([]); ax.grid(False)
        ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", length=0)
        axes = axes[1:]
        top = fig.axes[0]
    else:
        top = axes[0]

    ax = axes[0]
    shade_states(ax, r.regime, WINDOW)
    ax.axvspan(0, WINDOW, color=GRID, alpha=0.5, lw=0)
    ax.text(WINDOW / 2, 0.97, "warm-up", transform=ax.get_xaxis_transform(),
            ha="center", va="top", color=INK2, fontsize=9)
    ax.plot(x, r.close, color=C_PRICE, lw=1.1)
    if switch:
        seg = len(r.close) // 4
        for i in range(1, 4):
            ax.axvline(i * seg, color=INK2, lw=0.8, ls="--")
    ax.set_ylabel("price")
    handles = [Patch(color=C_MR, alpha=0.35, label="gate: mean reverting (entries on)"),
               Patch(color=C_TX, alpha=0.45, label="transitional"),
               Patch(color=C_TR, alpha=0.35, label="trending (flat)")]
    top.set_title(TITLES[name], loc="left", pad=30)
    top.legend(handles=handles, loc="lower left", ncol=3, fontsize=9,
               bbox_to_anchor=(0, 1.0), borderaxespad=0.2, handlelength=1.6)

    ax = axes[1]
    ax.plot(x, r.score, color=C_PRICE, lw=1.0, drawstyle="steps-post")
    for y, lab in [(0.80, "enter 0.80"), (0.55, "exit 0.55"), (0.30, "trend 0.30")]:
        ax.axhline(y, color=INK2, lw=0.7, ls=":")
        ax.text(len(x) * 1.004, y, lab, va="center", fontsize=8, color=INK2)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("gate score")

    ax = axes[2]
    cr, cg = np.cumsum(pnl_raw), np.cumsum(pnl_gate)
    ax.plot(x, cr, color=C_RAW, lw=1.6, label="ungated z-score")
    ax.plot(x, cg, color=C_GATED, lw=1.6, label="gated z-score")
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_ylabel("cum. PnL (pts)")
    ax.set_xlabel("bar")
    ya, yb = cr[-1], cg[-1]
    span = (max(cr.max(), cg.max()) - min(cr.min(), cg.min())) or 1
    if abs(ya - yb) < 0.08 * span:
        mid = (ya + yb) / 2
        ya, yb = (mid + 0.05 * span, mid - 0.05 * span) if ya >= yb \
            else (mid - 0.05 * span, mid + 0.05 * span)
    ax.text(len(x) * 1.004, ya, "ungated", color=INK2, fontsize=8, va="center")
    ax.text(len(x) * 1.004, yb, "gated", color=INK2, fontsize=8, va="center")
    ax.legend(loc="best", fontsize=9)

    stat = (f"time in MR {m['pct_mr']:.0f}%  ·  transitional {m['pct_tx']:.0f}%"
            f"  ·  trending {m['pct_tr']:.0f}%")
    if "precision" in m:
        stat += f"  ·  MR precision {m['precision']:.0f}%  recall {m['recall']:.0f}%"
    stat += "   (this seed)"
    fig.text(0.125, 0.015, stat, color=INK2, fontsize=9)
    axes[2].set_xlim(0, len(x))
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_summary(rows, path):
    names = [r["scenario"] for r in rows]
    labels = {"ou_strong": "OU strong", "ou_weak": "OU weak",
              "random_walk": "Random walk", "trend": "Trend",
              "regime_switch": "Regime switch"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"wspace": 0.3})
    y = np.arange(len(names))
    ax = axes[0]
    left = np.zeros(len(names))
    for key, col, lab in [("pct_mr", C_MR, "mean reverting"),
                          ("pct_tx", C_TX, "transitional"),
                          ("pct_tr", C_TR, "trending")]:
        v = np.array([r[key] for r in rows])
        ax.barh(y, v, left=left, color=col, height=0.6, label=lab,
                edgecolor=SURF, linewidth=2)
        for i, (l, w) in enumerate(zip(left, v)):
            if w >= 9:
                ax.text(l + w / 2, i, f"{w:.0f}%", ha="center", va="center",
                        fontsize=8, color=INK if col == C_TX else "white")
        left += v
    ax.set_yticks(y, [labels[n] for n in names]); ax.invert_yaxis()
    ax.set_xlim(0, 100); ax.set_xlabel("% of bars (mean over seeds)")
    ax.set_title("Where the gate spends its time", loc="left")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.38), ncol=3, fontsize=9)
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    g = np.array([r["pnl_gated"] for r in rows])
    u = np.array([r["pnl_ungated"] for r in rows])
    h = 0.36
    ax.barh(y - h / 2, u, height=h, color=C_RAW, label="ungated",
            edgecolor=SURF, linewidth=2)
    ax.barh(y + h / 2, g, height=h, color=C_GATED, label="gated",
            edgecolor=SURF, linewidth=2)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y, [labels[n] for n in names]); ax.invert_yaxis()
    ax.set_xlabel("final PnL, pts (mean over seeds)")
    ax.set_title("Toy z-score strategy: gated vs ungated", loc="left")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.38), ncol=2, fontsize=9)
    ax.grid(axis="y", visible=False)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------- main ----------------
def main(n_seeds=10):
    os.makedirs(OUT, exist_ok=True)
    scenarios = ["ou_strong", "ou_weak", "random_walk", "trend", "regime_switch"]
    rows = []
    for name in scenarios:
        acc = []
        lags_all = []
        for seed in range(n_seeds):
            rng = np.random.default_rng(1000 + seed)
            close, truth = gen(name, rng)
            r = run_gate(close, truth, rng)
            pr = zscore_strategy(close)
            pg = zscore_strategy(close, r.allow, r.force)
            m = metrics(r)
            m["pnl_ungated"], m["pnl_gated"] = pr.sum(), pg.sum()
            acc.append(m)
            if name == "regime_switch":
                lags_all += transition_lags(r)
            if seed == 0:
                plot_scenario(name, r, pr, pg, m, os.path.join(OUT, f"fig_{name}.png"))
        agg = {"scenario": name}
        for k in acc[0]:
            agg[k] = float(np.mean([a[k] for a in acc]))
        if lags_all:
            for kind in ("leave MR", "enter MR"):
                ls = [l for _, kk, l in lags_all if kk == kind and l is not None]
                agg[f"lag_{kind.replace(' ', '_')}_median"] = \
                    float(np.median(ls)) if ls else float("nan")
        rows.append(agg)
        print(name, {k: (round(v, 1) if isinstance(v, float) else v)
                     for k, v in agg.items()}, flush=True)

    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "scenario", k))
    with open(os.path.join(OUT, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 2) if isinstance(v, float) else v)
                        for k, v in r.items()})
    plot_summary(rows, os.path.join(OUT, "fig_summary.png"))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
