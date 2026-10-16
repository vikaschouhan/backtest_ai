"""
gate_ablation.py — compare v1 gate vs the proposed fixes on the same seeds.

Variants (each builds on the previous unless noted)
    v1            baseline: 6 checks, ADF on 120-bar window
    no_adf_fall   drop the adf_falling check
    +core         require (ADF pass OR Hurst < 0.40); failing caps score at 0.50
    +adf250       ADF p-value on a 250-bar window  (= full v2)
    no_adf_fall+adf250   long ADF window without the core rule (isolates fix 3)
    v3 core=ADF250 only  v2 but the mandatory test is ADF(250) alone (no Hurst)

Features are computed once per (scenario, seed) and shared by all variants, so
differences come from the gate alone. Metrics are measured from bar WINDOW
(120) for every variant; the 250-bar ADF has no value before bar 250, so its
longer warm-up is counted as a cost.

Outputs (results/v2/): ablation.csv, fig_ablation.png, fig_<scenario>.png for v3.
Usage: python gate_ablation.py [n_seeds=10]
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from regime_gate import (RegimeGate, GateConfig, Features, v2_config, v3_config,
                         hurst_exponent, ou_half_life, adf_pvalue, adx)
import gate_tests as gt
from gate_tests import (Run, gen, zscore_strategy, metrics, transition_lags,
                        plot_scenario, WINDOW, SURF, INK, INK2, GRID)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "v2")
ADF_LONG = 250

SCENARIOS = ["ou_strong", "ou_weak", "random_walk", "trend", "regime_switch"]
VARIANTS = {
    "v1":                 (GateConfig(), "adf120"),
    "no_adf_fall":        (GateConfig(use_adf_falling=False), "adf120"),
    "+core":              (v2_config(), "adf120"),
    "+adf250 (v2)":       (v2_config(), "adf250"),
    "no_adf_fall+adf250": (GateConfig(use_adf_falling=False), "adf250"),
    # v3: core = ADF(250) only; 120-bar Hurst reads ~0.45 on a random walk and
    # is < 0.40 a third of the time, so it can't serve as the mandatory test
    "v3 core=ADF250 only": (v3_config(), "adf250"),
}
# fixed categorical order (reference palette slots 1..5); v2 gets slot 1
VCOLOR = {"v1": "#b4b2a9", "no_adf_fall": "#eda100", "+core": "#1baf7a",
          "+adf250 (v2)": "#2a78d6", "no_adf_fall+adf250": "#e87ba4",
          "v3 core=ADF250 only": "#6250d6"}


def features(close, rng):
    hi = close + np.abs(rng.normal(0, 0.3, len(close)))
    lo = close - np.abs(rng.normal(0, 0.3, len(close)))
    a = adx(hi, lo, close)
    n = len(close)
    F = {k: [None] * n for k in ("hurst", "hl", "adx", "adf120", "adf250")}
    for t in range(WINDOW, n):
        w = close[t - WINDOW:t + 1]
        F["hurst"][t] = hurst_exponent(w)
        F["hl"][t] = ou_half_life(w)
        F["adx"][t] = None if np.isnan(a[t]) else float(a[t])
        F["adf120"][t] = adf_pvalue(w)
        if t >= ADF_LONG:
            F["adf250"][t] = adf_pvalue(close[t - ADF_LONG:t + 1])
    return F


def run_variant(close, truth, F, cfg, adf_key):
    n = len(close)
    gate = RegimeGate(cfg)
    regime = [gt.Regime.TRENDING] * n
    score = np.full(n, np.nan)
    allow = np.zeros(n, bool)
    force = np.zeros(n, bool)
    for t in range(WINDOW, n):
        d = gate.on_bar(Features(hurst=F["hurst"][t], adf_pvalue=F[adf_key][t],
                                 half_life=F["hl"][t], adx=F["adx"][t]))
        regime[t], score[t] = d.regime, d.score
        allow[t], force[t] = d.allow_entries, d.force_exit
    return Run(close, truth, regime, score, allow, force)


def main(n_seeds=10):
    os.makedirs(OUT, exist_ok=True)
    acc = {(s, v): [] for s in SCENARIOS for v in VARIANTS}
    lags = {v: {"leave MR": [], "enter MR": []} for v in VARIANTS}
    for s in SCENARIOS:
        for seed in range(n_seeds):
            rng = np.random.default_rng(1000 + seed)   # same seeds as v1 study
            close, truth = gen(s, rng)
            F = features(close, rng)
            pr = zscore_strategy(close)
            for v, (cfg, key) in VARIANTS.items():
                r = run_variant(close, truth, F, cfg, key)
                pg = zscore_strategy(close, r.allow, r.force)
                m = metrics(r)
                m["pnl_ungated"], m["pnl_gated"] = pr.sum(), pg.sum()
                acc[(s, v)].append(m)
                if s == "regime_switch":
                    for _, kind, l in transition_lags(r):
                        if l is not None:
                            lags[v][kind].append(l)
                if seed == 0 and v == "v3 core=ADF250 only":
                    plot_scenario(s, r, pr, pg, m, os.path.join(OUT, f"fig_{s}.png"))
        print(f"{s} done", flush=True)

    rows = []
    for s in SCENARIOS:
        for v in VARIANTS:
            ms = acc[(s, v)]
            row = {"scenario": s, "variant": v}
            for k in ms[0]:
                row[k] = round(float(np.mean([m[k] for m in ms])), 2)
            row["pnl_gated_sd"] = round(float(np.std([m["pnl_gated"] for m in ms])), 2)
            if s == "regime_switch":
                for kind in ("leave MR", "enter MR"):
                    L = lags[v][kind]
                    row[f"lag_{kind.replace(' ', '_')}_median"] = \
                        float(np.median(L)) if L else float("nan")
            rows.append(row)

    keys = ["scenario", "variant", "pct_mr", "pct_tx", "pct_tr", "pnl_ungated",
            "pnl_gated", "pnl_gated_sd", "precision", "recall",
            "lag_leave_MR_median", "lag_enter_MR_median"]
    with open(os.path.join(OUT, "ablation.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    for r in rows:
        print(f"{r['scenario']:14s} {r['variant']:20s} MR {r['pct_mr']:5.1f}%  "
              f"gated {r['pnl_gated']:+7.1f} (sd {r['pnl_gated_sd']:.1f})  "
              f"ungated {r['pnl_ungated']:+7.1f}"
              + (f"  prec {r['precision']:.0f} rec {r['recall']:.0f} "
                 f"lag out {r['lag_leave_MR_median']:.0f} in {r['lag_enter_MR_median']:.0f}"
                 if r['scenario'] == 'regime_switch' else ""))
    plot_ablation(rows, os.path.join(OUT, "fig_ablation.png"))


def plot_ablation(rows, path):
    labels = {"ou_strong": "OU strong", "ou_weak": "OU weak",
              "random_walk": "Random walk", "trend": "Trend",
              "regime_switch": "Regime switch"}
    vs = list(VARIANTS)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"wspace": 0.28})
    y = np.arange(len(SCENARIOS))
    h = 0.8 / len(vs)
    get = {(r["scenario"], r["variant"]): r for r in rows}
    for ax, key, title, xl in [
            (axes[0], "pct_mr", "Time in mean-reverting state",
             "% of bars (mean of 10 seeds)"),
            (axes[1], "pnl_gated", "Gated z-score strategy PnL",
             "final PnL, pts (mean of 10 seeds)")]:
        for i, v in enumerate(vs):
            vals = [get[(s, v)][key] for s in SCENARIOS]
            ax.barh(y - 0.4 + h * (i + 0.5), vals, height=h, color=VCOLOR[v],
                    label=v, edgecolor=SURF, linewidth=1.5)
        if key == "pnl_gated":
            ung = [get[(s, "v1")]["pnl_ungated"] for s in SCENARIOS]
            for yi, u in zip(y, ung):
                ax.plot([u, u], [yi - 0.45, yi + 0.45], color=INK, lw=1.2, ls="--")
            ax.plot([], [], color=INK, lw=1.2, ls="--", label="ungated")
            ax.axvline(0, color=INK2, lw=0.8)
        ax.set_yticks(y, [labels[s] for s in SCENARIOS]); ax.invert_yaxis()
        ax.set_title(title, loc="left"); ax.set_xlabel(xl)
        ax.grid(axis="y", visible=False)
    axes[1].set_xlim(left=min(-10, axes[1].get_xlim()[0]))
    h_, l_ = axes[1].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.09))
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
