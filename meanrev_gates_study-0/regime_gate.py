"""
regime_gate.py — Mean-reversion regime gate (state machine with hysteresis).

Pipeline position:
    feature engine (C kernel) -> RegimeGate.on_bar() -> signal / order manager

The gate does NOT generate trades. It answers, per closed bar:
    * may the strategy open NEW positions?          (allow_entries)
    * must it flatten existing positions right now? (force_exit)
    * and why                                         (reasons)

States
    TRENDING        no entries; existing positions must be flat
    TRANSITIONAL    no new entries; existing positions may run to their exits
    MEAN_REVERTING  entries allowed

Hysteresis
    * entering MR needs score >= enter_score for `confirm_in` consecutive bars
    * leaving MR needs score <  exit_score  for `confirm_out` consecutive bars
      (exit_score < enter_score, so the gate doesn't flicker at the boundary)
    * a HARD BREAK (strong trend evidence) skips confirmation, forces flat,
      and starts a cooldown during which MR cannot be re-entered

Features are expected from your own feature engine; reference Python
implementations are included below for testing / parity checks.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


# --------------------------------------------------------------------------
# Types
# --------------------------------------------------------------------------

class Regime(str, Enum):
    TRENDING = "TRENDING"
    TRANSITIONAL = "TRANSITIONAL"
    MEAN_REVERTING = "MEAN_REVERTING"


@dataclass
class Features:
    """One closed bar's regime features. None = not available (warm-up etc.)."""
    hurst: Optional[float] = None        # rolling Hurst exponent
    adf_pvalue: Optional[float] = None   # rolling ADF p-value
    half_life: Optional[float] = None    # OU half-life, in bars
    adx: Optional[float] = None          # ADX(14) or similar
    blackout: bool = False               # event calendar / VIX spike override


@dataclass
class GateConfig:
    # --- individual test thresholds ---
    hurst_max: float = 0.45
    adf_p_max: float = 0.05
    adx_max: float = 20.0
    half_life_min: float = 2.0           # too fast = noise / microstructure
    half_life_max: float = 30.0          # must fit inside your holding horizon
    half_life_cv_max: float = 0.35       # stability: std/mean of recent HLs
    adf_trend_window: int = 10           # bars per half for "p-value falling"

    # --- composite score hysteresis ---
    enter_score: float = 0.80            # fraction of checks passing to enter MR
    exit_score: float = 0.55             # drop below this to leave MR
    trend_score: float = 0.30            # TRANSITIONAL -> TRENDING below this
    confirm_in: int = 3
    confirm_out: int = 2

    # --- hard break (immediate exit) ---
    hard_hurst: float = 0.58
    hard_adx: float = 30.0
    cooldown_bars: int = 10

    # --- blackout behaviour ---
    blackout_forces_exit: bool = False   # True = flatten into events


@dataclass
class GateDecision:
    regime: Regime
    allow_entries: bool
    force_exit: bool
    score: float
    reasons: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Gate
# --------------------------------------------------------------------------

class RegimeGate:
    def __init__(self, cfg: GateConfig | None = None):
        self.cfg = cfg or GateConfig()
        self.regime = Regime.TRENDING      # start conservative
        self._in_count = 0
        self._out_count = 0
        self._cooldown = 0
        self._hl_hist: deque[float] = deque(maxlen=20)
        self._adf_hist: deque[float] = deque(maxlen=2 * self.cfg.adf_trend_window)

    # ---------------- scoring ----------------

    def _score(self, f: Features) -> tuple[float, list[str], list[str]]:
        """Return (score in [0,1], passed checks, failed checks).
        Missing features count as FAILED — no data, no trade."""
        c = self.cfg
        passed, failed = [], []

        def check(name: str, ok: Optional[bool]):
            (passed if ok else failed).append(name)

        check("hurst", f.hurst is not None and f.hurst < c.hurst_max)
        check("adf", f.adf_pvalue is not None and f.adf_pvalue < c.adf_p_max)
        check("adx", f.adx is not None and f.adx < c.adx_max)
        check("half_life_range", f.half_life is not None
              and c.half_life_min <= f.half_life <= c.half_life_max)

        # half-life stability
        hl_ok = None
        if len(self._hl_hist) >= 5:
            arr = np.asarray(self._hl_hist)
            m = arr.mean()
            hl_ok = m > 0 and (arr.std() / m) < c.half_life_cv_max
        check("half_life_stable", hl_ok)

        # ADF p-value trending down (or already deep)
        adf_ok = None
        w = c.adf_trend_window
        if len(self._adf_hist) >= 2 * w:
            arr = np.asarray(self._adf_hist)
            adf_ok = arr[w:].mean() <= arr[:w].mean() or arr[w:].mean() < 0.01
        check("adf_falling", adf_ok)

        score = len(passed) / (len(passed) + len(failed))
        return score, passed, failed

    def _hard_break(self, f: Features) -> list[str]:
        c, r = self.cfg, []
        if f.hurst is not None and f.hurst > c.hard_hurst:
            r.append(f"hard_break:hurst={f.hurst:.2f}")
        if f.adx is not None and f.adx > c.hard_adx:
            r.append(f"hard_break:adx={f.adx:.1f}")
        return r

    # ---------------- per-bar update ----------------

    def on_bar(self, f: Features) -> GateDecision:
        """Call once per CLOSED bar, in order."""
        c = self.cfg
        if f.half_life is not None and math.isfinite(f.half_life):
            self._hl_hist.append(f.half_life)
        if f.adf_pvalue is not None:
            self._adf_hist.append(f.adf_pvalue)

        score, passed, failed = self._score(f)
        reasons = [f"score={score:.2f}", f"fail={','.join(failed) or '-'}"]
        force_exit = False

        if self._cooldown > 0:
            self._cooldown -= 1

        hb = self._hard_break(f)
        prev = self.regime

        if hb:
            # immediate, no confirmation
            self.regime = Regime.TRENDING
            self._cooldown = c.cooldown_bars
            self._in_count = self._out_count = 0
            force_exit = True
            reasons += hb

        elif self.regime == Regime.MEAN_REVERTING:
            self._out_count = self._out_count + 1 if score < c.exit_score else 0
            if self._out_count >= c.confirm_out:
                self.regime = Regime.TRANSITIONAL
                self._out_count = 0

        else:  # TRENDING or TRANSITIONAL
            if score >= c.enter_score and self._cooldown == 0:
                self._in_count += 1
            else:
                self._in_count = 0
            if self._in_count >= c.confirm_in:
                self.regime = Regime.MEAN_REVERTING
                self._in_count = 0
            elif self.regime == Regime.TRANSITIONAL and score < c.trend_score:
                self.regime = Regime.TRENDING
                force_exit = True
            elif self.regime == Regime.TRENDING and score >= c.exit_score \
                    and self._cooldown == 0:
                self.regime = Regime.TRANSITIONAL

        if self.regime != prev:
            reasons.append(f"transition:{prev.value}->{self.regime.value}")
        if self._cooldown > 0:
            reasons.append(f"cooldown={self._cooldown}")

        allow = self.regime == Regime.MEAN_REVERTING
        if f.blackout:
            allow = False
            reasons.append("blackout")
            if c.blackout_forces_exit:
                force_exit = True

        return GateDecision(self.regime, allow, force_exit, score, reasons)

    # ---------------- persistence (survive restarts) ----------------

    def to_json(self) -> str:
        return json.dumps({
            "cfg": asdict(self.cfg),
            "regime": self.regime.value,
            "in": self._in_count, "out": self._out_count,
            "cooldown": self._cooldown,
            "hl": list(self._hl_hist), "adf": list(self._adf_hist),
        })

    @classmethod
    def from_json(cls, s: str) -> "RegimeGate":
        d = json.loads(s)
        g = cls(GateConfig(**d["cfg"]))
        g.regime = Regime(d["regime"])
        g._in_count, g._out_count = d["in"], d["out"]
        g._cooldown = d["cooldown"]
        g._hl_hist.extend(d["hl"])
        g._adf_hist.extend(d["adf"])
        return g


# --------------------------------------------------------------------------
# Reference feature implementations (for testing / parity with C kernel)
# --------------------------------------------------------------------------

def hurst_exponent(x: np.ndarray, max_lag: int = 20) -> float:
    """Variance-of-lagged-differences estimator. ~0.5 random walk,
    <0.5 mean-reverting, >0.5 trending."""
    lags = np.arange(2, max_lag)
    tau = np.array([np.std(x[lag:] - x[:-lag]) for lag in lags])
    tau = np.where(tau <= 0, 1e-12, tau)
    return float(np.polyfit(np.log(lags), np.log(tau), 1)[0])


def ou_half_life(x: np.ndarray) -> float:
    """AR(1) on levels: dx_t = a + b*x_{t-1}; HL = -ln2 / ln(1+b)."""
    dx = np.diff(x)
    xl = x[:-1]
    A = np.column_stack([np.ones_like(xl), xl])
    b = np.linalg.lstsq(A, dx, rcond=None)[0][1]
    if b >= 0 or b <= -1:
        return float("inf")
    return float(-math.log(2) / math.log(1 + b))


def adf_pvalue(x: np.ndarray) -> Optional[float]:
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        return None
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return float(adfuller(x, maxlag=1, autolag=None)[1])


def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray,
        n: int = 14) -> np.ndarray:
    """Wilder's ADX. Returns array aligned to inputs (NaN during warm-up)."""
    up = high[1:] - high[:-1]
    dn = low[:-1] - low[1:]
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.maximum.reduce([high[1:] - low[1:],
                            np.abs(high[1:] - close[:-1]),
                            np.abs(low[1:] - close[:-1])])

    def wilder(a):
        out = np.full_like(a, np.nan)
        out[n - 1] = a[:n].sum()
        for i in range(n, len(a)):
            out[i] = out[i - 1] - out[i - 1] / n + a[i]
        return out

    atr = wilder(tr)
    pdi = 100 * wilder(plus_dm) / atr
    mdi = 100 * wilder(minus_dm) / atr
    dx = 100 * np.abs(pdi - mdi) / np.where(pdi + mdi == 0, np.nan, pdi + mdi)
    res = np.full(len(close), np.nan)
    start = 2 * n - 1
    if len(dx) > start:
        res[start] = np.nanmean(dx[n - 1:start])
        for i in range(start + 1, len(dx)):
            res[i] = (res[i - 1] * (n - 1) + dx[i - 1]) / n
    return np.concatenate([[np.nan], res[:-1]])


# --------------------------------------------------------------------------
# Demo: synthetic series = OU (mean-reverting) -> trend -> OU
# --------------------------------------------------------------------------

def _demo():
    rng = np.random.default_rng(7)
    n1, n2, n3 = 400, 250, 400
    x, theta, sigma = [100.0], 0.12, 0.6
    for _ in range(n1):
        x.append(x[-1] + theta * (100 - x[-1]) + sigma * rng.standard_normal())
    for _ in range(n2):
        x.append(x[-1] + 0.25 + sigma * rng.standard_normal())
    m = x[-1]
    for _ in range(n3):
        x.append(x[-1] + theta * (m - x[-1]) + sigma * rng.standard_normal())
    close = np.array(x)
    high = close + np.abs(rng.normal(0, 0.3, len(close)))
    low = close - np.abs(rng.normal(0, 0.3, len(close)))
    adx_arr = adx(high, low, close)

    window = 120
    gate = RegimeGate()
    last = None
    hard_breaks = 0
    print(f"{'bar':>5} {'phase':<6} {'regime':<15} {'entry':<5} {'exit':<5} detail")
    for t in range(window, len(close)):
        w = close[t - window:t + 1]
        f = Features(
            hurst=hurst_exponent(w),
            adf_pvalue=adf_pvalue(w),
            half_life=ou_half_life(w),
            adx=None if np.isnan(adx_arr[t]) else float(adx_arr[t]),
        )
        d = gate.on_bar(f)
        hard_breaks += any(r.startswith("hard_break") for r in d.reasons)
        phase = "OU" if t <= n1 else ("TREND" if t <= n1 + n2 else "OU")
        if d.regime != last:
            print(f"{t:>5} {phase:<6} {d.regime.value:<15} "
                  f"{str(d.allow_entries):<5} {str(d.force_exit):<5} "
                  f"{' '.join(d.reasons)}")
            last = d.regime

    print(f"\nbars with hard-break flatten signal: {hard_breaks}")

    # restart round-trip
    g2 = RegimeGate.from_json(gate.to_json())
    assert g2.regime == gate.regime
    print("\nstate persisted & restored OK:", g2.regime.value)


if __name__ == "__main__":
    _demo()
