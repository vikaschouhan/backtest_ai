"""Behavioural checks for regime_gate.RegimeGate."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from regime_gate import (Features, GateConfig, Regime, RegimeGate,  # noqa: E402
                         ou_half_life)

GOOD = Features(hurst=0.30, adf_pvalue=0.001, half_life=6.0, adx=12.0)
BAD = Features(hurst=0.52, adf_pvalue=0.60, half_life=80.0, adx=25.0)
TREND = Features(hurst=0.65, adf_pvalue=0.90, half_life=None, adx=40.0)


def warm(gate, n=30, f=GOOD):
    for _ in range(n):
        d = gate.on_bar(f)
    return d


def test_starts_trending_and_missing_features_block_entries():
    g = RegimeGate()
    d = g.on_bar(Features())
    assert d.regime is Regime.TRENDING
    assert not d.allow_entries
    assert d.score == 0.0


def test_enter_mr_needs_confirm_in_bars():
    g = RegimeGate(GateConfig(confirm_in=3))
    states = [g.on_bar(GOOD).regime for _ in range(30)]
    assert states[-1] is Regime.MEAN_REVERTING
    first_mr = states.index(Regime.MEAN_REVERTING)
    # needs history for the stability / adf-trend checks, then 3 confirmations
    assert first_mr >= 3


def test_single_bad_bar_does_not_exit_mr():
    g = RegimeGate(GateConfig(confirm_out=2))
    assert warm(g).regime is Regime.MEAN_REVERTING
    d = g.on_bar(BAD)
    assert d.regime is Regime.MEAN_REVERTING
    d = g.on_bar(GOOD)
    assert d.regime is Regime.MEAN_REVERTING


def test_sustained_weakness_goes_transitional_without_forced_exit():
    g = RegimeGate(GateConfig(confirm_out=2))
    warm(g)
    weak = Features(hurst=0.30, adf_pvalue=0.60, half_life=80.0, adx=25.0)
    decisions = [g.on_bar(weak) for _ in range(2)]
    assert decisions[-1].regime is Regime.TRANSITIONAL
    assert not decisions[-1].allow_entries
    assert not any(d.force_exit for d in decisions)


def test_hard_break_forces_exit_and_cooldown_blocks_reentry():
    cfg = GateConfig(cooldown_bars=10, confirm_in=3)
    g = RegimeGate(cfg)
    warm(g)
    d = g.on_bar(TREND)
    assert d.regime is Regime.TRENDING and d.force_exit
    states = [g.on_bar(GOOD).regime for _ in range(cfg.cooldown_bars - 1)]
    assert Regime.MEAN_REVERTING not in states


def test_blackout_blocks_entries_but_keeps_state():
    g = RegimeGate()
    warm(g)
    f = Features(**{**GOOD.__dict__, "blackout": True})
    d = g.on_bar(f)
    assert d.regime is Regime.MEAN_REVERTING
    assert not d.allow_entries and not d.force_exit


def test_json_roundtrip_continues_identically():
    rng = np.random.default_rng(0)
    seq = [GOOD if rng.random() < 0.7 else BAD for _ in range(60)]
    a = RegimeGate()
    for f in seq[:30]:
        a.on_bar(f)
    b = RegimeGate.from_json(a.to_json())
    for f in seq[30:]:
        da, db = a.on_bar(f), b.on_bar(f)
        assert da.regime is db.regime and da.score == db.score


def test_ou_half_life_recovers_known_value():
    rng = np.random.default_rng(1)
    theta = 0.10
    x = np.empty(5000); x[0] = 0.0
    for i in range(1, len(x)):
        x[i] = x[i - 1] - theta * x[i - 1] + rng.standard_normal()
    expected = -np.log(2) / np.log(1 - theta)   # ~6.58 bars
    assert abs(ou_half_life(x) - expected) < 1.0


# ---------------- v2 / v3 options ----------------

from regime_gate import v2_config  # noqa: E402


def test_defaults_reproduce_v1():
    c = GateConfig()
    assert c.use_adf_falling and not c.require_core


def test_dropping_adf_falling_removes_the_check():
    g = RegimeGate(GateConfig(use_adf_falling=False))
    d = warm(g)
    assert not any("adf_falling" in r for r in d.reasons)
    assert d.score == 1.0          # 5 of 5


def test_core_fail_caps_score_and_blocks_entry():
    # everything passes except ADF, and Hurst is 0.42 (> core 0.40)
    f = Features(hurst=0.42, adf_pvalue=0.30, half_life=6.0, adx=12.0)
    g = RegimeGate(v2_config())
    decisions = [g.on_bar(f) for _ in range(40)]
    assert all(d.score <= 0.50 for d in decisions)
    assert all("core_fail" in d.reasons for d in decisions)
    assert all(d.regime is not Regime.MEAN_REVERTING for d in decisions)
    assert not any(d.force_exit for d in decisions)   # cap stays above 0.30


def test_core_passes_on_strong_hurst_in_v2_but_not_v3():
    f = Features(hurst=0.30, adf_pvalue=0.30, half_life=6.0, adx=12.0)
    v2 = RegimeGate(v2_config())
    v3 = RegimeGate(v2_config(core_hurst_max=float("-inf")))
    assert warm(v2, f=f).regime is Regime.MEAN_REVERTING
    assert warm(v3, f=f).regime is not Regime.MEAN_REVERTING


def test_core_failure_in_mr_moves_to_transitional_not_flat():
    g = RegimeGate(v2_config(confirm_out=2))
    assert warm(g).regime is Regime.MEAN_REVERTING
    f = Features(hurst=0.44, adf_pvalue=0.30, half_life=6.0, adx=12.0)
    d1, d2 = g.on_bar(f), g.on_bar(f)
    assert d2.regime is Regime.TRANSITIONAL and not d2.force_exit
