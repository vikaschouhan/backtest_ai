# meanrev_gates_study-0 — Mean-reversion regime gate

A causal regime gate that decides, bar by bar, whether a mean-reversion strategy may
open trades, plus a test harness that runs it on synthetic price processes. Full spec,
session history and findings: [`CONTEXT.txt`](CONTEXT.txt).

## The gate in one paragraph

Each closed bar is scored by six pass/fail checks (Hurst < 0.45, ADF p < 0.05, ADX < 20,
OU half-life in 2–30 bars, half-life stable, ADF p-value falling); the score is the
fraction passing, and a missing feature counts as a fail. A three-state machine with
hysteresis turns the score into a decision: **TRENDING** (no entries, stay flat) →
**TRANSITIONAL** (no new entries, open trades run to their own exits) →
**MEAN_REVERTING** (entries allowed). Entering MR needs score ≥ 0.80 for 3 bars; leaving
needs < 0.55 for 2 bars; TRANSITIONAL drops to TRENDING (forced flat) below 0.30. A
**hard break** (ADX > 30 or Hurst > 0.58) skips confirmation, forces flat and blocks
re-entry for 10 bars. A `blackout` flag (event days, VIX spikes) blocks entries in any
state.

```python
d = gate.on_bar(Features(hurst, adf_pvalue, half_life, adx, blackout))
if d.force_exit:
    close_all()
elif d.allow_entries and abs(z) > 2:
    enter()
```

## Files

| File | What it is |
|---|---|
| `regime_gate.py` | `RegimeGate`, `GateConfig`, `Features`, `GateDecision`; JSON state save/restore; reference Hurst, OU half-life, ADF, ADX implementations (for parity with a C feature kernel); `python regime_gate.py` runs a small demo |
| `gate_tests.py` | Synthetic scenarios (strong OU, weak OU, random walk, trend, regime switch), a toy z-score strategy run gated vs ungated, figures and `summary.csv` |
| `tests/test_regime_gate.py` | Hysteresis, hard break + cooldown, blackout, JSON round-trip, half-life estimator checks |
| `results/` | Figures and CSV produced by `gate_tests.py` |

## Run

```bash
pip install -r requirements.txt
python gate_tests.py 10        # 10 seeds per scenario, ~2 min; writes results/
python -m pytest -q tests
```

## Results (10 seeds × 1,500 bars, 120-bar feature window)

| Scenario | Time in MR | Ungated PnL | Gated PnL |
|---|---|---|---|
| Strong OU (half-life ≈ 6.6) | 86% | +64 | +53 |
| Weak OU (half-life ≈ 34) | 41% | +18 | +8 |
| Random walk | 28% | −6 | 0 |
| Trend | 6% | −74 | −7 |
| Regime switch | 44% | −5 | +20 |

Regime switch: MR precision 72%, recall 69%; median lag 16 bars to leave MR after a
regime ends, 60 bars to re-enter after mean reversion resumes.

![Summary](results/fig_summary.png)
![Regime switch](results/fig_regime_switch.png)

## Findings

- The gate does its main job: trend losses fall from −74 to −7, and the regime-switch
  run turns from a loss into a profit.
- It is too permissive on a random walk (on 28% of the time). Per-check pass rates on a
  random walk: Hurst 48%, **ADF 2%**, ADX 58%, half-life range 64%, half-life stable 70%,
  **ADF falling 52%** (vs 56% on strong OU — it carries no information).
- Re-entry lag is set by the window length: the window keeps seeing the old regime.

## Next steps (not yet done)

1. Drop the `adf_falling` check.
2. Require ADF pass **or** Hurst < 0.40 as a mandatory condition rather than one vote of six.
3. Compute ADF on a longer window (~250 bars) than the other features.
4. Re-run `gate_tests.py` and compare; then calibrate thresholds on Nifty/BankNifty using
   gated strategy PnL, not classification accuracy.
