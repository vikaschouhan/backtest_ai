import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from vswing import *

rng = np.random.default_rng(7)
n = 3000
c = 100*np.exp(np.cumsum(rng.normal(0, 0.012, n)))
o = np.r_[c[0], c[:-1]]
h = np.maximum(o, c)*(1+np.abs(rng.normal(0, .004, n)))
l = np.minimum(o, c)*(1-np.abs(rng.normal(0, .004, n)))
sp = StrategyParams(23.0, 20)
r = run(PriceData(o, h, l, c), sp)

# 1. causality
bad = 0
for t in rng.choice(np.arange(25, n), 300, replace=False):
    r2 = run(PriceData(o[:t+1], h[:t+1], l[:t+1], c[:t+1]), sp)
    for a, b in zip(r, r2):
        if not np.array_equal(a[:t+1], b, equal_nan=True): bad += 1
print("causality mismatches:", bad)

# 2. on each flip, previous extreme becomes the new swing
flips = np.where(~np.isnan(r.long_signals) | ~np.isnan(r.short_signals))[0]
ok = True
for i in flips:
    if not np.isnan(r.short_signals[i]): ok &= r.swing_high[i] == r.ath[i]
    else:                                ok &= r.swing_low[i]  == r.atl[i]
print("flips:", len(flips), " swing promotion ok:", bool(ok))
kinds = np.where(~np.isnan(r.long_signals[flips]), 1, -1)
print("signals alternate:", bool(np.all(kinds[1:] != kinds[:-1])))

# 3. stops always defined after init and within the leg range
s = slice(20, None)
print("stop every bar:", bool(np.all(~np.isnan(r.v_sl[s]) ^ ~np.isnan(r.v_sh[s]))))
up = r.trend == 1; dn = r.trend == -1
u = ~np.isnan(r.v_sl); d = ~np.isnan(r.v_sh)
print("swing_low < v_sl <= ath:", bool(np.all((r.v_sl[u] <= r.ath[u]) & (r.v_sl[u] >= r.swing_low[u]))))
print("atl <= v_sh < swing_high:", bool(np.all((r.v_sh[d] >= r.atl[d]) & (r.v_sh[d] <= r.swing_high[d]))))
print("trend split up/down:", up.sum(), dn.sum())
# 4. ratchet: vsl never falls, vsh never rises within a leg
leg_id = np.cumsum(np.r_[0, np.diff(r.trend) != 0])
ok_sl = ok_sh = True
for i in range(21, n):
    if leg_id[i] == leg_id[i-1]:
        if r.trend[i] == 1: ok_sl &= np.isnan(r.v_sl[i-1]) or r.v_sl[i] >= r.v_sl[i-1]
        if r.trend[i] == -1: ok_sh &= np.isnan(r.v_sh[i-1]) or r.v_sh[i] <= r.v_sh[i-1]
print("vsl ratchets up:", bool(ok_sl), " vsh ratchets down:", bool(ok_sh))
print("vsl <= rev_dn <= ath:", bool(np.all((r.v_sl[u] <= r.rev_dn[u]) & (r.rev_dn[u] <= r.ath[u]))))
print("atl <= rev_up <= vsh:", bool(np.all((r.atl[d] <= r.rev_up[d]) & (r.rev_up[d] <= r.v_sh[d]))))
fl = ~np.isnan(r.short_signals); print("shorts below rev_dn:", bool(np.all(r.short_signals[fl] < r.rev_dn[fl])))
fl = ~np.isnan(r.long_signals);  print("longs above rev_up:", bool(np.all(r.long_signals[fl] > r.rev_up[fl])))
