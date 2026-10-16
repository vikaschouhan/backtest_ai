"""
Adaptive Virtual Swing Crossover - core engine (Numba), no ZigZag.

Arrays are chronological (index 0 = oldest, -1 = latest).
Everything is causal: values at bar i use only bars 0..i.

threshold is given in PERCENT (e.g. 23.0) and divided by 100 internally.

UP trend (+1)
    ath  = highest high since the trend started (updated with this bar's high first)
    vsl  starts at the previous swing low, then on each bar:
         r = (ath - low) / (ath - vsl)
         r >  thr : vsl unchanged
         r <= thr : vsl = ath - (ath - low) / thr
    rev_dn = vsl + (ath - vsl) * thr
    close < rev_dn  ->  DOWN:  swing_high := ath,  atl := lowest low since the ath bar

DOWN trend (-1)   (mirror)
    atl  = lowest low since the trend started
    vsh  starts at the previous swing high, then on each bar:
         r = (high - atl) / (vsh - atl)
         r >  thr : vsh unchanged
         r <= thr : vsh = atl + (high - atl) / thr
    rev_up = vsh - (vsh - atl) * thr
    close > rev_up  ->  UP:    swing_low := atl,   ath := highest high since the atl bar

Note: when r <= thr the new level is always at or beyond the old one, so vsl only
ratchets up and vsh only ratchets down within a leg. For thr <= 50% the reversal
level after an update is still at or beyond that bar's low/high, so the bar that
moves the level cannot flip the trend on its own close.

Optional enhancements (all OFF by default -> identical to the spec above)
    rev_threshold      : separate % for the reversal line (default = threshold).
                         Decouples "how fast the floor tightens" from "how early we flip".
    min_cushion_atr k  : a floor update may never bring vsl closer than k*ATR to ath
                         (vsh: k*ATR from atl). Stops the floor hugging price in quiet bars.
    confirm_atr m      : flip needs close < rev_dn - m*ATR (mirror for up). Noise buffer.
    atr_len            : Wilder ATR length (causal: uses bars 0..i).

Initialisation
    From the first `init_bars` bars: swing_high = max(high), swing_low = min(low).
    High more recent -> start UP (ath = swing_high, vsl = swing_low),
    else start DOWN (atl = swing_low, vsh = swing_high).
"""
import numpy as np
from typing import NamedTuple
from numba import njit


class PriceData(NamedTuple):
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray


class StrategyParams(NamedTuple):
    threshold: float = 23.0        # percent
    init_bars: int = 20
    rev_threshold: float = -1.0    # percent; <0 -> same as threshold
    min_cushion_atr: float = 0.0   # 0 = off
    confirm_atr: float = 0.0       # 0 = off
    atr_len: int = 14


@njit(cache=True)
def wilder_atr(high, low, close, length):
    n = len(high)
    atr = np.zeros(n)
    if n == 0:
        return atr
    tr = high[0] - low[0]
    atr[0] = tr
    for i in range(1, n):
        tr = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        k = min(i + 1, length)                  # expanding mean until `length` bars exist
        atr[i] = atr[i - 1] + (tr - atr[i - 1]) / k
    return atr


class SwingResult(NamedTuple):
    trend: np.ndarray          # +1 up / long, -1 down / short, 0 before init
    swing_high: np.ndarray     # current reference swing high
    swing_low: np.ndarray      # current reference swing low
    ath: np.ndarray            # running high of the UP leg (NaN in DOWN)
    atl: np.ndarray            # running low of the DOWN leg (NaN in UP)
    v_sl: np.ndarray           # virtual swing low, active in UP (NaN in DOWN)
    v_sh: np.ndarray           # virtual swing high, active in DOWN (NaN in UP)
    rev_dn: np.ndarray         # UP-trend reversal level  vsl + (ath - vsl) * thr
    rev_up: np.ndarray         # DOWN-trend reversal level vsh - (vsh - atl) * thr
    long_signals: np.ndarray   # close of the bar that flipped to UP, NaN elsewhere
    short_signals: np.ndarray  # close of the bar that flipped to DOWN, NaN elsewhere


@njit(cache=True)
def virtual_swing_core(high, low, close, atr, threshold_pct, init_bars,
                       rev_pct, min_cush_atr, confirm_atr):
    n = len(high)
    trend = np.zeros(n)
    sh_o = np.full(n, np.nan)
    sl_o = np.full(n, np.nan)
    ath_o = np.full(n, np.nan)
    atl_o = np.full(n, np.nan)
    vsl_o = np.full(n, np.nan)
    vsh_o = np.full(n, np.nan)
    rdn_o = np.full(n, np.nan)
    rup_o = np.full(n, np.nan)
    long_sig = np.full(n, np.nan)
    short_sig = np.full(n, np.nan)

    thr = threshold_pct / 100.0
    rthr = (rev_pct if rev_pct >= 0.0 else threshold_pct) / 100.0
    if thr <= 0.0 or init_bars < 2 or n <= init_bars:
        return trend, sh_o, sl_o, ath_o, atl_o, vsl_o, vsh_o, rdn_o, rup_o, long_sig, short_sig

    # ---- initial estimate ----
    hi_i = 0
    lo_i = 0
    for k in range(1, init_bars):
        if high[k] >= high[hi_i]: hi_i = k
        if low[k] <= low[lo_i]: lo_i = k
    swing_high = high[hi_i]
    swing_low = low[lo_i]

    if hi_i > lo_i:
        state = 1.0
        ext = swing_high                # ath
        vlev = swing_low                # vsl
        opp = np.inf                    # lowest low since the ath bar
        for k in range(hi_i + 1, init_bars):
            if low[k] < opp: opp = low[k]
    else:
        state = -1.0
        ext = swing_low                 # atl
        vlev = swing_high               # vsh
        opp = -np.inf                   # highest high since the atl bar
        for k in range(lo_i + 1, init_bars):
            if high[k] > opp: opp = high[k]

    # ---- bar loop ----
    for i in range(init_bars, n):
        h, l, c = high[i], low[i], close[i]

        if state == 1.0:
            if h > ext:
                ext = h
                opp = l
            elif l < opp:
                opp = l
            den = ext - vlev
            if den > 0.0 and (ext - l) / den <= thr:
                cand = ext - (ext - l) / thr
                if min_cush_atr > 0.0:
                    cand = min(cand, ext - min_cush_atr * atr[i])
                if cand > vlev:
                    vlev = cand
            ath_o[i] = ext
            rev = vlev + (ext - vlev) * rthr
            vsl_o[i] = vlev
            rdn_o[i] = rev
            if c < rev - confirm_atr * atr[i]:
                swing_high = ext            # ath -> new swing high
                state = -1.0
                ext = opp                   # atl since the ath bar
                vlev = swing_high           # vsh starts at the new swing high
                opp = -np.inf
                short_sig[i] = c
        else:
            if l < ext:
                ext = l
                opp = h
            elif h > opp:
                opp = h
            den = vlev - ext
            if den > 0.0 and (h - ext) / den <= thr:
                cand = ext + (h - ext) / thr
                if min_cush_atr > 0.0:
                    cand = max(cand, ext + min_cush_atr * atr[i])
                if cand < vlev:
                    vlev = cand
            atl_o[i] = ext
            rev = vlev - (vlev - ext) * rthr
            vsh_o[i] = vlev
            rup_o[i] = rev
            if c > rev + confirm_atr * atr[i]:
                swing_low = ext             # atl -> new swing low
                state = 1.0
                ext = opp                   # ath since the atl bar
                vlev = swing_low            # vsl starts at the new swing low
                opp = np.inf
                long_sig[i] = c

        trend[i] = state
        sh_o[i] = swing_high
        sl_o[i] = swing_low

    return trend, sh_o, sl_o, ath_o, atl_o, vsl_o, vsh_o, rdn_o, rup_o, long_sig, short_sig


def run(data: PriceData, sp: StrategyParams = StrategyParams()) -> SwingResult:
    h = np.ascontiguousarray(data.high, dtype=np.float64)
    l = np.ascontiguousarray(data.low, dtype=np.float64)
    c = np.ascontiguousarray(data.close, dtype=np.float64)
    atr = wilder_atr(h, l, c, int(sp.atr_len))
    return SwingResult(*virtual_swing_core(h, l, c, atr, float(sp.threshold), int(sp.init_bars),
                                           float(sp.rev_threshold), float(sp.min_cushion_atr),
                                           float(sp.confirm_atr)))
