"""
Backtest for the Adaptive Virtual Swing engine (vswing.py).

Execution model
  - Signal on bar i's close -> position changes at bar i+1's OPEN (no lookahead).
  - Flat until the first signal (the init trend state is not traded).
  - mode="long_short": +1 / -1 after the first signal.
    mode="long_only" : +1 on long signals, 0 on short signals.
  - Cost charged per unit of position change (a full reversal pays twice).
  - Daily P&L: old position earns prev close -> open, new position earns open -> close.

Optional overlays (OFF by default)
  sma_len > 0 : trend-regime filter, decided on the signal bar's close.
                A long signal is taken only if close > SMA, a short only if close < SMA;
                a rejected signal means FLAT (it still closes the opposite position).
  vol_size    : position size at entry = min(1, typical_vol / current_vol), where
                current_vol = stdev of the last vol_len close-to-close returns and
                typical_vol = median of current_vol over all bars so far (causal).
                Smaller positions when the market is unusually volatile; never > 1.
"""
import numpy as np
import pandas as pd
from vswing import PriceData, StrategyParams, run


def _sma(x, n):
    out = np.full(len(x), np.nan)
    if n > 0 and len(x) >= n:
        cs = np.cumsum(np.r_[0.0, x])
        out[n - 1:] = (cs[n:] - cs[:-n]) / n
    return out


def _vol_scale(c, n):
    r = np.r_[np.nan, np.diff(np.log(c))]
    v = pd.Series(r).rolling(n).std().to_numpy()
    typ = pd.Series(v).expanding().median().to_numpy()
    s = np.where(np.isnan(v) | (v <= 0), 1.0, np.minimum(1.0, typ / v))
    return np.where(np.isnan(s), 1.0, s)


def backtest(df, sp=StrategyParams(), mode="long_short", cost_bps=5.0,
             sma_len=0, vol_size=False, vol_len=20):
    o, h, l, c = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    r = run(PriceData(o, h, l, c), sp)
    n = len(c)
    cost = cost_bps / 1e4
    sma = _sma(c, sma_len) if sma_len > 0 else None
    scale = _vol_scale(c, vol_len) if vol_size else np.ones(n)

    is_long = ~np.isnan(r.long_signals)
    is_short = ~np.isnan(r.short_signals)
    target = np.full(n, np.nan)                          # desired position after bar i's close
    for i in range(n):
        if is_long[i]:
            ok = sma is None or (not np.isnan(sma[i]) and c[i] > sma[i])
            target[i] = scale[i] if ok else 0.0
        elif is_short[i]:
            ok = mode == "long_short" and (sma is None or (not np.isnan(sma[i]) and c[i] < sma[i]))
            target[i] = -scale[i] if ok else 0.0

    pos = np.zeros(n)
    cur = 0.0
    for i in range(1, n):
        if not np.isnan(target[i - 1]):
            cur = target[i - 1]
        pos[i] = cur

    eq = np.ones(n)
    trades, open_t = [], None
    for i in range(1, n):
        p_old, p_new = pos[i - 1], pos[i]
        g = (1 + p_old * (o[i] / c[i - 1] - 1)) * (1 - abs(p_new - p_old) * cost) * (1 + p_new * (c[i] / o[i] - 1))
        eq[i] = eq[i - 1] * g
        if p_new != p_old:
            if open_t is not None:
                d, ei, ep = open_t
                trades.append((df.index[ei] if "Date" not in df else df.Date.iloc[ei],
                               df.index[i] if "Date" not in df else df.Date.iloc[i],
                               "LONG" if d > 0 else "SHORT", abs(d), ep, o[i],
                               np.sign(d) * (o[i] / ep - 1) - 2 * cost, i - ei))
                open_t = None
            if p_new != 0:
                open_t = (p_new, i, o[i])
    if open_t is not None:
        d, ei, ep = open_t
        trades.append((df.index[ei] if "Date" not in df else df.Date.iloc[ei],
                       df.index[-1] if "Date" not in df else df.Date.iloc[-1],
                       ("LONG" if d > 0 else "SHORT") + " (open)", abs(d), ep, c[-1],
                       np.sign(d) * (c[-1] / ep - 1) - cost, n - 1 - ei))

    tr = pd.DataFrame(trades, columns=["entry", "exit", "side", "size", "entry_px", "exit_px", "ret", "bars"])
    return r, pos, eq, tr


def stats(eq, pos, tr, start, ppy=252):
    e = eq[start:] / eq[start]
    dr = np.diff(e) / e[:-1]
    dd = e / np.maximum.accumulate(e) - 1
    years = max(len(e) - 1, 1) / ppy
    cagr = e[-1] ** (1 / years) - 1 if e[-1] > 0 else -1.0
    wins, losses = tr.ret[tr.ret > 0], tr.ret[tr.ret <= 0]
    return {
        "total_return_%": 100 * (e[-1] - 1),
        "cagr_%": 100 * cagr,
        "max_drawdown_%": 100 * dd.min(),
        "sharpe_ann": np.sqrt(ppy) * dr.mean() / dr.std() if dr.std() > 0 else np.nan,
        "mar": cagr / -dd.min() if dd.min() < 0 else np.nan,
        "trades": len(tr),
        "win_rate_%": 100 * len(wins) / len(tr) if len(tr) else np.nan,
        "avg_win_%": 100 * wins.mean() if len(wins) else np.nan,
        "avg_loss_%": 100 * losses.mean() if len(losses) else np.nan,
        "profit_factor": wins.sum() / -losses.sum() if losses.sum() < 0 else np.inf,
        "avg_bars_held": tr.bars.mean() if len(tr) else np.nan,
        "exposure_%": 100 * np.mean(pos[start:] != 0),
    }
