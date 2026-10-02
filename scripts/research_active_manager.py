"""
An ACTIVE portfolio manager: scan every day, add what's working, drop what isn't.

This tests the thing the monthly book cannot do -- react between rebalances, and let
each position's holding period be decided by the stock rather than the calendar.
Prompted by: a beaten-down name (SUNTV) popping +4.59% on a day 82% of stocks fell.
The earlier bounce test held 21 days and lost badly; a short-term trader would have
taken the pop and left. That is a different trade and deserves its own test.

Mechanics (daily loop, SLOTS positions):
  entry   a stock entering the top ENTRY_RANK is bought into a free slot
  exit    sold when it falls out of top EXIT_RANK (hysteresis stops churn),
          or hits MAX_HOLD days, or breaks a TRAIL trailing stop
  timing  signal uses closes up to day t; the trade executes at t+1's close
          (no look-ahead -- you cannot act on a close you have not seen)
  costs   RT/2 charged on EVERY entry and EVERY exit. This is where active
          strategies usually die, so it is charged in full, per trade.

Reports turnover and average holding period alongside return, because a strategy
that only wins before costs is not a strategy.

Run:  python scripts/research_active_manager.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.research_tranching import build_panel, PRICE_MIN, VOL_FLOOR, RT

BASE = Path(__file__).parent.parent
FROM_YEAR = 2010; IS_END = 2018
SLOTS = 15
ONE_WAY = RT/2


def build_scores(C, turn_sm, vol, malong, kind):
    """Score panel (days x stocks). NaN = not eligible that day."""
    m1 = C/C.shift(21)-1; m3 = C/C.shift(63)-1
    m6 = C/C.shift(126)-1; m12 = C/C.shift(252)-1
    st = C/C.shift(10)-1
    tr = turn_sm.rank(pct=True, axis=1)
    elig = (C >= PRICE_MIN) & (vol >= VOL_FLOOR) & (tr > 0.60) & (tr <= 0.90)

    if kind == "st_mom":                       # short-term momentum, risk adjusted
        s = st/vol
    elif kind == "st_mom_trend":               # same, but only in long-term uptrends
        s = (st/vol).where(C > malong)
    elif kind == "bounce_st":                  # SUNTV profile, ranked by short-term pop
        s = (st/vol).where((m6 < 0) & (m12 < 0))
    elif kind == "mom_daily":                  # the DEPLOYED signal, scanned daily
        s = sum((x/vol).rank(pct=True, axis=1) for x in (m3, m6, m12))/3
        s = s.where(C > malong)
    else:
        raise ValueError(kind)
    return s.where(elig)


def simulate(C, scores, start, entry_rank, exit_rank, max_hold, trail):
    """Daily slot-based active book. Returns (nav, stats)."""
    dates = C.index[start:]
    S = scores.loc[dates].values
    P = C.loc[dates].values
    cols = np.array(C.columns)
    nav = np.ones(len(dates))
    held = {}                                  # col_idx -> dict(entry_px, days, peak)
    trades = 0; hold_lens = []
    for t in range(1, len(dates)):
        px_prev, px = P[t-1], P[t]
        # --- mark the book on today's move (positions were set yesterday)
        if held:
            rets = []
            for j in list(held):
                if np.isfinite(px[j]) and np.isfinite(px_prev[j]) and px_prev[j] > 0:
                    rets.append(px[j]/px_prev[j]-1)
                    held[j]["peak"] = max(held[j]["peak"], px[j])
                    held[j]["days"] += 1
            r = np.mean(rets) if rets else 0.0
        else:
            r = 0.0
        nav[t] = nav[t-1]*(1+r*len(held)/SLOTS)     # uninvested slots earn nothing

        # --- decide on TODAY's close, act at tomorrow's close (t+1 handled next loop)
        s = S[t]
        if not np.isfinite(s).any(): continue
        order = np.argsort(-np.nan_to_num(s, nan=-np.inf))
        rank = np.empty(len(s), dtype=np.int64); rank[order] = np.arange(len(s))
        valid = np.isfinite(s)

        cost = 0.0
        for j in list(held):
            h = held[j]
            drop = (not valid[j]) or rank[j] >= exit_rank
            timeout = h["days"] >= max_hold
            stopped = trail > 0 and np.isfinite(px[j]) and px[j] <= h["peak"]*(1-trail)
            if drop or timeout or stopped:
                hold_lens.append(h["days"]); del held[j]
                cost += ONE_WAY/SLOTS; trades += 1
        for j in order:
            if len(held) >= SLOTS: break
            if not valid[j] or rank[j] >= entry_rank or j in held: continue
            if not (np.isfinite(px[j]) and px[j] > 0): continue
            held[j] = dict(entry_px=px[j], days=0, peak=px[j])
            cost += ONE_WAY/SLOTS; trades += 1
        nav[t] *= (1-cost)

    nav = pd.Series(nav, index=dates)
    yrs = (dates[-1]-dates[0]).days/365.25
    dd = float((nav/nav.cummax()-1).min())
    dr = nav.pct_change().dropna()
    return nav, dict(
        cagr=(nav.iloc[-1]**(1/yrs)-1)*100, dd=dd*100,
        sharpe=float(dr.mean()/dr.std()*np.sqrt(252)) if dr.std() > 0 else 0,
        trades_pa=trades/yrs,
        avg_hold=float(np.mean(hold_lens)) if hold_lens else np.nan,
        turnover_pa=trades/yrs/SLOTS*100)


def main():
    C, T = build_panel()
    turn_sm = T.rolling(21).mean()
    vol = C.pct_change().rolling(126).std()*np.sqrt(21)
    malong = C.rolling(210).mean()
    start = int(np.searchsorted(C.index.year.values, FROM_YEAR))
    print(f"panel {C.index[start].date()} -> {C.index[-1].date()} | {C.shape[1]} symbols")
    print(f"costs: {ONE_WAY*100:.3f}% charged on EVERY entry and EVERY exit\n")

    configs = [
        ("mom_daily",     30,  60, 9999, 0.0,  "deployed signal, scanned DAILY"),
        ("mom_daily",     30,  60,   21, 0.0,  "  + 21d max hold"),
        ("st_mom",        30,  60, 9999, 0.0,  "short-term (10d) momentum"),
        ("st_mom",        15,  45, 9999, 0.0,  "  tighter entry / wider exit"),
        ("st_mom",        30,  60,    5, 0.0,  "  + 5d max hold (quick flips)"),
        ("st_mom",        30,  60, 9999, 0.10, "  + 10% trailing stop"),
        ("st_mom_trend",  30,  60, 9999, 0.0,  "short-term mom, uptrends only"),
        ("bounce_st",     30,  60,    5, 0.0,  "SUNTV profile, 5d hold"),
        ("bounce_st",     30,  60,   10, 0.0,  "SUNTV profile, 10d hold"),
        ("bounce_st",     30,  60, 9999, 0.08, "SUNTV profile, 8% trail"),
    ]
    print(f"{'strategy':36} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>7} "
          f"{'trades/yr':>10} {'avg hold':>9} {'turnover':>9}")
    print("-"*94)
    cache = {}
    rows = []
    for kind, er, xr, mh, tr, label in configs:
        if kind not in cache:
            cache[kind] = build_scores(C, turn_sm, vol, malong, kind)
        nav, st = simulate(C, cache[kind], start, er, xr, mh, tr)
        rows.append(dict(label=label, kind=kind, **st))
        print(f"{label:36} {st['cagr']:7.2f}% {st['dd']:7.1f}% {st['sharpe']:7.2f} "
              f"{st['trades_pa']:10.0f} {st['avg_hold']:8.1f}d {st['turnover_pa']:8.0f}%")

    print("\n  REFERENCE — deployed monthly book: ~27.9% CAGR, -44% MaxDD, Sharpe 1.19,")
    print("              ~90 trades/yr, ~30d avg hold, ~600% turnover")
    pd.DataFrame(rows).to_csv(BASE/"data_store/active_manager.csv", index=False)
    print("\nsaved -> data_store/active_manager.csv")


if __name__ == "__main__":
    main()
