"""
Stress-test the daily-scan result before anyone believes it.

The active-manager run showed the DEPLOYED (slow) signal, scanned daily with
hysteresis bands, beating the monthly book on all three metrics:
    daily scan  30.51% CAGR / -40.8% DD / Sharpe 1.26
    monthly     27.87% CAGR / -44.3% DD / Sharpe 1.19
That is a big claim built on one parameter choice and one cost assumption, at 1254%
annual turnover. Three ways it could be fake, each tested here:

  1 PARAMETERS  is it a plateau or a lucky (entry_rank, exit_rank, slots) cell?
  2 COSTS       at 1254% turnover, does it survive 2x / 3x slippage? Mid-caps are
                not free to trade and the deployed 0.225%/side may be optimistic.
  3 REGIME      does it hold out-of-sample (2019-26), or is it a pre-2018 artifact?

Run:  python scripts/research_active_validate.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.research_tranching import build_panel
import scripts.research_active_manager as AM

BASE = Path(__file__).parent.parent
FROM_YEAR = 2010


def seg_stats(nav, lo, hi):
    s = nav[(nav.index.year >= lo) & (nav.index.year <= hi)]
    if len(s) < 50: return dict(cagr=np.nan, dd=np.nan, sharpe=np.nan)
    s = s/s.iloc[0]
    yrs = (s.index[-1]-s.index[0]).days/365.25
    dr = s.pct_change().dropna()
    return dict(cagr=(s.iloc[-1]**(1/yrs)-1)*100,
                dd=float((s/s.cummax()-1).min())*100,
                sharpe=float(dr.mean()/dr.std()*np.sqrt(252)) if dr.std() > 0 else 0)


def main():
    C, T = build_panel()
    turn_sm = T.rolling(21).mean()
    vol = C.pct_change().rolling(126).std()*np.sqrt(21)
    malong = C.rolling(210).mean()
    start = int(np.searchsorted(C.index.year.values, FROM_YEAR))
    scores = AM.build_scores(C, turn_sm, vol, malong, "mom_daily")
    print(f"panel {C.index[start].date()} -> {C.index[-1].date()}\n")

    # ---------------------------------------------------- 1. parameter plateau
    print("="*84); print("1  PARAMETER SENSITIVITY — plateau or lucky cell?"); print("="*84)
    print(f"{'entry/exit ranks':22} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>7} {'turnover':>9}")
    print("-"*60)
    grid = [(15, 30), (15, 45), (20, 40), (30, 45), (30, 60), (30, 90),
            (45, 60), (45, 90), (60, 120)]
    best = None
    for er, xr in grid:
        nav, st = AM.simulate(C, scores, start, er, xr, 9999, 0.0)
        tag = f"top{er} in / top{xr} out"
        star = ""
        if (er, xr) == (30, 60): star = "  <- tested config"
        print(f"{tag:22} {st['cagr']:7.2f}% {st['dd']:7.1f}% {st['sharpe']:7.2f} "
              f"{st['turnover_pa']:8.0f}%{star}")
        if best is None or st["cagr"] > best[1]["cagr"]: best = ((er, xr), st, nav)

    # ---------------------------------------------------- 2. cost stress
    print("\n" + "="*84); print("2  COST STRESS at the tested config (30/60)"); print("="*84)
    print(f"{'slippage assumption':26} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>7} {'vs monthly':>11}")
    print("-"*64)
    orig = AM.ONE_WAY
    for mult in (1.0, 1.5, 2.0, 3.0):
        AM.ONE_WAY = orig*mult
        nav, st = AM.simulate(C, scores, start, 30, 60, 9999, 0.0)
        print(f"{f'{mult:.1f}x  ({orig*mult*100:.3f}%/side)':26} {st['cagr']:7.2f}% "
              f"{st['dd']:7.1f}% {st['sharpe']:7.2f} {st['cagr']-27.87:+10.2f}pp")
    AM.ONE_WAY = orig

    # ---------------------------------------------------- 3. IS / OOS
    print("\n" + "="*84); print("3  IN-SAMPLE vs OUT-OF-SAMPLE (tested config)"); print("="*84)
    nav, st = AM.simulate(C, scores, start, 30, 60, 9999, 0.0)
    print(f"{'window':16} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>7}")
    print("-"*44)
    for lbl, lo, hi in [("full 2010-26", 2010, 2026), ("IS 2010-18", 2010, 2018),
                        ("OOS 2019-26", 2019, 2026)]:
        s = seg_stats(nav, lo, hi)
        print(f"{lbl:16} {s['cagr']:7.2f}% {s['dd']:7.1f}% {s['sharpe']:7.2f}")

    # ---------------------------------------------------- 4. slot count
    print("\n" + "="*84); print("4  PORTFOLIO SIZE"); print("="*84)
    print(f"{'slots':10} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>7} {'turnover':>9}")
    print("-"*48)
    orig_slots = AM.SLOTS
    for n in (10, 15, 20, 25):
        AM.SLOTS = n
        nav_, st_ = AM.simulate(C, scores, start, 30, 60, 9999, 0.0)
        print(f"{n:<10} {st_['cagr']:7.2f}% {st_['dd']:7.1f}% {st_['sharpe']:7.2f} "
              f"{st_['turnover_pa']:8.0f}%")
    AM.SLOTS = orig_slots

    nav.to_frame("nav").to_csv(BASE/"data_store/active_validate_nav.csv")
    print("\nsaved -> data_store/active_validate_nav.csv")


if __name__ == "__main__":
    main()
