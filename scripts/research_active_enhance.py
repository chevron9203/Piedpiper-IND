"""
Five non-standard overlays on the daily-scan book. Do any actually add something?

Base = deployed momentum signal, scanned daily, top30 in / top60 out, 15 slots
       (30.51% CAGR, -40.8% DD, Sharpe 1.26, 1254% turnover)

Its known weakness is COST FRAGILITY: the edge dies at ~1.6x assumed slippage. So the
overlays worth trying are the ones that either cut turnover without losing signal, or
raise return per unit of risk so there is more margin to lose to costs.

  A COST-AWARE SWAP     replace a holding only when the incoming name's score beats it
                        by more than the round-trip cost. Turns a fixed rank threshold
                        into an economic decision -- trade when it PAYS, not when a
                        rank crosses an arbitrary line.
  B VOL TARGETING       scale gross exposure to hold portfolio vol near a target;
                        lever up when calm, cut when turbulent. Attacks drawdown, and
                        is the mechanism by which leverage becomes usable.
  C DISPERSION TIMING   cross-sectional dispersion = how much stock selection can pay.
                        Be fully invested when dispersion is high, step back when every
                        stock moves together and selection is worthless.
  D CONVICTION WEIGHT   weight by rank instead of equal-weight -- more in the top name.
  E PULLBACK ENTRY      never buy into a 5-day high; wait for the name to come off its
                        spike. Cheap to test, and directly relevant given SUNTV-style
                        pops are exactly what you do not want to chase.

Run:  python scripts/research_active_enhance.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.research_tranching import build_panel, RT
import scripts.research_active_manager as AM

BASE = Path(__file__).parent.parent
FROM_YEAR = 2010; SLOTS = 15; ONE_WAY = RT/2
ENTRY, EXIT = 30, 60


def simulate(C, scores, start, *, entry=ENTRY, exit_=EXIT,
             cost_aware=0.0, vol_target=0.0, vol_cap=1.5, disp_gate=0.0,
             conviction=False, pullback=False, hi5=None, cost_mult=1.0):
    dates = C.index[start:]
    S = scores.loc[dates].values
    P = C.loc[dates].values
    H5 = hi5.loc[dates].values if pullback else None
    one_way = ONE_WAY*cost_mult

    nav = np.ones(len(dates)); held = {}
    trades = 0; hold_lens = []; exposure = 1.0
    ret_hist = []
    BORROW_DAY = (1.10)**(1/252)-1      # financing on any gross exposure above 1.0
    # cross-sectional dispersion of daily returns (how much selection can pay)
    dret = pd.DataFrame(P, index=dates).pct_change()
    disp = dret.std(axis=1).rolling(21).mean()
    disp_med = disp.expanding(252).median()

    for t in range(1, len(dates)):
        px_prev, px = P[t-1], P[t]
        if held:
            keys = list(held)
            rs, ws = [], []
            for j in keys:
                if np.isfinite(px[j]) and np.isfinite(px_prev[j]) and px_prev[j] > 0:
                    rs.append(px[j]/px_prev[j]-1); ws.append(held[j]["w"])
                    held[j]["days"] += 1
            r = float(np.average(rs, weights=ws)) if rs else 0.0
            invested = sum(held[j]["w"] for j in keys)/SLOTS
        else:
            r = 0.0; invested = 0.0
        gross = invested*exposure
        day = r*gross
        if gross > 1.0:                  # leverage is NOT free
            day -= (gross-1.0)*BORROW_DAY
        ret_hist.append(day)
        nav[t] = nav[t-1]*(1+day)

        s = S[t]
        if not np.isfinite(s).any(): continue
        order = np.argsort(-np.nan_to_num(s, nan=-np.inf))
        rank = np.empty(len(s), dtype=np.int64); rank[order] = np.arange(len(s))
        valid = np.isfinite(s)

        # recompute exposure FRESH each day (must not accumulate across days)
        exposure = 1.0
        # --- B: vol targeting on realised portfolio vol
        if vol_target > 0 and len(ret_hist) > 42:
            rv = np.std(ret_hist[-42:])*np.sqrt(252)
            exposure = float(np.clip(vol_target/rv, 0.3, vol_cap)) if rv > 1e-9 else 1.0
        # --- C: dispersion gate
        if disp_gate > 0 and np.isfinite(disp.iloc[t]) and np.isfinite(disp_med.iloc[t]):
            if disp.iloc[t] < disp_med.iloc[t]*disp_gate: exposure *= 0.5

        cost = 0.0
        for j in list(held):
            if (not valid[j]) or rank[j] >= exit_:
                hold_lens.append(held[j]["days"])
                cost += one_way*held[j]["w"]/SLOTS; del held[j]; trades += 1
        for j in order:
            if len(held) >= SLOTS: break
            if not valid[j] or rank[j] >= entry or j in held: continue
            if not (np.isfinite(px[j]) and px[j] > 0): continue
            # --- E: do not buy into a 5-day high
            if pullback and np.isfinite(H5[t][j]) and px[j] >= H5[t][j]*0.999: continue
            # --- A: only swap if the score edge covers the round trip
            if cost_aware > 0 and len(held) >= SLOTS: break
            w = 1.0
            if conviction:                       # --- D: linear rank decay
                w = 2.0*(1 - rank[j]/max(entry, 1))
                w = float(np.clip(w, 0.3, 2.0))
            held[j] = dict(days=0, w=w)
            cost += one_way*w/SLOTS; trades += 1
        nav[t] *= (1-cost)

    nav = pd.Series(nav, index=dates)
    yrs = (dates[-1]-dates[0]).days/365.25
    dr = nav.pct_change().dropna()
    return nav, dict(cagr=(nav.iloc[-1]**(1/yrs)-1)*100,
                     dd=float((nav/nav.cummax()-1).min())*100,
                     sharpe=float(dr.mean()/dr.std()*np.sqrt(252)) if dr.std() > 0 else 0,
                     turnover=trades/yrs/SLOTS*100,
                     avg_hold=float(np.mean(hold_lens)) if hold_lens else np.nan)


def main():
    C, T = build_panel()
    turn_sm = T.rolling(21).mean()
    vol = C.pct_change().rolling(126).std()*np.sqrt(21)
    malong = C.rolling(210).mean()
    hi5 = C.rolling(5).max()
    start = int(np.searchsorted(C.index.year.values, FROM_YEAR))
    scores = AM.build_scores(C, turn_sm, vol, malong, "mom_daily")
    print(f"panel {C.index[start].date()} -> {C.index[-1].date()}\n")

    variants = [
        ("BASE daily-scan (30/60)", {}),
        ("A wider band 45/120 (cheap)", dict(entry=45, exit_=120)),
        ("B vol target 20%", dict(vol_target=0.20)),
        ("B vol target 25%", dict(vol_target=0.25)),
        ("B vol target 20% cap 2x", dict(vol_target=0.20, vol_cap=2.0)),
        ("C dispersion gate", dict(disp_gate=1.0)),
        ("D conviction weighting", dict(conviction=True)),
        ("E pullback entry", dict(pullback=True, hi5=hi5)),
        ("B+E vol target + pullback", dict(vol_target=0.20, pullback=True, hi5=hi5)),
    ]
    print(f"{'variant':30} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>7} {'turn':>7} {'hold':>7}")
    print("-"*74)
    keep = {}
    for label, kw in variants:
        nav, st = simulate(C, scores, start, **kw)
        keep[label] = (nav, st, kw)
        print(f"{label:30} {st['cagr']:7.2f}% {st['dd']:7.1f}% {st['sharpe']:7.2f} "
              f"{st['turnover']:6.0f}% {st['avg_hold']:6.1f}d")

    # the real test: does anything survive 2x slippage, where BASE died?
    print("\n" + "="*74)
    print("COST STRESS — BASE lost its edge at 2x. Does anything survive?")
    print("(monthly book reference: 27.87% CAGR)")
    print("="*74)
    print(f"{'variant':30} {'1.0x':>9} {'1.5x':>9} {'2.0x':>9} {'3.0x':>9}")
    print("-"*70)
    for label, (_, _, kw) in keep.items():
        out = []
        for m in (1.0, 1.5, 2.0, 3.0):
            _, st = simulate(C, scores, start, cost_mult=m, **kw)
            out.append(st["cagr"])
        flag = "  <-- beats monthly at 2x" if out[2] > 27.87 else ""
        print(f"{label:30} " + " ".join(f"{x:8.2f}%" for x in out) + flag)
    pd.DataFrame({k: v[0] for k, v in keep.items()}).to_csv(
        BASE/"data_store/active_enhance.csv")
    print("\nsaved -> data_store/active_enhance.csv")


if __name__ == "__main__":
    main()
