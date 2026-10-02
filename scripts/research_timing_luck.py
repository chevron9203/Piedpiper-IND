"""
How much of the momentum book's backtested return is just WHICH DAY we rebalance?

Runs the identical strategy 21 times, once for every possible start day inside the
month, and reports the distribution. Then builds K-tranche portfolios (capital split
across K staggered start days) to show how much of that dispersion tranching removes.

This is the honest denominator for every CAGR number we quote.

Run:  python scripts/research_timing_luck.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np, pandas as pd
from scripts.research_tranching import (build_panel, run_offset, stats, seg,
                                        STEP, L3, L6, L12, MA_LONG, FROM_YEAR, IS_END)

BASE = Path(__file__).parent.parent


def main():
    C, T = build_panel()
    turn_sm = T.rolling(21).mean()
    feats = (C/C.shift(L3)-1, C/C.shift(L6)-1, C/C.shift(L12)-1,
             C.pct_change().rolling(126).std()*np.sqrt(21),
             C.rolling(MA_LONG).mean(), turn_sm)
    start = int(np.searchsorted(C.index.year.values, FROM_YEAR))
    print(f"panel {C.index[start].date()} -> {C.index[-1].date()} | {C.shape[1]} symbols")
    print(f"running all {STEP} start-day offsets ...\n")

    navs = {}
    for o in range(STEP):
        navs[o] = run_offset(C, T, feats, start, o)
        s = stats(navs[o]/navs[o].iloc[0])
        print(f"  offset {o:2d}d  CAGR {s['cagr']:6.2f}%  MaxDD {s['dd']:6.1f}%  Sharpe {s['sh']:.2f}")

    common = navs[0].index
    for o in navs: common = common.intersection(navs[o].index)
    navs = {o: (v.reindex(common).ffill()) for o, v in navs.items()}
    navs = {o: v/v.iloc[0] for o, v in navs.items()}

    cagrs = np.array([stats(navs[o])['cagr'] for o in range(STEP)])
    dds   = np.array([stats(navs[o])['dd']   for o in range(STEP)])
    print("\n" + "="*78)
    print("TIMING LUCK — same strategy, only the rebalance day differs")
    print("="*78)
    print(f"  CAGR   mean {cagrs.mean():6.2f}%   sd {cagrs.std():5.2f}pp   "
          f"min {cagrs.min():6.2f}%   max {cagrs.max():6.2f}%   spread {cagrs.max()-cagrs.min():5.2f}pp")
    print(f"  MaxDD  mean {dds.mean():6.1f}%    worst {dds.min():6.1f}%   best {dds.max():6.1f}%")

    print("\n" + "="*78)
    print("TRANCHING — split capital across K staggered start days")
    print("="*78)
    print(f"{'K':>3} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>7} {'IS':>8} {'OOS':>8} {'OOS DD':>8}")
    print("-"*60)
    rows = []
    for K in (1, 2, 3, 4, 7, 21):
        offs = [round(i*STEP/K) % STEP for i in range(K)]
        nav = sum(navs[o] for o in offs)/K
        nav = nav/nav.iloc[0]
        a = stats(nav); i = seg(nav, 2010, IS_END); o_ = seg(nav, IS_END+1, 2026)
        rows.append((K, a, i, o_))
        tag = f"{K}" if K > 1 else "1*"
        print(f"{tag:>3} {a['cagr']:7.2f}% {a['dd']:7.1f}% {a['sh']:7.2f} "
              f"{i['cagr']:7.2f}% {o_['cagr']:7.2f}% {o_['dd']:7.1f}%")
    print("\n  * K=1 shown at offset 0 = today's deployed behaviour (one month-end trade)")
    print(f"    K=1 across ALL start days actually ranges {cagrs.min():.2f}%..{cagrs.max():.2f}%")

    pd.DataFrame({f"off_{o}": navs[o] for o in range(STEP)}).to_csv(
        BASE/"data_store/timing_luck_navs.csv")
    print("\nsaved -> data_store/timing_luck_navs.csv")


if __name__ == "__main__":
    main()
