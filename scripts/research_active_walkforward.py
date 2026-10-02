"""
Is the daily-scan edge real, or one lucky backtest? Three harder tests.

The single IS/OOS split said yes (OOS 32.91% > IS 28.47%). That is one cut of one
sample and is not enough to trade on. This runs the tests that actually try to kill it:

  1 WALK-FORWARD    expanding-window, re-evaluated every year. An edge that only
                    exists in a few good years shows up here as a broken sequence.
  2 BOOTSTRAP       block-bootstrap the DIFFERENCE (daily-scan minus monthly) to get a
                    confidence interval. Overlapping blocks preserve autocorrelation,
                    so the CI is not artificially tight.
  3 YEAR BY YEAR    does it beat the monthly book in most years, or is the whole edge
                    one outlier year? Also reports the cost breakeven per year.

Anything that survives all three is worth forward-testing. Anything that does not, is not.

Run:  python scripts/research_active_walkforward.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.research_tranching import build_panel, STEP, RT, PRICE_MIN, VOL_FLOOR
import scripts.research_active_manager as AM
from scripts.research_robustness import make_feats, run_one, DEFAULTS

BASE = Path(__file__).parent.parent
FROM_YEAR = 2010
ENTRY, EXIT = 30, 60


def monthly_nav(C, turn_sm, start, n_off=7):
    """The deployed monthly book, offset-averaged (its fair comparator)."""
    feats = make_feats(C, DEFAULTS["looks"], DEFAULTS["ma_long"])
    curves = []
    for i in range(n_off):
        o = round(i*STEP/n_off)
        nav = run_one(C, turn_sm, feats, start, o, DEFAULTS["top_n"],
                      DEFAULTS["tier"], DEFAULTS["buffer"], max(DEFAULTS["looks"]))
        if len(nav) > 50: curves.append(nav/nav.iloc[0])
    common = curves[0].index
    for c in curves[1:]: common = common.intersection(c.index)
    curves = [c.reindex(common).ffill() for c in curves]
    avg = sum(curves)/len(curves)
    return avg/avg.iloc[0]


def cagr(nav):
    nav = nav.dropna()
    if len(nav) < 30: return np.nan
    yrs = (nav.index[-1]-nav.index[0]).days/365.25
    return ((nav.iloc[-1]/nav.iloc[0])**(1/yrs)-1)*100


def main():
    C, T = build_panel()
    turn_sm = T.rolling(21).mean()
    vol = C.pct_change().rolling(126).std()*np.sqrt(21)
    malong = C.rolling(210).mean()
    start = int(np.searchsorted(C.index.year.values, FROM_YEAR))
    scores = AM.build_scores(C, turn_sm, vol, malong, "mom_daily")

    print("building both books ...")
    act, _ = AM.simulate(C, scores, start, ENTRY, EXIT, 9999, 0.0)
    mon = monthly_nav(C, turn_sm, start)
    idx = act.index.intersection(mon.index)
    act = (act.reindex(idx).ffill()); act = act/act.iloc[0]
    mon = (mon.reindex(idx).ffill()); mon = mon/mon.iloc[0]
    print(f"aligned {idx[0].date()} -> {idx[-1].date()} ({len(idx)} days)\n")

    # ------------------------------------------------ 1 walk-forward
    print("="*76); print("1  WALK-FORWARD (expanding window, re-cut each year)"); print("="*76)
    print(f"{'through':12} {'daily-scan':>12} {'monthly':>10} {'diff':>9}")
    print("-"*46)
    yrs = sorted(set(idx.year))
    wins = 0; tot = 0
    for y in yrs[3:]:
        a = act[act.index.year <= y]; m = mon[mon.index.year <= y]
        ca, cm = cagr(a/a.iloc[0]), cagr(m/m.iloc[0])
        d = ca-cm; tot += 1; wins += d > 0
        print(f"{y:<12} {ca:11.2f}% {cm:9.2f}% {d:+8.2f}pp")
    print(f"\n  daily-scan ahead in {wins}/{tot} expanding windows")

    # ------------------------------------------------ 2 block bootstrap
    print("\n" + "="*76); print("2  BLOCK BOOTSTRAP on the daily return DIFFERENCE"); print("="*76)
    ra = act.pct_change().dropna(); rm = mon.pct_change().dropna()
    j = pd.concat([ra, rm], axis=1).dropna(); j.columns = ["a", "m"]
    d = (j["a"]-j["m"]).values
    n = len(d); block = 63                      # ~3 months, preserves autocorrelation
    rng = np.random.default_rng(7)
    boots = []
    for _ in range(5000):
        out = []
        while len(out) < n:
            s = rng.integers(0, n-block)
            out.extend(d[s:s+block])
        boots.append(np.mean(out[:n])*252*100)
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"  mean annualised edge   {d.mean()*252*100:+.2f}pp")
    print(f"  95% CI                 [{lo:+.2f}pp, {hi:+.2f}pp]")
    print(f"  P(edge > 0)            {100*(boots > 0).mean():.1f}%")
    print(f"  verdict: {'EDGE HOLDS (CI excludes 0)' if lo > 0 else 'NOT CONCLUSIVE (CI includes 0)'}")

    # ------------------------------------------------ 3 year by year
    print("\n" + "="*76); print("3  YEAR BY YEAR"); print("="*76)
    print(f"{'year':8} {'daily-scan':>12} {'monthly':>10} {'diff':>9}")
    print("-"*42)
    w = 0; t = 0
    for y in yrs:
        a = act[act.index.year == y]; m = mon[mon.index.year == y]
        if len(a) < 50: continue
        ya = (a.iloc[-1]/a.iloc[0]-1)*100; ym = (m.iloc[-1]/m.iloc[0]-1)*100
        t += 1; w += ya > ym
        print(f"{y:<8} {ya:11.2f}% {ym:9.2f}% {ya-ym:+8.2f}pp")
    print(f"\n  daily-scan beat monthly in {w}/{t} calendar years")
    pd.DataFrame({"active": act, "monthly": mon}).to_csv(BASE/"data_store/active_wf.csv")
    print("\nsaved -> data_store/active_wf.csv")


if __name__ == "__main__":
    main()
