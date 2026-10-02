"""
Vol targeting on the DEPLOYED MONTHLY book -- validated the hard way.

Why this and not the rest: the overlay screen ran on the daily-scan base, and that base
turned out to be a 2020 artifact (41% year win rate, identical to monthly ex-2020), so
"improves on daily-scan" means nothing. Vol targeting is the one idea that (a) raised
Sharpe rather than just return, and (b) does NOT need daily scanning -- it is a
portfolio-level risk control, which is where this system's edge has actually come from.

Mechanic: scale gross exposure by target_vol / trailing_realised_vol, rebalanced monthly
(NOT daily -- no extra stock turnover, just a position-size dial). Leverage above 1.0 is
charged financing at 10%/yr. Exposure is recomputed fresh each period.

Validated with the tests that killed the daily-scan idea:
  * year-by-year win rate (not expanding windows -- those are not independent)
  * performance excluding 2020 and excluding the top-3 years
  * block bootstrap 95% CI on the return DIFFERENCE

Run:  python scripts/research_voltarget_monthly.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.research_tranching import build_panel, STEP
from scripts.research_robustness import make_feats, run_one, DEFAULTS

BASE = Path(__file__).parent.parent
FROM_YEAR = 2010
BORROW_DAY = (1.10)**(1/252)-1


def monthly_nav(C, turn_sm, start, n_off=7):
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


def apply_voltarget(nav, target, cap, lookback=42, reb=STEP):
    """Scale exposure to hold realised vol near target. Re-set every `reb` days only,
    so this is a position-size dial, not extra trading."""
    r = nav.pct_change().fillna(0.0).values
    out = np.ones(len(r)); expo = 1.0
    for t in range(1, len(r)):
        if t % reb == 0 and t > lookback:
            rv = np.std(r[t-lookback:t])*np.sqrt(252)
            expo = float(np.clip(target/rv, 0.3, cap)) if rv > 1e-9 else 1.0
        d = r[t]*expo
        if expo > 1.0: d -= (expo-1.0)*BORROW_DAY
        out[t] = out[t-1]*(1+d)
    return pd.Series(out, index=nav.index)


def stats(nav):
    nav = nav.dropna()/nav.dropna().iloc[0]
    yrs = (nav.index[-1]-nav.index[0]).days/365.25
    dr = nav.pct_change().dropna()
    return dict(cagr=(nav.iloc[-1]**(1/yrs)-1)*100,
                dd=float((nav/nav.cummax()-1).min())*100,
                sharpe=float(dr.mean()/dr.std()*np.sqrt(252)) if dr.std() > 0 else 0)


def yearly(a, b):
    rows = []
    for y in sorted(set(a.index.year)):
        aa, bb = a[a.index.year == y], b[b.index.year == y]
        if len(aa) < 50: continue
        rows.append((y, (aa.iloc[-1]/aa.iloc[0]-1)*100, (bb.iloc[-1]/bb.iloc[0]-1)*100))
    d = pd.DataFrame(rows, columns=["year", "vt", "base"])
    d["diff"] = d.vt-d.base
    return d


def bootstrap(a, b, n=5000, block=63, seed=11):
    ra, rb = a.pct_change().dropna(), b.pct_change().dropna()
    j = pd.concat([ra, rb], axis=1).dropna(); j.columns = ["a", "b"]
    d = (j["a"]-j["b"]).values; n_ = len(d)
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        acc = []
        while len(acc) < n_:
            s = rng.integers(0, n_-block); acc.extend(d[s:s+block])
        out.append(np.mean(acc[:n_])*252*100)
    out = np.array(out)
    return out.mean(), np.percentile(out, 2.5), np.percentile(out, 97.5), (out > 0).mean()*100


def main():
    C, T = build_panel()
    turn_sm = T.rolling(21).mean()
    start = int(np.searchsorted(C.index.year.values, FROM_YEAR))
    print("building deployed monthly book (offset-averaged) ...")
    base = monthly_nav(C, turn_sm, start)
    s0 = stats(base)
    print(f"  baseline: {s0['cagr']:.2f}% CAGR / {s0['dd']:.1f}% DD / Sharpe {s0['sharpe']:.2f}\n")

    print("="*78); print("VOL TARGETING ON THE MONTHLY BOOK"); print("="*78)
    print(f"{'config':28} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>7} {'vs base':>9}")
    print("-"*64)
    cands = {}
    for tgt in (0.18, 0.20, 0.25, 0.30):
        for cap in (1.0, 1.5):
            nav = apply_voltarget(base, tgt, cap)
            s = stats(nav); lbl = f"target {int(tgt*100)}% cap {cap:g}x"
            cands[lbl] = nav
            print(f"{lbl:28} {s['cagr']:7.2f}% {s['dd']:7.1f}% {s['sharpe']:7.2f} "
                  f"{s['cagr']-s0['cagr']:+8.2f}pp")

    # take the best-Sharpe candidate through the hard tests
    best = max(cands, key=lambda k: stats(cands[k])["sharpe"])
    nav = cands[best]
    print(f"\n{'='*78}\nHARD TESTS on best-Sharpe config: {best}\n{'='*78}")
    y = yearly(nav, base)
    print(f"  year win rate      {100*(y['diff']>0).mean():.0f}%  ({(y['diff']>0).sum()}/{len(y)})")
    print(f"  mean annual diff   {y['diff'].mean():+.2f}pp   MEDIAN {y['diff'].median():+.2f}pp")
    ex = y[y.year != 2020]
    print(f"  excluding 2020     mean {ex['diff'].mean():+.2f}pp  median {ex['diff'].median():+.2f}pp  "
          f"win {100*(ex['diff']>0).mean():.0f}%")
    top3 = y.nlargest(3, "diff").year.tolist()
    ex3 = y[~y.year.isin(top3)]
    print(f"  excluding top-3 {top3}  mean {ex3['diff'].mean():+.2f}pp  "
          f"win {100*(ex3['diff']>0).mean():.0f}%")
    m, lo, hi, p = bootstrap(nav, base)
    print(f"\n  block bootstrap    edge {m:+.2f}pp   95% CI [{lo:+.2f}, {hi:+.2f}]   P(>0) {p:.1f}%")
    print(f"  VERDICT: {'HOLDS — CI excludes zero' if lo > 0 else 'NOT CONCLUSIVE — CI includes zero'}")
    print(f"\n  year-by-year diff:")
    print("   " + "  ".join(f"{int(r.year)}:{r['diff']:+.1f}" for _, r in y.iterrows()))
    pd.DataFrame({"base": base, "voltarget": nav}).to_csv(BASE/"data_store/voltarget_monthly.csv")
    print("\nsaved -> data_store/voltarget_monthly.csv")


if __name__ == "__main__":
    main()
