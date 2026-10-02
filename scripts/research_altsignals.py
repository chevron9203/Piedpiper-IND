"""
Could a CONTRARIAN / BOUNCE sleeve add what momentum structurally cannot?

Prompted by a real observation: on 2026-09-24 the market fell hard (only 17.7% of
stocks up, median -1.26%) and SUNTV rose +4.59%. SUNTV is inside our mid-cap tier but
was NOT eligible -- it sits below its 210d MA, and is -13.5% over 6m / -8.4% over 12m
with +9.3% over the last month. Momentum is built to refuse exactly that. A different
strategy family holds it.

The point is NOT "is reversal better than momentum" -- it very likely is not. It is
whether a NEGATIVELY CORRELATED sleeve improves the combined book even while being
weaker standalone. That is the only reason to add complexity.

Signals (same universe, same costs, top-15, 21-day rebalance):
  mom       deployed multi-TF risk-adjusted momentum + trend filter   (baseline)
  rev1m     classic short-term reversal: buy the worst 1-month losers
  fast1m    short-term momentum: buy the best 1-month winners
  bounce    SUNTV's profile: beaten down 6m/12m, BUT strong last month
  bounce_v  bounce, additionally requiring a volume spike (event confirmation)

Every config is averaged over N_OFF rebalance offsets, because timing luck (+/-2.07pp
sd, measured 2026-09-23) otherwise swamps the comparison.

Run:  python scripts/research_altsignals.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.research_tranching import build_panel, stats, seg, STEP, RT, PRICE_MIN, VOL_FLOOR

BASE = Path(__file__).parent.parent
FROM_YEAR = 2010; IS_END = 2018; TOP_N = 15; N_OFF = 7


def run_signal(C, turn_sm, vol, malong, feats, start, offset, kind):
    """One offset of one signal. Daily mark-to-market NAV."""
    m1, m3, m6, m12, volspike = feats
    dates = C.index
    pts = list(range(start+offset, len(dates)-1, STEP))
    parts = []; level = 1.0; prev = []
    for a, b in zip(pts, pts[1:]):
        pr = C.iloc[a]; tn = turn_sm.iloc[a]; v = vol.iloc[a]
        hist = C.iloc[:a+1].notna().sum()
        base = pr.index[(pr >= PRICE_MIN) & (hist >= 252) & tn.notna() & (tn > 0)
                        & (v >= VOL_FLOOR)]
        seg_idx = dates[a+1:b+1]
        picks = []
        if len(base) >= 100:
            rank = tn[base].rank(pct=True)
            mid = list(rank.index[(rank > 0.60) & (rank <= 0.90)])
            ml = malong.iloc[a]
            if kind == "mom":
                elig = [s for s in mid if pr[s] > ml.get(s, np.inf)]
                sc = pd.Series(0.0, index=elig); nc = 0
                for sg in (m3, m6, m12):
                    x = (sg.iloc[a][elig]/v[elig].replace(0, np.nan)).dropna()
                    if len(x): sc = sc.add(x.rank(pct=True), fill_value=0); nc += 1
                ranked = (sc/nc).sort_values(ascending=False) if nc else pd.Series(dtype=float)
            elif kind == "rev1m":
                elig = mid
                ranked = m1.iloc[a][elig].dropna().sort_values()          # worst first
            elif kind == "fast1m":
                elig = mid
                ranked = m1.iloc[a][elig].dropna().sort_values(ascending=False)
            elif kind in ("bounce", "bounce_v"):
                # SUNTV profile: beaten down over 6m AND 12m, but strong last month
                elig = [s for s in mid
                        if m6.iloc[a].get(s, 0) < 0 and m12.iloc[a].get(s, 0) < 0
                        and m1.iloc[a].get(s, -1) > 0]
                if kind == "bounce_v":
                    elig = [s for s in elig if volspike.iloc[a].get(s, 0) > 1.5]
                ranked = m1.iloc[a][elig].dropna().sort_values(ascending=False)
            else:
                ranked = pd.Series(dtype=float)
            if len(ranked):
                rk = {s: i for i, s in enumerate(ranked.index)}
                picks = [s for s in prev if rk.get(s, 10**9) < TOP_N*1.67]
                for s in ranked.index:
                    if len(picks) >= TOP_N: break
                    if s not in picks: picks.append(s)
                picks = picks[:TOP_N]
        held = [s for s in picks
                if np.isfinite(C.iloc[a].get(s, np.nan)) and C.iloc[a].get(s, 0) > 0]
        churn = len(set(prev)-set(held))/max(len(prev), 1) if prev else 1.0
        level *= (1 - churn*RT)
        if held:
            basket = (C.loc[seg_idx, held]/C.iloc[a][held]).mean(axis=1).ffill().fillna(1.0)
        else:
            basket = pd.Series(1.0, index=seg_idx)      # nothing qualifies -> flat
        parts.append(basket*level)
        level = float(basket.iloc[-1]*level)
        prev = held
    return pd.concat(parts) if parts else pd.Series(dtype=float)


def main():
    C, T = build_panel()
    turn_sm = T.rolling(21).mean()
    vol = C.pct_change().rolling(126).std()*np.sqrt(21)
    malong = C.rolling(210).mean()
    V = T/C                                   # share volume proxy
    feats = (C/C.shift(21)-1, C/C.shift(63)-1, C/C.shift(126)-1, C/C.shift(252)-1,
             V/V.rolling(21).mean())
    start = int(np.searchsorted(C.index.year.values, FROM_YEAR))
    print(f"panel {C.index[start].date()} -> {C.index[-1].date()} | {C.shape[1]} symbols")
    print(f"each signal averaged over {N_OFF} rebalance offsets\n")

    offs = [round(i*STEP/N_OFF) for i in range(N_OFF)]
    navs = {}
    print(f"{'signal':26} {'CAGR':>8} {'sd':>6} {'MaxDD':>8} {'Sharpe':>7} {'OOS':>8}")
    print("-"*70)
    for kind, label in [("mom", "momentum (DEPLOYED)"), ("rev1m", "reversal (1m losers)"),
                        ("fast1m", "fast mom (1m winners)"), ("bounce", "bounce (SUNTV profile)"),
                        ("bounce_v", "bounce + volume spike")]:
        curves = []
        for o in offs:
            n = run_signal(C, turn_sm, vol, malong, feats, start, o, kind)
            if len(n) > 50: curves.append(n/n.iloc[0])
        if not curves: print(f"{label:26} -- no data --"); continue
        common = curves[0].index
        for c in curves[1:]: common = common.intersection(c.index)
        curves = [c.reindex(common).ffill() for c in curves]
        avg = sum(curves)/len(curves); avg = avg/avg.iloc[0]
        navs[kind] = avg
        cs = [stats(c)["cagr"] for c in curves]
        a = stats(avg); o_ = seg(avg, IS_END+1, 2026)
        print(f"{label:26} {a['cagr']:7.2f}% {np.std(cs):5.2f} {a['dd']:7.1f}% "
              f"{a['sh']:7.2f} {o_['cagr']:7.2f}%")

    # the actual question: correlation, and does a blend beat momentum alone?
    print("\n" + "="*70); print("CORRELATION WITH THE DEPLOYED BOOK (monthly returns)"); print("="*70)
    mr = navs["mom"].resample("ME").last().pct_change().dropna()
    for k in navs:
        if k == "mom": continue
        r = navs[k].resample("ME").last().pct_change().dropna()
        j = pd.concat([mr, r], axis=1).dropna()
        print(f"  momentum vs {k:22} corr {j.corr().iloc[0,1]:+.3f}")

    print("\n" + "="*70); print("DOES A BLEND BEAT MOMENTUM ALONE?"); print("="*70)
    print(f"{'blend':30} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>7}")
    print("-"*56)
    base = stats(navs["mom"])
    print(f"{'100% momentum':30} {base['cagr']:7.2f}% {base['dd']:7.1f}% {base['sh']:7.2f}")
    for k in navs:
        if k == "mom": continue
        for w in (0.20, 0.30):
            b = (1-w)*navs["mom"] + w*navs[k]; b = b/b.iloc[0]
            s = stats(b)
            flag = "  <-- better Sharpe" if s["sh"] > base["sh"] else ""
            print(f"{f'{int((1-w)*100)}% mom / {int(w*100)}% {k}':30} "
                  f"{s['cagr']:7.2f}% {s['dd']:7.1f}% {s['sh']:7.2f}{flag}")
    pd.DataFrame(navs).to_csv(BASE/"data_store/altsignals.csv")
    print("\nsaved -> data_store/altsignals.csv")


if __name__ == "__main__":
    main()
