"""
Is the momentum config REAL or FITTED? -- and what are the error bars?

Two questions that have to be answered together:
  1. How much of any result is just which day of the month we rebalance?
     (measured 2026-09-23 at +/-2.07pp sd -- large enough to swamp most edges)
  2. Does the deployed config sit on a knife-edge (curve-fit) or a broad plateau (real)?

You cannot answer (2) without (1). Comparing two parameter settings at a single
rebalance offset compares noise: a 2pp "improvement" is inside the luck band. So every
config here is run at N_OFF staggered offsets and reported as mean +/- sd across them.

Baseline = the deployed pure MID-cap momentum config.

Run:  python scripts/research_robustness.py
"""
from __future__ import annotations
import sys, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np, pandas as pd
from scripts.research_tranching import build_panel, stats, seg, STEP, RT, PRICE_MIN, VOL_FLOOR

BASE = Path(__file__).parent.parent
FROM_YEAR = 2010; IS_END = 2018
N_OFF = 7                      # offsets averaged per config (sd of mean ~= 2.07/sqrt(7) ~= 0.8pp)

# deployed configuration
DEFAULTS = dict(top_n=15, tier=(0.60, 0.90), looks=(63, 126, 252), buffer=1.67, ma_long=210)


def make_feats(C, looks, ma_long):
    m3, m6, m12 = (C/C.shift(l)-1 for l in looks)
    vol = C.pct_change().rolling(126).std()*np.sqrt(21)
    malong = C.rolling(ma_long).mean() if ma_long else None
    return m3, m6, m12, vol, malong


def run_one(C, turn_sm, feats, start, offset, top_n, tier, buffer, longest):
    """One offset of one config. Daily mark-to-market NAV."""
    m3, m6, m12, vol, malong = feats
    dates = C.index
    pts = list(range(start+offset, len(dates)-1, STEP))
    parts = []; level = 1.0; prev = []
    for a, b in zip(pts, pts[1:]):
        pr = C.iloc[a]; tn = turn_sm.iloc[a]; v = vol.iloc[a]
        hist = C.iloc[:a+1].notna().sum()
        base = pr.index[(pr >= PRICE_MIN) & (hist >= longest) & tn.notna() & (tn > 0)
                        & (v >= VOL_FLOOR)]
        seg_idx = dates[a+1:b+1]
        picks = []
        if len(base) >= 100:
            rank = tn[base].rank(pct=True)
            mid = rank.index[(rank > tier[0]) & (rank <= tier[1])]
            if malong is not None:
                ml = malong.iloc[a]
                elig = [s for s in mid if pr[s] > ml.get(s, np.inf)]
            else:
                elig = list(mid)
            score = pd.Series(0.0, index=elig); nc = 0
            for sg in (m3, m6, m12):
                x = (sg.iloc[a][elig] / v[elig].replace(0, np.nan)).dropna()
                if len(x): score = score.add(x.rank(pct=True), fill_value=0); nc += 1
            if nc and elig:
                ranked = (score/nc).sort_values(ascending=False)
                rk = {s: i for i, s in enumerate(ranked.index)}
                picks = [s for s in prev if rk.get(s, 10**9) < top_n*buffer]
                for s in ranked.index:
                    if len(picks) >= top_n: break
                    if s not in picks: picks.append(s)
                picks = picks[:top_n]
        held = [s for s in picks
                if np.isfinite(C.iloc[a].get(s, np.nan)) and C.iloc[a].get(s, 0) > 0]
        churn = len(set(prev)-set(held))/max(len(prev), 1) if prev else 1.0
        level *= (1 - churn*RT)
        if held:
            basket = (C.loc[seg_idx, held]/C.iloc[a][held]).mean(axis=1).ffill().fillna(1.0)
        else:
            basket = pd.Series(1.0, index=seg_idx)
        parts.append(basket*level)
        level = float(basket.iloc[-1]*level)
        prev = held
    return pd.concat(parts) if parts else pd.Series(dtype=float)


def eval_config(C, turn_sm, start, label, **kw):
    """Run a config at N_OFF offsets; report mean +/- sd so differences are readable."""
    p = {**DEFAULTS, **kw}
    feats = make_feats(C, p["looks"], p["ma_long"])
    longest = max(p["looks"])
    offs = [round(i*STEP/N_OFF) for i in range(N_OFF)]
    cs, ds, os_ = [], [], []
    for o in offs:
        nav = run_one(C, turn_sm, feats, start, o, p["top_n"], p["tier"], p["buffer"], longest)
        if len(nav) < 50: continue
        nav = nav/nav.iloc[0]
        s = stats(nav); cs.append(s["cagr"]); ds.append(s["dd"])
        os_.append(seg(nav, IS_END+1, 2026)["cagr"])
    if not cs:
        print(f"{label:34} -- no data --"); return None
    r = dict(label=label, cagr=np.mean(cs), sd=np.std(cs), dd=np.mean(ds), oos=np.mean(os_))
    print(f"{label:34} {r['cagr']:6.2f}% +/-{r['sd']:4.2f}   {r['dd']:7.1f}%   {r['oos']:6.2f}%")
    return r


def main():
    C, T = build_panel()
    turn_sm = T.rolling(21).mean()
    start = int(np.searchsorted(C.index.year.values, FROM_YEAR))
    print(f"panel {C.index[start].date()} -> {C.index[-1].date()} | {C.shape[1]} symbols")
    print(f"each config averaged over {N_OFF} rebalance offsets\n")
    hdr = f"{'config':34} {'CAGR (mean+/-sd)':>16}   {'MaxDD':>7}   {'OOS':>6}"
    rows = []

    print("="*72); print("BASELINE (deployed config)"); print("="*72)
    print(hdr); print("-"*72)
    base = eval_config(C, turn_sm, start, "deployed (15, .60-.90, 3/6/12)")
    rows.append(base)

    print("\n" + "="*72); print("SWEEP: number of positions"); print("="*72)
    print(hdr); print("-"*72)
    for n in (10, 15, 20, 25, 30):
        rows.append(eval_config(C, turn_sm, start, f"top_n = {n}", top_n=n))

    print("\n" + "="*72); print("SWEEP: liquidity tier (turnover percentile)"); print("="*72)
    print(hdr); print("-"*72)
    for t in ((0.50, 0.80), (0.60, 0.90), (0.70, 0.95), (0.85, 1.00), (0.0, 1.0)):
        rows.append(eval_config(C, turn_sm, start, f"tier = {t[0]:.2f}-{t[1]:.2f}", tier=t))

    print("\n" + "="*72); print("SWEEP: momentum lookbacks (trading days)"); print("="*72)
    print(hdr); print("-"*72)
    for lk in ((21, 63, 126), (42, 84, 189), (63, 126, 252), (84, 168, 336), (126, 252, 378)):
        rows.append(eval_config(C, turn_sm, start, f"looks = {lk}", looks=lk))

    print("\n" + "="*72); print("SWEEP: rank buffer / trend filter"); print("="*72)
    print(hdr); print("-"*72)
    for b in (1.0, 1.33, 1.67, 2.0, 2.5):
        rows.append(eval_config(C, turn_sm, start, f"buffer = {b}", buffer=b))
    for m in (0, 100, 210, 252):
        rows.append(eval_config(C, turn_sm, start, f"trend MA = {m or 'OFF'}", ma_long=m))

    rows = [r for r in rows if r]
    df = pd.DataFrame(rows)
    df.to_csv(BASE/"data_store/robustness_results.csv", index=False)

    print("\n" + "="*72); print("VERDICT"); print("="*72)
    b = base["cagr"]; band = base["sd"]
    better = df[df.cagr > b + band]
    print(f"baseline {b:.2f}% (offset sd {band:.2f}pp). A config must beat "
          f"{b+band:.2f}% to be distinguishable from luck.")
    if len(better):
        print(f"configs clearing that bar ({len(better)}):")
        for _, r in better.sort_values("cagr", ascending=False).iterrows():
            print(f"   {r['label']:34} {r['cagr']:6.2f}% (+{r['cagr']-b:.2f}pp)")
    else:
        print("NO config beats the baseline by more than one offset-sd "
              "-> the deployed settings are on a plateau, not a fitted peak.")
    spread = df.cagr.max()-df.cagr.min()
    print(f"\nfull sweep spread {df.cagr.min():.2f}%..{df.cagr.max():.2f}% ({spread:.2f}pp) "
          f"vs timing-luck band ~{band*2:.2f}pp")
    print("saved -> data_store/robustness_results.csv")


if __name__ == "__main__":
    main()
