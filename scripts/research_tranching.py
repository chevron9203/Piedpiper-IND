"""
TRANCHING: does staggering the rebalance across the month beat one big month-end trade?

The complaint this tests: a name that starts running on the 5th sits un-owned until
the next month-end. Weekly rebalance already failed (tested 2026-09-03: more turnover,
worse everything). Tranching is different -- each slice of capital still holds for a
full month, we just stop rebalancing the WHOLE book on one arbitrary day. That cuts
timing luck and lets new names in within ~21/K days, without raising per-name turnover.

Everything is run on a 21-trading-day grid (not calendar month-end) so the baseline
and the tranched version differ ONLY by the staggering.

Run:  python scripts/research_tranching.py
"""
from __future__ import annotations
import sys, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np, pandas as pd

BASE = Path(__file__).parent.parent
DAILY = BASE/"data_store/eod2/src/eod2_data/daily"
PANEL = BASE/"data_store/daily_panel_full.parquet"

FROM_YEAR = 2010; IS_END = 2018
TOP_N = 15; PRICE_MIN = 30.0; VOL_FLOOR = 0.02
RT = 2*(0.0010+0.0010)+0.0005
STEP = 21                                  # trading days between a tranche's rebalances
L3, L6, L12 = 63, 126, 252                 # momentum lookbacks in trading days
MA_LONG = 210                              # ~10 month trend filter


def is_stock(sym):
    s = str(sym).upper()
    return not (s.endswith("BEES") or s.endswith("ETF") or s.endswith("IETF")
                or any(p in s for p in ("LIQUID","GILT","GSEC","BHARATBOND","CASHIETF")))


def build_panel():
    """Full daily close + turnover panel for the whole tradeable universe."""
    if PANEL.exists():
        d = pd.read_parquet(PANEL)
        return d.xs("close",1,0), d.xs("turn",1,0)
    closes, turns = {}, {}
    for f in glob.glob(str(DAILY/"*.csv")):
        sym = Path(f).stem.upper()
        if not is_stock(sym): continue
        try:
            d = pd.read_csv(f, usecols=["Date","Close","Volume","Series"])
        except Exception:
            continue
        d = d[d["Series"]=="EQ"]
        if len(d) < 300: continue
        d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
        d = d.dropna(subset=["Date"]).set_index("Date").sort_index()
        d = d[~d.index.duplicated(keep="last")]
        c = pd.to_numeric(d["Close"], errors="coerce")
        v = pd.to_numeric(d["Volume"], errors="coerce")
        closes[sym] = c; turns[sym] = c*v
    C = pd.DataFrame(closes).sort_index()
    T = pd.DataFrame(turns).reindex(C.index)
    C = C.astype("float32"); T = T.astype("float32")
    pd.concat({"close":C,"turn":T}, axis=1).to_parquet(PANEL)
    return C, T


def run_offset(C, T, feats, grid_idx, offset):
    """One tranche: rebalance every STEP days starting at `offset`.

    Returns a DAILY NAV series -- positions are marked to market every day, so
    volatility/Sharpe are real. (Chaining only period-end returns would make the
    NAV a step function and understate vol, which flatters the tranched book.)"""
    m3, m6, m12, vol, malong, turn_sm = feats
    dates = C.index
    pts = list(range(grid_idx + offset, len(dates)-1, STEP))
    nav_parts = []; level = 1.0; prev = []
    for a, b in zip(pts, pts[1:]):
        pr = C.iloc[a]; tn = turn_sm.iloc[a]; v = vol.iloc[a]
        hist = C.iloc[:a+1].notna().sum()
        base = pr.index[(pr>=PRICE_MIN)&(hist>=L12)&tn.notna()&(tn>0)&(v>=VOL_FLOOR)]
        seg_idx = dates[a+1:b+1]
        picks = []
        if len(base) >= 100:
            rank = tn[base].rank(pct=True)
            mid = rank.index[(rank>0.60)&(rank<=0.90)]
            ml = malong.iloc[a]
            elig = [s for s in mid if pr[s] > ml.get(s, np.inf)]
            score = pd.Series(0.0, index=elig); nc = 0
            for sg in (m3, m6, m12):
                x = (sg.iloc[a][elig] / v[elig].replace(0,np.nan)).dropna()
                if len(x): score = score.add(x.rank(pct=True), fill_value=0); nc += 1
            if nc and len(elig):
                ranked = (score/nc).sort_values(ascending=False)
                rk = {s:i for i,s in enumerate(ranked.index)}
                picks = [s for s in prev if rk.get(s,10**9) < TOP_N*1.67]
                for s in ranked.index:
                    if len(picks) >= TOP_N: break
                    if s not in picks: picks.append(s)
                picks = picks[:TOP_N]
        held = [s for s in picks
                if np.isfinite(C.iloc[a].get(s, np.nan)) and C.iloc[a].get(s, 0) > 0]
        churn = len(set(prev)-set(held))/max(len(prev),1) if prev else 1.0
        level *= (1 - churn*RT)                       # costs hit at the rebalance
        if held:
            # equal-weight at entry, weights drift within the period (buy & hold)
            block = C.loc[seg_idx, held]
            basket = (block/C.iloc[a][held]).mean(axis=1)   # NaN-tolerant mean
            basket = basket.ffill().fillna(1.0)
        else:
            basket = pd.Series(1.0, index=seg_idx)
        nav_parts.append(basket*level)
        level = float(basket.iloc[-1]*level)
        prev = held
    return pd.concat(nav_parts) if nav_parts else pd.Series(dtype=float)


def stats(nav):
    nav = nav.dropna()
    if len(nav) < 2: return dict(cagr=0,dd=0,sh=0)
    yrs = (nav.index[-1]-nav.index[0]).days/365.25
    cagr = nav.iloc[-1]**(1/yrs)-1 if yrs>0 else 0
    dd = (nav/nav.cummax()-1).min()
    dr = nav.pct_change().dropna()
    sh = dr.mean()/dr.std()*np.sqrt(252) if dr.std()>0 else 0
    return dict(cagr=cagr*100, dd=dd*100, sh=sh)


def seg(nav, lo, hi):
    s = nav[(nav.index.year>=lo)&(nav.index.year<=hi)]
    return stats(s/s.iloc[0]) if len(s)>1 else dict(cagr=0,dd=0,sh=0)


def main():
    print("Building full daily panel (first run is slow, then cached) ...")
    C, T = build_panel()
    print(f"panel: {C.index.min().date()} -> {C.index.max().date()} | {C.shape[1]} symbols, {len(C)} days")

    turn_sm = T.rolling(21).mean()
    m3  = C/C.shift(L3)-1
    m6  = C/C.shift(L6)-1
    m12 = C/C.shift(L12)-1
    vol = C.pct_change().rolling(126).std()*np.sqrt(21)
    malong = C.rolling(MA_LONG).mean()
    feats = (m3, m6, m12, vol, malong, turn_sm)

    start = int(np.searchsorted(C.index.year.values, FROM_YEAR))
    print(f"start index {start} ({C.index[start].date()})\n")

    K = 3                                   # number of tranches
    offsets = [round(i*STEP/K) for i in range(K)]
    navs = {}
    for o in offsets:
        navs[o] = run_offset(C, T, feats, start, o)
        print(f"  tranche offset {o:2d}d done ({len(navs[o])} daily obs)")

    # align all tranches on a common daily index before averaging
    common = navs[offsets[0]].index
    for o in offsets[1:]: common = common.intersection(navs[o].index)
    navs = {o: (navs[o].reindex(common).ffill()) for o in offsets}
    navs = {o: v/v.iloc[0] for o, v in navs.items()}
    base = navs[offsets[0]]                              # single month-end style book
    tranched = sum(navs.values())/len(navs)              # K equal slices, staggered
    tranched = tranched/tranched.iloc[0]

    print(f"\n{'variant':30} {'CAGR':>8} {'MaxDD':>9} {'Sharpe':>7} | {'IS CAGR':>8} | {'OOS CAGR':>9} {'OOS DD':>8} {'OOS Sh':>7}")
    print("-"*100)
    for name, nav in [("BASELINE single rebalance", base),
                      *[(f"  tranche {o}d alone", navs[o]) for o in offsets],
                      (f"TRANCHED ({K} slices)", tranched)]:
        a = stats(nav); i = seg(nav, 2010, IS_END); o_ = seg(nav, IS_END+1, 2026)
        print(f"{name:30} {a['cagr']:7.2f}% {a['dd']:8.1f}% {a['sh']:7.2f} "
              f"| {i['cagr']:7.2f}% | {o_['cagr']:8.2f}% {o_['dd']:7.1f}% {o_['sh']:7.2f}")

    sp = [stats(navs[o])['cagr'] for o in offsets]
    print(f"\nTIMING LUCK: identical strategy, different start day -> CAGR spread "
          f"{min(sp):.2f}% .. {max(sp):.2f}%  (range {max(sp)-min(sp):.2f}pp)")
    pd.DataFrame({**{f"tranche_{o}":navs[o] for o in offsets}, "tranched":tranched}).to_csv(
        BASE/"data_store/tranching_results.csv")
    print("saved -> data_store/tranching_results.csv")


if __name__ == "__main__":
    main()
