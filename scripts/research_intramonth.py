"""
Does an INTRA-MONTH exit rule beat the plain monthly rebalance?

The problem being tested: the book picks 15 mid-cap momentum names at month-end
and holds them blind until the next month-end. When a name's momentum breaks in
week 2, we give the gain back. This measures whether reacting to that costs or
earns money -- on our own data, IS 2010-18 / OOS 2019-26, honest costs.

Picks come from the EXACT validated monthly logic (scripts/validate_pure_momentum.py);
only the exit is varied, so any difference is attributable to the exit rule alone.

Run:  python scripts/research_intramonth.py
"""
from __future__ import annotations
import sys, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np, pandas as pd, duckdb

BASE = Path(__file__).parent.parent
CACHE = BASE/"data_store/mom_v2_panels.parquet"
DB = BASE/"data_store/piedpiper.duckdb"
DAILY = BASE/"data_store/eod2/src/eod2_data/daily"
DAILY_CACHE = BASE/"data_store/intramonth_daily.parquet"

CAP = 200_000.0; PRICE_MIN = 30.0; FROM_YEAR = 2010; IS_END = 2018
TOP_N = 15; TURN_TOP = 500; VOL_FLOOR = 0.02
RT = 2*(0.0010+0.0010)+0.0005          # round-trip cost, as validated
ONE_WAY = RT/2                          # charged on an intra-month stop-out
CASH_MO = (1.065)**(1/12)-1
CASH_DAY = (1.065)**(1/252)-1


# ---------------------------------------------------------------- picking
def is_stock(sym):
    s = str(sym).upper()
    return not (s.endswith("BEES") or s.endswith("ETF") or s.endswith("IETF")
                or any(p in s for p in ("LIQUID","GILT","GSEC","BHARATBOND","CASHIETF")))


def pick_stocks(close, turn, p, prev, m3, m6, m12, vol, ma10):
    """Verbatim from the deployed live signal: MID-cap tier, multi-TF risk-adj
    momentum, trend confirm, top-15 with rank buffer."""
    pr = close.iloc[p]; tn = turn.iloc[p]
    hist = close.iloc[:p+1].notna().sum(); v = vol.iloc[p]
    base = pr.index[(pr>=PRICE_MIN)&(hist>=14)&tn.notna()&(tn>0)&(v>=VOL_FLOOR)]
    base = [s for s in base if is_stock(s)]
    if len(base) < 100: return []
    rank = tn[base].rank(pct=True)
    mid = list(rank.index[(rank>0.60)&(rank<=0.90)])
    elig = [s for s in mid if pr[s] > ma10.iloc[p].get(s, 1e9)]
    score = pd.Series(0.0, index=elig); nc = 0
    for sg in (m3, m6, m12):
        x = (sg.iloc[p][elig] / vol.iloc[p][elig].replace(0,np.nan)).dropna()
        if len(x): score = score.add(x.rank(pct=True), fill_value=0); nc += 1
    if nc == 0: return []
    ranked = (score/nc).sort_values(ascending=False)
    rk = {s:i for i,s in enumerate(ranked.index)}
    picks = [s for s in prev if rk.get(s, 10**9) < TOP_N*1.67]
    for s in ranked.index:
        if len(picks) >= TOP_N: break
        if s not in picks: picks.append(s)
    return picks[:TOP_N], list(ranked.index)


def build_picks():
    """Month-by-month picks from the validated monthly panel."""
    pk = pd.read_parquet(CACHE)
    close = pk.xs("close",1,0); turn = pk.xs("turn",1,0)
    ret = close.pct_change(); vol = ret.rolling(6).std()
    m3 = close.shift(1)/close.shift(4)-1
    m6 = close.shift(1)/close.shift(7)-1
    m12 = close.shift(1)/close.shift(13)-1
    ma10 = close.rolling(10).mean()
    idx = close.index
    start = max(14, int(np.searchsorted(idx.year.values, FROM_YEAR)))
    out = {}; prev = []
    for p in range(start, len(idx)-1):
        res = pick_stocks(close, turn, p, prev, m3, m6, m12, vol, ma10)
        if not res: continue
        picks, ranked = res
        out[idx[p]] = (picks, ranked)
        prev = picks
    return out, close


# ---------------------------------------------------------------- daily data
def load_daily(symbols):
    """Daily close/high panel for just the symbols the strategy ever holds."""
    if DAILY_CACHE.exists():
        d = pd.read_parquet(DAILY_CACHE)
        if set(symbols) <= set(d.columns.levels[1]):
            return d.xs("close",1,0), d.xs("high",1,0)
    closes, highs = {}, {}
    for s in symbols:
        f = DAILY/f"{s.lower()}.csv"
        if not f.exists(): continue
        try:
            d = pd.read_csv(f, usecols=["Date","Close","High","Series"])
        except Exception:
            continue
        d = d[d["Series"]=="EQ"]
        if not len(d): continue
        d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
        d = d.dropna(subset=["Date"]).set_index("Date").sort_index()
        d = d[~d.index.duplicated(keep="last")]
        closes[s] = pd.to_numeric(d["Close"], errors="coerce")
        highs[s]  = pd.to_numeric(d["High"],  errors="coerce")
    C = pd.DataFrame(closes).sort_index(); H = pd.DataFrame(highs).reindex(C.index)
    pd.concat({"close":C,"high":H}, axis=1).to_parquet(DAILY_CACHE)
    return C, H


# ---------------------------------------------------------------- simulation
def simulate(picks_by_month, C, H, rule, redeploy=False):
    """Hold each month's picks; apply `rule` daily. Returns a monthly return series.

    rule: None | ("trail", pct) | ("chand", k) | ("ma", n) | ("hard", pct)
    redeploy=False -> stopped-out cash sits in cash until month end (no look-ahead).
    """
    months = sorted(picks_by_month)
    atr = None
    if rule and rule[0] == "chand":
        tr = (H - C.shift(1)).abs().combine((H-C).abs(), np.maximum)
        atr = tr.rolling(20).mean()
    ma = C.rolling(rule[1]).mean() if rule and rule[0] == "ma" else None

    out = {}
    for i in range(len(months)-1):
        m0, m1 = months[i], months[i+1]
        picks = picks_by_month[m0][0]
        if not picks: out[m1] = CASH_MO; continue
        win = C.loc[(C.index > m0) & (C.index <= m1)]
        if len(win) < 2: out[m1] = CASH_MO; continue

        leg = []
        for s in picks:
            if s not in C.columns: continue
            px = win[s].dropna()
            if len(px) < 2: continue
            entry = px.iloc[0]
            if not np.isfinite(entry) or entry <= 0: continue
            exit_i = None
            if rule:
                run_max = px.cummax()
                if rule[0] == "trail":
                    hit = px <= run_max*(1-rule[1])
                elif rule[0] == "hard":
                    hit = px <= entry*(1-rule[1])
                elif rule[0] == "ma":
                    m = ma[s].reindex(px.index)
                    hit = px < m
                elif rule[0] == "chand":
                    a = atr[s].reindex(px.index)
                    hit = px <= (run_max - rule[1]*a)
                hit = hit.fillna(False)
                if hit.any(): exit_i = hit.idxmax()
            if exit_i is not None:
                r = px.loc[exit_i]/entry - 1 - ONE_WAY      # extra cost of the stop
                if redeploy:                                 # park in cash to month end
                    days = (px.index > exit_i).sum()
                    r = (1+r)*(1+CASH_DAY)**days - 1
            else:
                r = px.iloc[-1]/entry - 1
            leg.append(r)
        gross = float(np.mean(leg)) if leg else CASH_MO
        # monthly turnover cost: fraction of book actually replaced
        nxt = picks_by_month[m1][0] if m1 in picks_by_month else []
        churn = 1.0 if not nxt else len(set(picks)-set(nxt))/max(len(picks),1)
        out[m1] = gross - churn*RT
    return pd.Series(out).sort_index()


# ---------------------------------------------------------------- metrics
def stats(r):
    if not len(r): return dict(cagr=0,dd=0,sh=0,n=0)
    nav = (1+r).cumprod()
    yrs = len(r)/12
    cagr = nav.iloc[-1]**(1/yrs)-1 if yrs>0 else 0
    dd = (nav/nav.cummax()-1).min()
    sh = r.mean()/r.std()*np.sqrt(12) if r.std()>0 else 0
    return dict(cagr=cagr*100, dd=dd*100, sh=sh, n=len(r))


def report(name, r):
    a = stats(r)
    is_ = stats(r[r.index.year <= IS_END]); oos = stats(r[r.index.year > IS_END])
    print(f"{name:28} {a['cagr']:7.2f}% {a['dd']:8.1f}% {a['sh']:6.2f} "
          f"| {is_['cagr']:7.2f}% {is_['dd']:7.1f}% "
          f"| {oos['cagr']:7.2f}% {oos['dd']:7.1f}% {oos['sh']:6.2f}")
    return a, is_, oos


def main():
    print("Building monthly picks from validated logic ...")
    picks_by_month, _ = build_picks()
    universe = sorted({s for v in picks_by_month.values() for s in v[0]})
    print(f"months: {len(picks_by_month)} | unique names ever held: {len(universe)}")
    print("Loading daily panel ...")
    C, H = load_daily(universe)
    print(f"daily panel: {C.index.min().date()} -> {C.index.max().date()}  ({C.shape[1]} symbols)\n")

    rules = [
        ("BASELINE monthly (no stop)", None, False),
        ("trail 10%",  ("trail",0.10), True),
        ("trail 15%",  ("trail",0.15), True),
        ("trail 20%",  ("trail",0.20), True),
        ("trail 25%",  ("trail",0.25), True),
        ("hard stop 15%", ("hard",0.15), True),
        ("hard stop 20%", ("hard",0.20), True),
        ("chandelier 3xATR", ("chand",3.0), True),
        ("chandelier 4xATR", ("chand",4.0), True),
        ("close < 20d MA", ("ma",20), True),
        ("close < 50d MA", ("ma",50), True),
    ]
    print(f"{'rule':28} {'CAGR':>8} {'MaxDD':>9} {'Sh':>6} "
          f"| {'IS CAGR':>8} {'IS DD':>8} | {'OOS CAGR':>8} {'OOS DD':>8} {'OOS Sh':>6}")
    print("-"*108)
    res = {}
    for name, rule, redep in rules:
        r = simulate(picks_by_month, C, H, rule, redep)
        res[name] = r
        report(name, r)
    pd.DataFrame(res).to_csv(BASE/"data_store/intramonth_results.csv")
    print(f"\nsaved -> data_store/intramonth_results.csv")


if __name__ == "__main__":
    main()
