"""Market falls vs rebounds (zigzag legs of the equal-weight market, >=12%): how ML-only, momentum and the 50/50 blend
behave in each, plus the perfect-hindsight switching ceiling. Diagnostic only: legs are defined with hindsight.
Run: python scripts/research_smart_legs.py"""
import sys, pickle; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from scripts import research_smart_exit as E, research_smart_verify as V
G = E.inputs(); C, mkt, cm = G["C"], G["mkt"], G["cm"]
pct = lambda s: s.groupby(level="date").rank(pct=True)
# ---- 2010-2012 books (5-day grid, ext models) ; 2013+ books from the 5-alignment averaged results
F = pd.read_parquet("data_store/smart_features_s5.parquet", columns=["mom_riskadj"])["mom_riskadj"]
ml = [pct(pd.read_parquet(f"data_store/smart_ml_pred_h{h}_s5_ev_pr_ni_x3_px_tf2010-2012.parquet")["pred"]) for h in (10, 21, 63)]
ix = ml[0].index; mlx = (ml[0] + ml[1].reindex(ix) + ml[2].reindex(ix))/3; mom = pct(F).reindex(ix)
ext = {}
for lab, w in (("smart", 0.5), ("ML only", 0.0), ("momentum", 1.0)):
    out = E.simulate_rules(((1 - w)*mlx + w*mom).dropna(), C, mkt, cm, end_date="2012-12-31", **dict(E.FINAL, every=1, offset=0))
    ext[lab] = out["nav"][:"2012-12-31"]
S = pickle.load(open("data_store/smart_split_results.pkl", "rb"))["nav"]
nav = {}
for lab in ("smart", "ML only", "momentum"):
    a, b = ext[lab], S[lab].dropna()
    nav[lab] = pd.concat([a, b*float(a.iloc[-1])]).sort_index()
mc = (1 + mkt.fillna(0)).cumprod().loc["2010-01-07":]
n500 = V.bench("nifty500")[0]
# ---- zigzag legs of the equal-weight market (>= 12% moves)
thr = np.log(1/0.88); lg = np.log(mc.values); piv = [0]; direction = 0; ext_i = 0
for i in range(1, len(lg)):
    if direction >= 0:
        if lg[i] > lg[ext_i]: ext_i = i
        if lg[ext_i] - lg[i] >= thr and (direction == 0 or True):
            if piv[-1] != ext_i: piv.append(ext_i)
            direction = -1; ext_i = i
    else:
        if lg[i] < lg[ext_i]: ext_i = i
        if lg[i] - lg[ext_i] >= thr:
            piv.append(ext_i); direction = 1; ext_i = i
piv.append(len(lg) - 1)
piv = sorted(set(piv)); dates = mc.index[piv]
def val(s, d): return float(s.dropna().loc[:d].iloc[-1])
def leg(s, a, b): return val(s, b)/val(s, a) - 1
rows = []
for a, b in zip(dates[:-1], dates[1:]):
    r = {k: leg(nav[k], a, b) for k in nav}; r["market"] = leg(mc, a, b)
    r["Nifty 500"] = leg(n500, a, b) if a >= pd.Timestamp("2012-03-01") else np.nan
    rows.append(dict(a=a, b=b, days=(b - a).days, **r))
L = pd.DataFrame(rows); L["down"] = L.market < 0
fmt = lambda x: f"{x*100:+7.1f}%" if np.isfinite(x) else "      n/a"
def show(df, title):
    print(f"\n{title}\n  {'from':<11}{'to':<11}{'days':>5}  {'equal-wt mkt':>12}{'Nifty 500':>11}{'ML only':>10}{'momentum':>10}{'smart 50/50':>12}   ML-only minus momentum")
    for r in df.itertuples():
        print(f"  {r.a.date()!s:<11}{r.b.date()!s:<11}{r.days:>5}  {fmt(r.market):>12}{fmt(getattr(r, '_8')) if False else fmt(r[8]):>11}{fmt(r[5]):>10}{fmt(r[6]):>10}{fmt(r[4]):>12}   {(r[5] - r[6])*100:+6.1f}pp")
Ld = L[L.down]; Lu = L[~L.down]
show(Ld, f"MARKET FALLS (peak -> trough, >= 12%): {len(Ld)} episodes")
print(f"\n  in the falls: ML-only beat momentum in {(Ld['ML only'] > Ld['momentum']).sum()} of {len(Ld)}; avg ML-only {Ld['ML only'].mean()*100:+.1f}%  momentum {Ld['momentum'].mean()*100:+.1f}%  smart {Ld['smart'].mean()*100:+.1f}%  market {Ld['market'].mean()*100:+.1f}%")
show(Lu, f"MARKET REBOUNDS (trough -> peak): {len(Lu)} episodes")
print(f"\n  in the rebounds: ML-only beat momentum in {(Lu['ML only'] > Lu['momentum']).sum()} of {len(Lu)}; avg ML-only {Lu['ML only'].mean()*100:+.1f}%  momentum {Lu['momentum'].mean()*100:+.1f}%  smart {Lu['smart'].mean()*100:+.1f}%  market {Lu['market'].mean()*100:+.1f}%")
# ---- ceiling: what if we could switch PERFECTLY (knowing each leg's direction in advance)
yrs = (dates[-1] - dates[0]).days/365.25
def chain(pick):
    g = 1.0
    for r in L.itertuples(): g *= 1 + pick(r)
    return g, g**(1/yrs) - 1
print(f"\nCEILING (perfect hindsight switching at every peak/trough; ignores trading costs), {dates[0].date()} -> {dates[-1].date()}, {yrs:.1f} yrs:")
for lab, f in (("always smart 50/50", lambda r: r.smart), ("always ML only", lambda r: r._6 if False else getattr(r, "_6", 0)),
               ):
    pass
cols = {c: i for i, c in enumerate(L.columns)}
def pick(down_col, up_col):
    return lambda r: (r[cols[down_col] + 1] if r[cols["down"] + 1] else r[cols[up_col] + 1])
for lab, d_, u_ in (("always smart 50/50", "smart", "smart"), ("always ML only", "ML only", "ML only"), ("always momentum", "momentum", "momentum"),
                    ("PERFECT: ML-only in falls, smart in rebounds", "ML only", "smart"),
                    ("PERFECT: ML-only in falls, momentum in rebounds", "ML only", "momentum")):
    g, c = chain(pick(d_, u_)); print(f"  {lab:<50} total x{g:6.1f}   CAGR {c*100:+6.1f}%")
