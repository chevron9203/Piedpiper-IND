"""
Audit of the chosen smart system before anyone trades it.

System ("best"): rank-average of the h10/h21/h63 walk-forward LightGBM models (events +
peers), 75%, plus risk-adjusted momentum 25%; top 20, exit rank 100, weekly; per-stock
costs (fees + spread + impact) at the given capital.

  audit    leak/plumbing checks, biggest position-days (data errors?), concentration,
           dead names, extra-day execution delay, and -- once -- the 2023+ holdout
  window   Rs P&L of the running system between two dates, stock by stock

Run:  python scripts/research_smart_audit.py audit
      python scripts/research_smart_audit.py window --start 2026-09-10
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))
from scripts import research_smart_ml as SM
from scripts import research_smart_iter as IT

ML_TAGS = ("h10_s5_ev_pr_ni_px_x3", "h21_s5_ev_pr_ni_px_x3", "h63_s5_ev_pr_ni_px_x3")   # v1.1: traded-price fix, no insider
MOM_W = 0.50; TOP = 20; EXIT = 100; STEP = 5


def best_score(Xr, tags=None):
    tags = ML_TAGS if tags is None else tags     # resolved at CALL time (a default arg froze stale tags once)
    comp = [IT.xs(pd.read_parquet(BASE/f"data_store/smart_ml_pred_{t}.parquet")["pred"]) for t in tags]
    ml = sum(comp)/len(comp)
    mom = IT.xs(Xr["mom_riskadj"]).reindex(ml.index)
    return (1 - MOM_W)*ml + MOM_W*mom


def setup(capital):
    P = SM.load_panel()
    X, C, univ, mkt = SM.compute_features(P, STEP)
    Xr = SM.rank_features(X)
    cm = SM.cost_model(P, capital, TOP)
    return P, X, Xr, C, univ, mkt, cm


def run(sc, C, univ, cm):
    log = []
    nav, to, nt = SM.simulate(sc, C, univ, TOP, EXIT, STEP, cost=cm, log=log)
    L = pd.DataFrame(log, columns=["kind", "date", "sym", "w", "x"])
    return nav, to, nt, L


def line(name, nav, to=None):
    IT.SHOW_HOLDOUT = True
    st = IT.split_stats(nav)
    return f"  {name:<34} {IT.fmt(st)}" + (f"  TO {to:.0%}" if to is not None else "")


def cmd_audit(a):
    P, X, Xr, C, univ, mkt, cm = setup(a.capital)
    sc = best_score(Xr)
    nav, to, nt, L = run(sc, C, univ, cm)
    H = L[L.kind == "hold"].rename(columns={"x": "r"})
    T = L[L.kind == "trade"].rename(columns={"x": "cost"})

    print("== 1. headline: dev 2013-22 vs HOLDOUT 2023+ (first look) ==")
    print(line("smart system", nav, to))
    ew = (1 + mkt.loc[nav.index[0]:].fillna(0)).cumprod()
    print(line("equal-wt universe", ew))
    yr = nav.resample("YE").last().pct_change(); yr.iloc[0] = nav.resample("YE").last().iloc[0] - 1
    ey = ew.resample("YE").last().pct_change(); ey.iloc[0] = ew.resample("YE").last().iloc[0] - 1
    print("  by year:", "  ".join(f"{d.year} {v:+.0%}/{e:+.0%}" for d, v, e in zip(yr.index, yr, ey)))

    print("\n== 2. leak checks ==")
    shuf = BASE/"data_store/smart_ml_pred_h21_s5_ev_pr_shuf.parquet"
    if shuf.exists():
        s2 = IT.xs(pd.read_parquet(shuf)["pred"])
        n2, t2, _, _ = run(s2, C, univ, cm)
        print(line("shuffled-label model (must ~= EW-costs)", n2, t2))
    d = sc.index.get_level_values("date")
    nxt = pd.Series(C.index[1:], index=C.index[:-1])
    late = sc[d < C.index[-1]].copy()
    late.index = pd.MultiIndex.from_arrays([nxt.reindex(late.index.get_level_values("date")).values,
                                            late.index.get_level_values("sym")], names=["date", "sym"])
    n3, t3, _, _ = run(late, C, univ, cm)
    print(line("execute 1 day later (t+2 close)", n3, t3))

    print("\n== 3. data sanity: largest single position-days ==")
    H["contrib"] = H["w"]*H["r"]
    big = pd.concat([H.nlargest(12, "r"), H.nsmallest(8, "r")])
    Cv = P["close"]
    for _, row in big.iterrows():
        s, dt = row.sym, row.date
        i = C.index.get_loc(dt)
        px = Cv[s].iloc[max(i - 2, 0): i + 3].round(2).tolist()
        vol = P["vol"][s]
        vr = vol.iloc[i]/vol.iloc[max(i - 60, 0): i].median()
        print(f"  {dt.date()} {s:<12} r {row.r:+7.1%}  w {row.w:.3f}  closes t-2..t+2 {px}  vol x{vr:.1f}")
    print(f"  position-days with |r| > 20%: {(H.r.abs() > 0.2).sum()} of {len(H):,}")

    print("\n== 4. concentration: is it a few lucky names? ==")
    g = H.assign(lr=np.log1p(H["contrib"])).groupby("sym")["lr"].sum().sort_values()
    tot = np.log(nav.iloc[-1])
    print(f"  total log-return {tot:.2f}; top-10 names {g.tail(10).sum():.2f}, top-30 {g.tail(30).sum():.2f}")
    print("  top 10:", ", ".join(f"{k} {v:+.2f}" for k, v in g.tail(10)[::-1].items()))
    print("  bottom 5:", ", ".join(f"{k} {v:+.2f}" for k, v in g.head(5).items()))
    print(f"  distinct names held {H.sym.nunique()}, trades {len(T):,}, cost paid {T['cost'].sum():.2f} (sum of wt*cost)")
    keep = set(g.index[:-10])
    n0 = (1 + H.groupby("date")["contrib"].sum()).cumprod()
    n4 = (1 + H[H.sym.isin(keep)].groupby("date")["contrib"].sum()).reindex(n0.index).fillna(1).cumprod()
    print(line("all names, before costs", n0))
    print(line("without the 10 best names (as cash)", n4))

    print("\n== 5. dead names held ==")
    dead = H[(H.r == -SM.DEAD_HAIRCUT)]
    print(f"  delisting haircuts booked: {len(dead)}  {', '.join(f'{r.sym} {r.date.date()}' for r in dead.itertuples())}")
    nav.to_frame("nav").to_parquet(BASE/"data_store/smart_best_nav.parquet")


def cmd_window(a):
    P, X, Xr, C, univ, mkt, cm = setup(a.capital)
    sc = best_score(Xr)
    nav, to, nt, L = run(sc, C, univ, cm)
    s, e = pd.Timestamp(a.start), C.index[-1]
    prev = nav[nav.index < s].index[-1]
    ret = nav.loc[e]/nav.loc[prev] - 1
    print(f"running system, {s.date()} -> {e.date()} (from close {prev.date()}): {ret:+.2%}  "
          f"= Rs {ret*a.capital:+,.0f} on Rs {a.capital:,.0f}")
    ix = pd.read_parquet(BASE/"data_store/nse_index.parquet")
    for nm in ("Nifty 50", "Nifty Midcap 150", "Nifty Smallcap 250", "Nifty 500"):
        q = ix[ix["index"].str.lower() == nm.lower()].set_index("date")["close"].sort_index()
        if prev in q.index and e in q.index:
            print(f"  {nm:<20} {q.loc[e]/q.loc[prev] - 1:+.2%}")
    ew = (1 + mkt.loc[s:e].fillna(0)).prod() - 1
    print(f"  {'equal-wt universe':<20} {ew:+.2%}")
    H = L[(L.kind == "hold") & (L.date >= s) & (L.date <= e)].rename(columns={"x": "r"})
    navd = nav.shift(1).reindex(H.date.values).values
    H["rs"] = H["w"].values*H["r"].values*navd/nav.loc[prev]*a.capital
    T = L[(L.kind == "trade") & (L.date >= s) & (L.date <= e)].rename(columns={"x": "cost"})
    T["rs"] = -T["cost"].values*nav.reindex(T.date.values).values/nav.loc[prev]*a.capital
    by = H.groupby("sym")["rs"].sum().add(T.groupby("sym")["rs"].sum(), fill_value=0).sort_values(ascending=False)
    print(f"\nRs P&L by stock (incl. its trading costs), {len(by)} names:")
    print("  " + "\n  ".join(f"{k:<12} {v:+9,.0f}" for k, v in by.items()))
    print(f"  {'TOTAL':<12} {by.sum():+9,.0f}")
    print("\ntrades in window:")
    for r in T.itertuples():
        print(f"  {r.date.date()} {'BUY ' if r.w > 0 else 'SELL'} {r.sym:<12} {abs(r.w):.1%} of book")
    last = L[(L.kind == "hold") & (L.date == e)]
    print(f"\nbook on {e.date()}: {', '.join(last.sym)}")
    print("\ndaily NAV (Rs):\n" + (nav.loc[prev:e]/nav.loc[prev]*a.capital).round(0).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["audit", "window"])
    ap.add_argument("--capital", type=float, default=2e5)
    ap.add_argument("--start", default="2026-09-10")
    a = ap.parse_args()
    {"audit": cmd_audit, "window": cmd_window}[a.cmd](a)
