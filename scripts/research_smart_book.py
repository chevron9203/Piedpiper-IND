"""
Portfolio construction for the smart system: same scores, better book.

Diagnosis (2026-10-03) that motivates each knob:
  * ranks 21-50 earn about as much as the top 20 -> the very top is noisy: try a BROADER book
  * 76% of trades are tiny re-trims to equal weight -> tolerance BAND, skip them
  * Sept 2026 book was 8/20 banks -> CORRELATION CAP: skip a candidate that moves with
    names already picked (market-residual returns, point-in-time)
  * equal weight gives a 6%-vol stock the same risk as a 2%-vol one -> INVERSE-VOL sizing

All judged on DEV (2013-2022) only; the 2023+ holdout has been looked at once already.

Run:  python scripts/research_smart_book.py grid
"""
from __future__ import annotations
import argparse, itertools, sys
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))
from scripts import research_smart_ml as SM
from scripts import research_smart_iter as IT
from scripts import research_smart_audit as A


def decide(s, w, Rres_win, top_n=20, exit_rank=100, sizing="eq", band=0.0, max_corr=None,
           max_w=0.10, vol_row=None, bad=frozenset()):
    """One rebalance: today's scores `s` (sorted desc), current drifted weights `w`, the
    trailing market-residual returns `Rres_win` (corr window, rows up to today).
    Returns target weights. Shared by the backtest and the live service."""
    rank = pd.Series(np.arange(1, len(s) + 1), index=s.index)
    keep = [x for x in w.index if x in rank.index and rank[x] <= exit_rank and x not in bad]
    need = top_n - len(keep)
    new = []
    if need > 0:
        cands = [x for x in s.index[: top_n*6] if x not in keep and x not in bad]
        if max_corr is None:
            new = cands[:need]
        else:
            pool = keep + cands
            M = Rres_win[pool].corr(min_periods=60)
            chosen = list(keep)
            for x in cands:
                if len(new) >= need: break
                if chosen and M.loc[x, chosen].max() > max_corr:
                    continue
                new.append(x); chosen.append(x)
            if len(new) < need:                    # never leave slots empty
                new += [x for x in cands if x not in new][: need - len(new)]
    names = keep + new
    if sizing == "ivol":
        v = vol_row.reindex(names)
        raw = (1/v.clip(lower=0.005)).fillna((1/v).median() if v.notna().any() else 1.0)
        target = (raw/raw.sum()).clip(upper=max_w)
        target /= target.sum()
    else:
        target = pd.Series(1.0/len(names), index=names)
    if band > 0:                                   # leave kept names alone inside the band
        for x in keep:
            if abs(target[x] - w[x]) <= band*target[x]:
                target[x] = w[x]
        fresh = [x for x in names if x not in keep or abs(target[x] - w.get(x, 0)) > band*target[x]]
        # renormalise only names being traded so the book stays fully invested
        fixed = [x for x in names if x not in fresh]
        if fresh:
            t2 = target[fresh]; t2 = t2/t2.sum()*(1 - target[fixed].sum())
            target[fresh] = t2
    return target


def simulate_book(score, C, mkt, cost, top_n=20, exit_rank=100, sizing="eq", band=0.0,
                  max_corr=None, corr_win=126, max_w=0.10, log=None, veto=None):
    """Decide at t, trade at t+1 close. Returns daily NAV and turnover/yr.
    veto: optional {date: set(sym)} -- red-flag names are never bought and are sold."""
    Cf = C.ffill(limit=5)
    R = Cf.pct_change(fill_method=None)
    Rres = R.sub(mkt, axis=0)
    vol = R.rolling(63, min_periods=40).std()
    last_valid = C.apply(lambda s: s.last_valid_index())
    idx = C.index
    sdates = [d for d in sorted(set(score.index.get_level_values("date"))) if idx.get_loc(d) + 1 < len(idx)]
    by_date = {d: g.droplevel(0).sort_values(ascending=False) for d, g in score.groupby(level="date")}
    Rv = R.values; colpos = {c: j for j, c in enumerate(C.columns)}
    nav = [1.0]; navd = [idx[idx.get_loc(sdates[0]) + 1]]
    w = pd.Series(dtype=float); turnover = 0.0
    for k, d in enumerate(sdates):
        i_d = idx.get_loc(d)
        target = decide(by_date[d], w, Rres.iloc[max(0, i_d - corr_win + 1): i_d + 1], top_n, exit_rank,
                        sizing, band, max_corr, max_w, vol.iloc[i_d],
                        veto.get(d, set()) if veto is not None else frozenset())
        i0 = i_d + 1
        dwi = target.sub(w, fill_value=0).abs()
        ci = cost.iloc[i0].reindex(dwi.index).fillna(0.01)
        c = (dwi*ci).sum()
        if log is not None:
            for x, dx in target.sub(w, fill_value=0).items():
                if abs(dx) > 1e-9:
                    log.append(("trade", idx[i0], x, dx, abs(dx)*ci[x]))
        turnover += dwi.sum()/2
        w = target
        i1 = idx.get_loc(sdates[k + 1]) + 1 if k + 1 < len(sdates) else len(idx) - 1
        v = nav[-1]*(1 - c)
        cols = [colpos[x] for x in w.index]
        for i in range(i0 + 1, i1 + 1):
            r = pd.Series(Rv[i, cols], index=w.index)
            died = [x for x in w.index if np.isnan(r[x]) and last_valid[x] < idx[i]
                    and last_valid[x] < idx[-1] - pd.Timedelta(days=30)]
            r = r.fillna(0.0)
            for x in died:
                r[x] = -SM.DEAD_HAIRCUT
            pr = float((w*r).sum())
            if log is not None:
                for x in w.index:
                    log.append(("hold", idx[i], x, w[x], r[x]))
            v *= 1 + pr
            w = w*(1 + r)/(1 + pr)
            if died:
                w = w.drop(died); cols = [colpos[x] for x in w.index]
            nav.append(v); navd.append(idx[i])
    nav = pd.Series(nav, index=navd)
    return nav, turnover/((nav.index[-1] - nav.index[0]).days/365.25)


def cmd_grid(a):
    P, X, Xr, C, univ, mkt, cm = A.setup(a.capital)
    sc = A.best_score(Xr)
    sc = sc[sc.index.get_level_values("date") <= IT.DEV_END]        # DEV only, by construction
    IT.SHOW_HOLDOUT = False
    runs = [dict()]                                                  # baseline
    runs += [dict(top_n=n, exit_rank=e) for n, e in ((30, 120), (30, 150), (40, 160), (40, 200))]
    runs += [dict(band=b) for b in (0.25, 0.5)]
    runs += [dict(sizing="ivol")]
    runs += [dict(max_corr=c) for c in (0.5, 0.35)]
    for kw in runs:
        n = kw.get("top_n", 20)
        cmx = cm if n == 20 else SM.cost_model(P, a.capital, n)
        nav, to = simulate_book(sc, C, mkt, cmx, **kw)
        lab = ", ".join(f"{k}={v}" for k, v in kw.items()) or "baseline top20/exit100/eq"
        print(f"  {lab:<34} {IT.fmt(IT.split_stats(nav))}  {IT.mfmt(nav)}  TO {to:4.0%}", flush=True)


FINAL = dict(band=0.25, max_corr=0.5)          # chosen on DEV (iteration 4)


def cmd_final(a):
    """Final numbers of the chosen system, and a fresh book started on --start."""
    P, X, Xr, C, univ, mkt, cm = A.setup(a.capital)
    sc = A.best_score(Xr)
    mom = Xr["mom_riskadj"].reindex(sc.index)
    IT.SHOW_HOLDOUT = True
    ix = pd.read_parquet(BASE/"data_store/nse_index.parquet")
    n500 = ix[ix["index"].str.lower() == "nifty 500"].set_index("date")["close"].sort_index()
    nav, to = simulate_book(sc, C, mkt, cm, **FINAL)
    mnav, mto = simulate_book(mom, C, mkt, cm, **FINAL)
    ew = (1 + mkt.loc[nav.index[0]:].fillna(0)).cumprod()
    n5 = n500.reindex(nav.index).ffill().dropna(); n5 = n5/n5.iloc[0]
    print(f"== final system: {1 - A.MOM_W:.0%} ML(h10+h21+h63, events+peers) + {A.MOM_W:.0%} momentum, top {20}, weekly, "
          f"band 0.25, corr cap 0.5, Rs{a.capital/1e5:.0f}L per-stock costs ==")
    print(f"{'':28}  dev 2013-22 CAGR/DD/Sh    holdout 2023-26 CAGR/DD/Sh")
    for name, n in (("SMART SYSTEM", nav), ("momentum (same rules)", mnav), ("equal-wt universe", ew), ("Nifty 500 (price)", n5)):
        print(f"  {name:<26} {IT.fmt(IT.split_stats(n))}")
    full = SM.stats(nav); print(f"\n  full 2013-2026: CAGR {full[0]:+.1%}  MaxDD {full[1]:.1%}  Sharpe {full[2]:.2f}  "
                             f"turnover {to:.0%}/yr  Rs1 -> Rs{nav.iloc[-1]:.0f}")
    m = nav.resample("ME").last().pct_change().dropna()
    print(f"  months up {(m > 0).mean():.0%}, avg month {m.mean():+.2%}, best {m.max():+.1%}, worst {m.min():+.1%}")
    df = pd.DataFrame({"smart": nav, "mom": mnav, "n500": n5}).ffill()
    yr = df.resample("YE").last().pct_change(); yr.iloc[0] = df.resample("YE").last().iloc[0]/df.iloc[0] - 1
    print("\n  by year (%):\n" + (yr*100).round(1).rename(index=lambda d: d.year).to_string())

    # ---- fresh book started on --start
    st = pd.Timestamp(a.start)
    dts = sc.index.get_level_values("date").unique()
    d0 = dts[dts < st].max()
    log = []
    fnav, _ = simulate_book(sc[sc.index.get_level_values("date") >= d0], C, mkt, cm, log=log, **FINAL)
    L = pd.DataFrame(log, columns=["kind", "date", "sym", "w", "x"])
    t0, e = fnav.index[0], fnav.index[-1]
    print(f"\n== fresh Rs{a.capital:,.0f} book: decided {d0.date()}, bought at close {t0.date()}, valued {e.date()} ==")
    print(f"  portfolio {fnav.iloc[-1] - 1:+.2%}  = Rs {a.capital*(fnav.iloc[-1] - 1):+,.0f}  (after all costs)")
    for nm in ("Nifty 50", "Nifty Midcap 150", "Nifty Smallcap 250", "Nifty 500"):
        q = ix[ix["index"].str.lower() == nm.lower()].set_index("date")["close"].sort_index()
        if t0 in q.index and e in q.index:
            print(f"  {nm:<20} {q.loc[e]/q.loc[t0] - 1:+.2%}")
    H = L[L.kind == "hold"].copy(); T = L[L.kind == "trade"].copy()
    prevnav = fnav.shift(1).reindex(H.date.values).values
    H["rs"] = H["w"].values*H["x"].values*prevnav*a.capital
    T["rs"] = -T["x"].values*fnav.reindex(T.date.values).values*a.capital
    pnl = H.groupby("sym")["rs"].sum().add(T.groupby("sym")["rs"].sum(), fill_value=0)
    first_buy = T[T.w > 0].groupby("sym")["date"].min()
    last_sell = T[T.w < 0].groupby("sym")["date"].max()
    held_end = set(H[H.date == e].sym)
    rows = []
    for x in pnl.index:
        b = first_buy.get(x); sdt = last_sell.get(x) if x not in held_end else None
        rows.append((x, b.date() if b is not None else "", sdt.date() if sdt is not None else "holding", pnl[x]))
    R = pd.DataFrame(rows, columns=["stock", "bought", "sold", "pnl"]).sort_values("pnl", ascending=False)
    print("\n  stock         bought      sold         P&L (Rs, incl. costs)")
    for r in R.itertuples():
        print(f"  {r.stock:<12}  {str(r.bought):<10}  {str(r.sold):<10}  {r.pnl:+8,.0f}")
    print(f"  {'TOTAL':<12}  {'':<10}  {'':<10}  {R.pnl.sum():+8,.0f}")
    print(f"\n  trading costs paid: Rs {-T.rs.sum():,.0f} on {len(T[T.w.abs() > 0.02])} buys/sells")
    print("\n  value path (Rs):  " + "  ".join(f"{d:%d-%b} {v*a.capital:,.0f}" for d, v in fnav.items()))


RED_STRICT = ("pledge", "default", "rating_down", "regulatory", "key_exit", "auditor_exit")
RED_BROAD = RED_STRICT + ("clarify", "fund_raise", "bonus_split")


def red_flags(C, types, window=63):
    """{decision date: names with a red-flag announcement in the last `window` trading
    days}, point-in-time (an announcement after 15:30 counts from the next day)."""
    from scripts.research_smart_ann import load, tradable_day
    D = load()
    D = D[D["type"].isin(types) & D["sym"].isin(C.columns)].copy()
    idx = C.index
    D["pos"] = tradable_day(D["ts"], idx)
    D = D[D["pos"] < len(idx)]
    hit = pd.DataFrame(0.0, index=idx, columns=C.columns)
    for p, sy in zip(D["pos"].values, D["sym"].values):
        hit.iat[p, hit.columns.get_loc(sy)] = 1.0
    live = hit.rolling(window, min_periods=1).max() > 0
    return {d: set(live.columns[live.loc[d].values]) for d in idx}


def cmd_veto(a):
    P, X, Xr, C, univ, mkt, cm = A.setup(a.capital)
    sc = A.best_score(Xr); sc = sc[sc.index.get_level_values("date") <= IT.DEV_END]
    IT.SHOW_HOLDOUT = False
    for lab, v in (("no veto", None), ("veto strict", red_flags(C, RED_STRICT)), ("veto broad", red_flags(C, RED_BROAD))):
        nav, to = simulate_book(sc, C, mkt, cm, veto=v, **FINAL)
        print(f"  {lab:<14} {IT.fmt(IT.split_stats(nav))}  {IT.mfmt(nav)}  TO {to:4.0%}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["grid", "final", "veto"])
    ap.add_argument("--start", default="2026-09-01")
    ap.add_argument("--capital", type=float, default=2e5)
    a = ap.parse_args()
    {"grid": cmd_grid, "final": cmd_final, "veto": cmd_veto}[a.cmd](a)
