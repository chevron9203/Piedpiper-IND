"""
Iteration harness for the smart system: evaluate saved walk-forward predictions under
realistic, per-stock costs, with a strict dev / holdout split.

  DEV      2013..2022   every design decision is made here
  HOLDOUT  2023+        reported, never tuned on -- look once a variant is chosen

Run:  python scripts/research_smart_iter.py costs [--pred h21_s5_ev]
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))
from scripts import research_smart_ml as SM

DEV_END = pd.Timestamp("2022-12-31")


SHOW_HOLDOUT = False


def split_stats(nav):
    out = {}
    parts = [("dev", nav[:DEV_END])] + ([("hold", nav[DEV_END:])] if SHOW_HOLDOUT else [])
    for name, s in parts:
        s = s/s.iloc[0]
        out[name] = SM.stats(s)
    return out


def fmt(st):
    return "  ".join(f"{k} {c:+6.1%}/{d:6.1%}/{sh:4.2f}" for k, (c, d, sh) in st.items())


def load(tag):
    P = SM.load_panel()
    X, C, univ, mkt = SM.compute_features(P, 5)
    Xr = SM.rank_features(X)
    pred = pd.read_parquet(BASE/f"data_store/smart_ml_pred_{tag}.parquet")["pred"]
    dts = pred.index.get_level_values("date").unique()
    mom = Xr["mom_riskadj"]
    mom = mom[mom.index.get_level_values("date").isin(dts)]
    return P, X, C, univ, mkt, pred, mom


def cmd_costs(a):
    P, X, C, univ, mkt, pred, mom = load(a.pred)
    # sanity: estimated one-way cost by liquidity rank, at Rs 25L
    cm = SM.cost_model(P, 25e5, 20)
    liq = X["log_turn60"].groupby(level="date").rank(ascending=False)
    c = cm.stack(future_stack=True).rename_axis(["date", "sym"]).reindex(X.index)
    b = pd.cut(liq, [0, 100, 300, 500, 800], labels=["1-100", "101-300", "301-500", "501-800"])
    print("median one-way cost (fees+half-spread+impact @Rs25L) by liquidity rank:")
    print((c.groupby(b, observed=True).median()*100).round(3).to_string(), "\n")

    print(f"{'':30} {'dev 2013-22 CAGR/DD/Sh':>30}  {'holdout 2023+':>22}   TO/yr")
    for cap in (2e5, 25e5, 1e7, 5e7):
        cm = SM.cost_model(P, cap, a.top)
        for name, sc in (("ML", pred), ("mom", mom)):
            nav, to, _ = SM.simulate(sc, C, univ, a.top, a.exit, 5, cost=cm)
            print(f"  Rs{cap/1e5:>5.0f}L {name:<4}            {fmt(split_stats(nav))}  {to:5.0%}", flush=True)


def cmd_window(a):
    """A live-like check on one recent window: the book you would have started on the
    last decision date before START, with models trained only on data before that year."""
    P, X, C, univ, mkt, pred, mom = load(a.pred)
    start = pd.Timestamp(a.start)
    dts = pred.index.get_level_values("date").unique()
    d0 = dts[dts < start].max()
    cm = SM.cost_model(P, a.capital, a.top)
    ix = pd.read_parquet(BASE/"data_store/nse_index.parquet")
    print(f"window: decide {d0.date()}, trade next close, through {C.index[-1].date()}\n")
    res = {}
    for name, sc in (("ML", pred), ("momentum", mom)):
        sc = sc[sc.index.get_level_values("date") >= d0]
        nav, to, nt = SM.simulate(sc, C, univ, a.top, a.exit, 5, cost=cm)
        res[name] = nav
        first = sc.xs(d0, level="date").sort_values(ascending=False).head(a.top)
        print(f"{name}: {nav.iloc[-1] - 1:+.2%} after costs, {nt} trades; first book: {', '.join(first.index)}")
    t0 = res["ML"].index[0]
    ew = (1 + mkt.loc[t0:].iloc[1:].fillna(0)).cumprod()
    print(f"equal-wt universe: {ew.iloc[-1] - 1:+.2%}")
    for nm in ("Nifty 50", "Nifty Midcap 150", "Nifty Smallcap 250", "Nifty 500"):
        s = ix[ix["index"].str.lower() == nm.lower()].set_index("date")["close"].sort_index()
        s = s[s.index >= t0]
        if len(s) > 1:
            print(f"{nm:<20} {s.iloc[-1]/s.iloc[0] - 1:+.2%}  (through {s.index[-1].date()})")
    out = pd.DataFrame(res); out["ew"] = ew
    print("\nweekly path:\n" + ((out.resample("W-FRI").last() - 1)*100).round(2).to_string())


def cmd_downmonths(a):
    """The user's real test, scaled up: in every month the market fell, did we make money?"""
    P, X, C, univ, mkt, pred, mom = load(a.pred)
    cm = SM.cost_model(P, a.capital, a.top)
    ix = pd.read_parquet(BASE/"data_store/nse_index.parquet")
    n500 = ix[ix["index"].str.lower() == "nifty 500"].set_index("date")["close"].sort_index()
    navs = {}
    for name, sc in (("ML", pred), ("mom", mom)):
        navs[name], _, _ = SM.simulate(sc, C, univ, a.top, a.exit, 5, cost=cm)
    navs["nifty500"] = n500
    M = pd.DataFrame(navs).ffill().resample("ME").last().pct_change().dropna()
    if not SHOW_HOLDOUT:
        M = M[:DEV_END]
    print(f"months {M.index[0]:%Y-%m}..{M.index[-1]:%Y-%m} ({len(M)}), capital Rs{a.capital/1e5:.0f}L")
    for lab, mask in (("all months", M["nifty500"] > -9), ("Nifty500 down", M["nifty500"] < 0),
                      ("Nifty500 down >3%", M["nifty500"] < -0.03), ("Nifty500 up", M["nifty500"] >= 0)):
        g = M[mask]
        print(f"\n{lab}: {len(g)} months, Nifty500 avg {g['nifty500'].mean():+.2%}")
        for k in ("ML", "mom"):
            print(f"  {k:<4} avg {g[k].mean():+.2%}  made money {(g[k] > 0).mean():4.0%}  "
                  f"beat Nifty {(g[k] > g['nifty500']).mean():4.0%}  worst {g[k].min():+.1%}")


def cmd_hedge(a):
    """Keep the stock-picking edge, remove market risk when the trend is down:
    short Nifty 50 futures (beta-sized) while a regime rule says bearish.
    Regime is decided on the close and applied from the next day."""
    P, X, C, univ, mkt, pred, mom = load(a.pred)
    cm = SM.cost_model(P, a.capital, a.top)
    ix = pd.read_parquet(BASE/"data_store/nse_index.parquet")
    n50 = ix[ix["index"].str.lower() == "nifty 50"].set_index("date")["close"].sort_index()
    rn = n50.pct_change()
    FUT_COST = 0.0004 + 0.0002          # per unit of hedge changed + monthly roll, rough
    rules = {
        "no hedge": pd.Series(0.0, index=n50.index),
        "always beta-hedged": pd.Series(1.0, index=n50.index),
        "N50 < 100dma": (n50 < n50.rolling(100).mean()).astype(float),
        "N50 < 200dma": (n50 < n50.rolling(200).mean()).astype(float),
        "N50 50dma<200dma": (n50.rolling(50).mean() < n50.rolling(200).mean()).astype(float),
        "N50 63d ret < 0": (n50/n50.shift(63) < 1).astype(float),
    }
    for name, sc in (("ML", pred), ("mom", mom)):
        nav, _, _ = SM.simulate(sc, C, univ, a.top, a.exit, 5, cost=cm)
        rp = nav.pct_change().dropna()
        rnn = rn.reindex(rp.index).fillna(0)
        beta = (rp.rolling(126, min_periods=60).cov(rnn)/rnn.rolling(126, min_periods=60).var()).shift(1).clip(0, 1.5).fillna(1.0)
        print(f"\n{name} (capital Rs{a.capital/1e5:.0f}L)   dev CAGR/DD/Sharpe   | down>3% months avg / made money")
        for rule, on in rules.items():
            h = (on.reindex(rp.index).ffill().shift(1).fillna(0))*beta
            r = rp - h*rnn - h.diff().abs().fillna(0)*FUT_COST - h*FUT_COST/21
            hn = (1 + r).cumprod()
            m = pd.DataFrame({"s": hn, "n": n50.reindex(hn.index).ffill()}).resample("ME").last().pct_change().dropna()
            if not SHOW_HOLDOUT: m = m[:DEV_END]
            dm = m[m["n"] < -0.03]
            print(f"  {rule:<20} {fmt(split_stats(hn))}  | {dm['s'].mean():+.2%} / {(dm['s'] > 0).mean():.0%} (n={len(dm)})"
                  f"  hedged {h.gt(0).mean():.0%} of days")


def monthly(nav):
    m = nav.resample("ME").last().pct_change().dropna()
    if not SHOW_HOLDOUT:
        m = m[:DEV_END]
    return m


def mfmt(nav):
    m = monthly(nav)
    return f"month avg {m.mean():+.2%} up {(m > 0).mean():3.0%} worst {m.min():+.1%}"


def xs(score):
    return score.groupby(level="date").rank(pct=True)


def cmd_blend(a):
    """Better PICKING, not hedging: combine model views, vary book size and cadence."""
    P, X, C, univ, mkt, pred, mom = load(a.pred)
    cm = SM.cost_model(P, a.capital, 20)
    comp = {"mom": xs(mom)}
    for tag in a.blend_preds.split(","):
        f = BASE/f"data_store/smart_ml_pred_{tag}.parquet"
        if f.exists():
            comp[tag] = xs(pd.read_parquet(f)["pred"])
    common = None
    for v in comp.values():
        common = v.index if common is None else common.intersection(v.index)
    comp = {k: v.reindex(common) for k, v in comp.items()}
    print("components:", list(comp), f"rows {len(common):,}\n")
    ml = [k for k in comp if k != "mom"]
    recipes = {k: {k: 1} for k in comp}
    recipes["ML all horizons"] = {k: 1 for k in ml}
    for w in (0.25, 0.5):
        recipes[f"ML all + mom {w:.0%}"] = {**{k: (1 - w)/len(ml) for k in ml}, "mom": w}
    dates = sorted(set(common.get_level_values("date")))
    for name, wts in recipes.items():
        sc = sum(comp[k]*w for k, w in wts.items())
        for top in (10, 20):
            for step in (5, 20):
                keep = set(dates[::step//5])
                s2 = sc[sc.index.get_level_values("date").isin(keep)]
                cmx = SM.cost_model(P, a.capital, top) if top != 20 else cm
                nav, to, _ = SM.simulate(s2, C, univ, top, a.exit, step, cost=cmx)
                print(f"  {name:<22} top{top:<3}{'weekly ' if step == 5 else 'monthly'} {fmt(split_stats(nav))}  "
                      f"{mfmt(nav)}  TO {to:4.0%}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["costs", "window", "downmonths", "hedge", "blend"])
    ap.add_argument("--holdout", action="store_true", help="also report 2023+ (look rarely)")
    ap.add_argument("--start", default="2026-09-01")
    ap.add_argument("--capital", type=float, default=2e5)
    ap.add_argument("--pred", default="h21_s5_ev")
    ap.add_argument("--blend-preds", default="h10_s5_ev,h21_s5_ev,h63_s5_ev")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--exit", type=int, default=100)
    a = ap.parse_args()
    SHOW_HOLDOUT = a.holdout
    {"costs": cmd_costs, "window": cmd_window, "downmonths": cmd_downmonths, "hedge": cmd_hedge, "blend": cmd_blend}[a.cmd](a)
