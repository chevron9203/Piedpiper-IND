"""
Independent verification of the smart-system backtest chain ("is the data TRUE?").

  bench      stitched benchmark series (index renames) + sanity of the splice days
  panel      rebuilt panel == backup panel (build is deterministic) ; adjusted vs traded price algebra
  trips      every round trip of the live rules re-priced from eod2's independently adjusted
             closes; disagreements listed and judged against raw price gaps
  circuits   how many entries/exits would have hit a price-band lock (cannot be filled at the
             close) and what the backtest would earn if those fills slip one day
  truncate   LOOK-AHEAD test of the feature code: rebuild features at date T from data that ends
             at T (nothing after T exists) and compare with the research feature store

Run:  python scripts/research_smart_verify.py bench|panel|trips|circuits|truncate
"""
from __future__ import annotations
import argparse, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))
from scripts import research_smart_ml as SM          # noqa: E402
from scripts import research_smart_exit as E         # noqa: E402

BACKUP = Path("/private/tmp/claude-501/-Users-abc-Documents-ozi-code-piedpiper/39072078-e4f7-482f-a865-14219d95bb04/scratchpad/nse_panel.bak.parquet")
EOD2 = BASE/"data_store/eod2/src/eod2_data/daily"
CHAINS = {"nifty500": ["S&P CNX 500", "CNX 500", "NIFTY 500"],
          "nifty50": ["S&P CNX NIFTY", "CNX NIFTY", "NIFTY 50"]}
# NOT stitched: Midcap 150 (starts 2016-04) - NSE ran a different 'FULL MIDCAP 100' series in between; quote it only from its own start


def bench(name):
    """Stitched index close: chain the old/new names by DAILY RETURNS so a rename or rebase
    can never show up as a fake jump. Returns (series, list of splice-day returns)."""
    ix = pd.read_parquet(BASE/"data_store/nse_index.parquet")
    parts = [ix[ix["index"].str.upper() == n].set_index("date")["close"].sort_index() for n in CHAINS[name]]
    rets, splice = [], []
    for k, s in enumerate(parts):
        r = s.pct_change()
        if k > 0:
            prev_end = parts[k - 1].index[-1]
            gap_r = s.iloc[0]/parts[k - 1].iloc[-1] - 1            # last old close -> first new close
            splice.append((CHAINS[name][k - 1], CHAINS[name][k], str(prev_end.date()), str(s.index[0].date()), float(gap_r)))
            r.iloc[0] = gap_r
        rets.append(r.iloc[1:] if k == 0 else r)
    r = pd.concat(rets).sort_index()
    r = r[~r.index.duplicated(keep="last")]
    out = (1 + r.fillna(0)).cumprod()
    return out, splice


def cmd_bench(a):
    for nm in CHAINS:
        s, sp = bench(nm)
        print(f"{nm}: {s.index[0].date()} .. {s.index[-1].date()}  ({len(s)} days)")
        for old, new, d0, d1, r in sp:
            print(f"   splice {old} -> {new}: {d0} -> {d1}: return across the rename {r:+.2%}")
        big = s.pct_change().abs().nlargest(3)
        print("   largest daily moves:", {str(k.date()): f"{v:.1%}" for k, v in big.items()})


def cmd_panel(a):
    new = pd.read_parquet(BASE/"data_store/nse_panel.parquet")
    old = pd.read_parquet(BACKUP)
    print(f"panel shape new {new.shape} backup {old.shape}")
    ok = True
    for f in ("close", "vol", "turn", "ca_factor", "high", "low", "open"):
        x = new[f]; y = old[f].reindex(index=x.index, columns=x.columns)
        same = np.array_equal(np.nan_to_num(x.values, nan=-1.0), np.nan_to_num(y.values, nan=-1.0))
        diff = float(np.nanmax(np.abs(x.values - y.values))) if not same else 0.0
        print(f"   {f:<10} identical: {same}  max abs diff {diff:.3g}")
        ok &= same
    print("REBUILD DETERMINISTIC:", ok)
    # price algebra: traded price * cumulative later factors == adjusted close (spot-check every symbol)
    P = SM.load_panel(); C = P["close"]; tp = SM.traded_price(P)
    f = P["ca_factor"].reindex(index=C.index, columns=C.columns).fillna(1.0)
    after = f.iloc[::-1].cumprod().iloc[::-1]/f
    err = ((tp*after - C).abs()/C).stack().dropna()
    print(f"traded_price * later-factors vs adjusted close: max rel err {err.max():.2e}, share > 1e-4: {(err > 1e-4).mean():.2e}")
    n_act = int(((f != 1.0) & f.notna()).sum().sum())
    print(f"corporate-action factors applied across history: {n_act} (symbol-days)")


def trips_full():
    G = E.inputs()
    comp = pickle.load(open(E.COMP, "rb"))
    sc = (0.5*comp["ml"] + 0.5*comp["mom"].reindex(comp["ml"].index)).dropna()
    out = E.simulate_rules(sc, G["C"], G["mkt"], G["cm"], want_trips=True, **dict(E.FINAL, every=5, offset=0))
    return out, G


def cmd_trips(a):
    out, G = trips_full()
    tr = out["trips"].copy(); idx = G["C"].index
    tr["entry"] = idx[tr.entry_i]; tr["exit"] = idx[tr.exit_i]
    tr["ret_panel"] = tr.exit_px/tr.entry_px - 1
    print(f"{len(tr):,} round trips, {tr.sym.nunique()} distinct stocks, {tr.entry.min().date()}..{tr.exit.max().date()}")
    rows = []
    miss = []
    for sym, g in tr.groupby("sym"):
        f = EOD2/f"{sym.lower()}.csv"
        if not f.exists():
            miss.append(sym); continue
        e = pd.read_csv(f, parse_dates=["Date"]).set_index("Date")["Close"]
        for r in g.itertuples():
            if r.entry in e.index and r.exit in e.index and e[r.entry] > 0:
                rows.append((sym, r.entry, r.exit, r.ret_panel, e[r.exit]/e[r.entry] - 1, r.reason))
    R = pd.DataFrame(rows, columns=["sym", "entry", "exit", "panel", "eod2", "reason"])
    R["d"] = (R.panel - R.eod2).abs()
    dead = tr[tr.sym.isin(miss)]
    print(f"re-priced from eod2: {len(R):,} trips ({len(R)/len(tr):.0%}); not in eod2 (delisted/renamed names): {len(dead)} trips in {len(miss)} stocks")
    print(f"agreement |panel - eod2| : <1pp {(R.d < .01).mean():.1%}   <3pp {(R.d < .03).mean():.1%}   <10pp {(R.d < .10).mean():.1%}   max {R.d.max():.2f}")
    bad = R[R.d >= 0.10].sort_values("d", ascending=False)
    print(f"\ndisagreements >= 10pp: {len(bad)} trips")
    P = pd.read_parquet(BASE/"data_store/nse_panel.parquet", columns=None)
    Cr = SM.traded_price({"close": P["close"], "ca_factor": P["ca_factor"]})
    for r in bad.head(25).itertuples():
        raw = Cr[r.sym].loc[r.entry:r.exit].dropna()
        mv = raw.pct_change().abs()
        big = mv.nlargest(1)
        caf = P["ca_factor"][r.sym].loc[r.entry:r.exit]
        caf = caf[(caf != 1.0) & caf.notna()]
        print(f"  {r.sym:<11} {r.entry.date()}->{r.exit.date()}  panel {r.panel:+7.1%}  eod2 {r.eod2:+7.1%}  "
              f"biggest raw 1-day move {big.iloc[0]:.0%} on {big.index[0].date()}  our CA factors in window: "
              f"{ {str(k.date()): round(float(v), 3) for k, v in caf.items()} }")
    # effect on strategy: mean trip return under each source (equal weight)
    print(f"\nmean trip return: panel {R.panel.mean():+.2%}  eod2 {R.eod2.mean():+.2%}  (same trips; eod2 = independent adjustment)")
    big_gap = R[R.d >= .10]
    print(f"if EVERY disagreeing trip took eod2's number instead: mean {(R.eod2.where(R.d >= .10, R.panel)).mean():+.2%}  "
          f"(vs panel {R.panel.mean():+.2%}); trips affected {len(big_gap)}")
    pickle.dump(dict(R=R, tr=tr), open(BASE/"data_store/smart_verify_trips.pkl", "wb"))


def cmd_circuits(a):
    out, G = trips_full()
    tr = out["trips"].copy()
    P = SM.load_panel()
    idx = P["close"].index; cols = {c: j for j, c in enumerate(P["close"].columns)}
    Ca, H, L = P["close"].ffill(limit=5).values, P["high"].values, P["low"].values
    tpv = SM.traded_price(P).values
    _, R, alive, turn60, age, univ, mkt = SM.base(P)
    Uv = univ.values
    def flags(i, j):
        c, h, l = tpv[i, j], H[i, j], L[i, j]
        prev = tpv[i - 1, j]
        if not np.isfinite(prev) or prev <= 0: return (False, False)
        # price-band lock proxy: closes on its high after a >= +4.5% day / on its low after a <= -4.5% day
        # (note: H/L are adjusted too; compare adjusted close to adjusted high)
        r = Ca[i, j]/Ca[i - 1, j] - 1
        return (r >= 0.045 and Ca[i, j] >= 0.998*H[i, j], r <= -0.045 and Ca[i, j] <= 1.002*L[i, j])
    ent = np.array([flags(i, cols[s]) for i, s in zip(tr.entry_i, tr.sym)])
    ext = np.array([flags(i, cols[s]) for i, s in zip(tr.exit_i, tr.sym)])
    base_up = base_dn = n = 0
    rng = np.random.default_rng(0)
    for i in rng.choice(np.arange(300, len(idx) - 2), 400, replace=False):
        for j in np.where(Uv[i])[0]:
            u, d = flags(i, j); base_up += u; base_dn += d; n += 1
    print(f"price-band-lock proxy (|1-day move| >= 4.5% and close at the day's high/low), universe base rate over {n:,} stock-days:")
    print(f"   upper-lock {base_up/n:.2%}   lower-lock {base_dn/n:.2%}")
    print(f"   our ENTRIES on an upper-lock day: {ent[:, 0].mean():.1%} of {len(tr):,}   (the backtest assumes a fill at that close)")
    print(f"   our EXITS   on a lower-lock day : {ext[:, 1].mean():.1%} of {len(tr):,}   (the backtest assumes a fill at that close)")
    # re-price: locked entries buy the NEXT close; locked exits sell the NEXT close
    ret0 = tr.exit_px/tr.entry_px - 1
    e_px = tr.entry_px.values.copy(); x_px = tr.exit_px.values.copy()
    for k, r in enumerate(tr.itertuples()):
        j = cols[r.sym]
        if ent[k, 0] and r.entry_i + 1 < r.exit_i and np.isfinite(Ca[r.entry_i + 1, j]):
            e_px[k] = Ca[r.entry_i + 1, j]
        if ext[k, 1] and r.exit_i + 1 < len(idx) and np.isfinite(Ca[r.exit_i + 1, j]):
            x_px[k] = Ca[r.exit_i + 1, j]
    ret1 = x_px/e_px - 1
    print(f"\n   mean trip return, backtest fills {ret0.mean():+.2%}  |  locked fills slip one day {ret1.mean():+.2%}  "
          f"(cost of the assumption: {(ret0.mean() - ret1.mean())*100:.2f}pp per trip)")
    n_tr_year = len(tr)/ ((idx[tr.exit_i.max()] - idx[tr.entry_i.min()]).days/365.25)
    print(f"   at {n_tr_year:.0f} trips a year (1/20 of capital each) that is about {(ret0.mean() - ret1.mean())*n_tr_year/20*100:.1f}pp of CAGR")
    f = ent[:, 0]
    print(f"   upper-lock entries: avg trip return {ret0[f].mean():+.1%} (n={f.sum()}) vs others {ret0[~f].mean():+.1%}")


def cmd_truncate(a):
    """Rebuild features at T from data that ENDS at T, compare to the research store row at T."""
    from smart import window as W, features as F
    store = pd.read_parquet(BASE/"data_store/smart_live/feature_store.parquet")
    dates = store.index.get_level_values("date").unique()
    refresh = dates[::4]                                          # peers are refreshed on these dates only
    P_full = SM.load_panel()
    pick = [refresh[k] for k in (60, 100, 140, 180, 215, 230)]
    allbad = 0
    for T in pick:
        st = W.init_state_from_panel(P_full["close"].loc[:T])
        P = W.build(asof=T.strftime("%Y-%m-%d"))
        assert P["close"].index[-1] == T, (P["close"].index[-1], T)
        X, *_ = F.rows(P, pd.DatetimeIndex([T]), W.age_offset(st, P))
        R = store[store.index.get_level_values("date") == T]
        common = X.index.intersection(R.index)
        a_, b_ = X.loc[common].astype(float), R.loc[common, X.columns].astype(float)
        both = a_.notna() & b_.notna()
        rel = ((a_ - b_).abs()/(b_.abs() + 1e-6)).where(both)
        nanmis = (a_.isna() != b_.isna()).mean()
        bad = [(c, float(rel[c].max()), float(nanmis[c])) for c in X.columns if rel[c].max() > 2e-3 or nanmis[c] > 0.003]
        allbad += len(bad)
        print(f"T={T.date()}  window ends T  rows live {len(X)} store {len(R)} common {len(common)}  "
              f"max rel err {np.nanmax(rel.values):.1e}  features failing: {len(bad)} {bad[:4]}", flush=True)
    print("LOOK-AHEAD TEST:", "PASS (no feature changes when the future is removed)" if allbad == 0 else f"{allbad} feature mismatches - investigate")


def cmd_scramble(a):
    """DECISIVE look-ahead test. Same window, same code, same history length -- but every price/volume
    value AFTER T is replaced by random garbage. Features at T must be bit-identical."""
    from smart import window as W, features as F
    store = pd.read_parquet(BASE/"data_store/smart_live/feature_store.parquet", columns=["r1"])
    dates = store.index.get_level_values("date").unique(); refresh = dates[::4]
    P_full = SM.load_panel(); idx = P_full["close"].index
    rng = np.random.default_rng(42)
    fails = 0
    for k in (100, 140, 180, 215):
        T = refresh[k]; T2 = idx[idx.get_loc(T) + 60]
        st = W.init_state_from_panel(P_full["close"].loc[:T2])
        P = W.build(asof=T2.strftime("%Y-%m-%d"))
        off = W.age_offset(st, P)
        P2 = {f: v.copy() for f, v in P.items()}
        fut = P2["close"].index > T
        for f in ("open", "high", "low", "close", "vol", "turn", "trades", "dlv"):
            v = P2[f].values.copy()
            noise = rng.uniform(0.4, 2.5, size=v[fut].shape).astype("float32")
            v[fut] = v[fut]*noise
            P2[f] = pd.DataFrame(v, index=P2[f].index, columns=P2[f].columns)
        # ca_factor is left AS BUILT: erasing future factors while the closes stay back-adjusted would create
        # an inconsistent world (wrong traded price at T for any stock splitting after T) - a test artefact
        X1, *_ = F.rows(P, pd.DatetimeIndex([T]), off)
        X2, *_ = F.rows(P2, pd.DatetimeIndex([T]), off)
        same_idx = X1.index.equals(X2.index)
        A, B = X1.values.astype("float64"), X2.reindex(X1.index).values.astype("float64")
        eq = np.array_equal(np.nan_to_num(A, nan=-9e9), np.nan_to_num(B, nan=-9e9))
        mx = np.nanmax(np.abs(A - B)) if not eq else 0.0
        bad = [c_ for j, c_ in enumerate(X1.columns) if not np.array_equal(np.nan_to_num(A[:, j], nan=-9e9), np.nan_to_num(B[:, j], nan=-9e9))]
        print(f"T={T.date()} (data to {T2.date()}, {int(fut.sum())} future days scrambled): rows {len(X1)}, same stock set {same_idx}, "
              f"features bit-identical: {eq}  max abs diff {mx:.2e}  differing columns: {bad}", flush=True)
        fails += (not eq) or (not same_idx)
    print("SCRAMBLE-THE-FUTURE TEST:", "PASS - no feature at T depends on anything after T" if fails == 0 else f"FAIL on {fails} dates")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bench", "panel", "trips", "circuits", "truncate", "scramble"])
    a = ap.parse_args()
    {"bench": cmd_bench, "panel": cmd_panel, "trips": cmd_trips, "circuits": cmd_circuits, "truncate": cmd_truncate, "scramble": cmd_scramble}[a.cmd](a)
