"""
Exit and quality-gate research for the smart system (DEV 2013-2022 only).

Questions (user, 2026-10-03): why hold 20 stocks when only a few look good? how and how
often should we EXIT? what if a holding suddenly goes bad?

  prep     build the compact inputs (close, market, costs, v1.1 blended score) once
  diag     evidence first: what each score rank is worth, what swaps earn net of cost,
           what happens to stocks after they fall X% from entry / peak
  exits    grid of exit rules, rank thresholds and decision frequencies
  gates    absolute quality gates on entries (trend etc.), cash vs concentrate

simulate_rules() is simulate_book() plus: daily exit rules (flag at close t, sell at close
t+1, with that day's cost), cooldown after a rule-exit, cash that earns a rate, gates,
decision frequency, and a full round-trip log. With no rules it must reproduce
simulate_book() exactly (checked in `exits`).

Run:  python scripts/research_smart_exit.py prep | diag | exits | gates
"""
from __future__ import annotations
import argparse, multiprocessing as mp, pickle, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))
from scripts import research_smart_ml as SM          # noqa: E402
from scripts import research_smart_iter as IT        # noqa: E402

INPUTS = BASE/"data_store/smart_exit_inputs.pkl"
FINAL = dict(top_n=20, exit_rank=100, band=0.25, max_corr=0.5)


# ------------------------------------------------------------------ the decision
def decide2(s, w, Rres_win, top_n, exit_rank, band, max_corr, can_buy, must_sell,
            min_slots=1, depth=6, winners=None, boost=frozenset(), boost_mult=1.0, boost_rank=20,
            boost_max=0.15):
    """research_smart_book.decide() with hooks: names that fail a gate can't be bought
    (can_buy) and/or must be sold (must_sell); weight = 1/max(n_names, min_slots), so
    min_slots=top_n leaves unfilled slots in cash instead of concentrating."""
    rank = pd.Series(np.arange(1, len(s) + 1), index=s.index)
    keep = [x for x in w.index if x in rank.index and rank[x] <= exit_rank and not must_sell(x)]
    need = top_n - len(keep)
    new = []
    if need > 0:
        cands = [x for x in s.index[: top_n*depth] if x not in keep and can_buy(x) and not must_sell(x)]
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
            if len(new) < need:                    # never leave slots empty (as decide())
                new += [x for x in cands if x not in new][: need - len(new)]
    names = keep + new
    if not names:
        return pd.Series(dtype=float)
    if winners is not None:
        # LET WINNERS RUN: kept names are never trimmed or topped up (only clipped at `cap`);
        # capital released by exits (and by clipping) funds the new entrants equally.
        cap = winners.get("cap")
        wk = w.reindex(keep).astype(float)
        if cap:
            wk = wk.clip(upper=cap)
        free = max(1.0 - float(wk.sum()), 0.0)
        tn = pd.Series(min(free/len(new), cap or 0.10), index=new) if new else pd.Series(dtype=float)
        return pd.concat([wk, tn])
    unit = 1.0/max(len(names), min_slots)
    target = pd.Series(unit, index=names)
    if band > 0:
        for x in keep:
            if abs(target[x] - w[x]) <= band*target[x]:
                target[x] = w[x]
        fresh = [x for x in names if x not in keep or abs(target[x] - w.get(x, 0)) > band*target[x]]
        fixed = [x for x in names if x not in fresh]
        if fresh:
            t2 = target[fresh]
            t2 = t2/t2.sum()*(len(names)*unit - target[fixed].sum())
            target[fresh] = t2
    if boost and boost_mult != 1.0:                    # size up (or down) chosen holdings
        rb = [x for x in boost if x in target.index and rank.get(x, 10**9) <= boost_rank]
        if rb:
            tot = float(target.sum())
            tb = pd.Series(min(unit*boost_mult, boost_max), index=rb)
            rest = target.drop(rb)
            if len(rest) and rest.sum() > 0:
                rest = rest*max(tot - float(tb.sum()), 0.0)/rest.sum()
            target = pd.concat([tb, rest])
    return target


# ------------------------------------------------------------------ the simulator
def simulate_rules(score, C, mkt, cost, top_n=20, exit_rank=100, band=0.25, max_corr=0.5,
                   corr_win=126, every=1, offset=0, rules=None, cooldown=21, cash_r=0.0, gate_mask=None,
                   gate_hold=False, min_slots=1, depth=6, want_trips=False, end_date=None, winners=None, addon=None, hold_log=None, expo=None):
    """rules (all optional, all evaluated on the CLOSE; a flag sells at the NEXT close):
         stop=0.10        close <= entry*(1-stop)
         trail=0.20       close <= peak-since-entry*(1-trail)
         trail_vol=2.0    drop from peak > k * 63d daily vol * sqrt(21)
         shock=0.10       one-day return <= -shock
         time=(T, thr)    held >= T trading days and return since entry < thr
         trend_n=3        close below its 50-day average for n days in a row
         tp=0.5           take profit: close >= entry*(1+tp)
         spike=0.25       take profit on a parabolic move: 5-day return >= spike
       cash_r: annual rate on uninvested cash, or a daily-return Series (park in the market).
       gate_mask: bool matrix (days x symbols): may BUY only where True (and, with
       gate_hold, must SELL where False). Rule-exits wait `cooldown` days before re-entry."""
    rules = rules or {}
    Cf = C.ffill(limit=5)
    R = Cf.pct_change(fill_method=None)
    Rres = R.sub(mkt, axis=0)
    idx = C.index; cols_all = list(C.columns)
    colpos = {c: j for j, c in enumerate(cols_all)}
    Pv = Cf.values; Rv = R.values
    vol = R.rolling(63, min_periods=40).std().values if rules.get("trail_vol") else None
    sma = Cf.rolling(50, min_periods=40).mean().values if rules.get("trend_n") else None
    last_valid = C.apply(lambda s: s.last_valid_index())
    if isinstance(cash_r, pd.Series):
        cash_d = cash_r.reindex(idx).fillna(0.0).values
    else:
        cash_d = np.full(len(idx), (1 + float(cash_r))**(1/252) - 1)

    sdates = [d for d in sorted(set(score.index.get_level_values("date"))) if idx.get_loc(d) + 1 < len(idx)]
    sdates = sdates[offset::every]
    end_i = int(idx.searchsorted(pd.Timestamp(end_date), side="right")) - 1 if end_date is not None else len(idx) - 1
    by_date = {d: g.droplevel(0).sort_values(ascending=False) for d, g in score.groupby(level="date")}
    nav = [1.0]; navd = [idx[idx.get_loc(sdates[0]) + 1]]
    w = pd.Series(dtype=float); turnover = 0.0
    m_prev = 1.0; switches = 0
    pos, pending, cool, trips, expo_hist = {}, {}, {}, [], []

    def open_trip(x, i):
        p = Pv[i, colpos[x]]
        pos[x] = dict(entry_i=i, entry_px=p, peak=p, trough=p, last=p, brk=0)

    def close_trip(x, i, reason, dead=False):
        st = pos.pop(x, None)
        if st is None:
            return
        px = st["last"]*(1 - SM.DEAD_HAIRCUT) if dead else Pv[i, colpos[x]]
        if not np.isfinite(px):
            px = st["last"]
        if want_trips:
            trips.append(dict(sym=x, entry_i=st["entry_i"], exit_i=i, entry_px=st["entry_px"], exit_px=px,
                              peak=st["peak"], trough=st["trough"], reason=reason))

    for k, d in enumerate(sdates):
        i_d = idx.get_loc(d); i0 = i_d + 1
        mrow = gate_mask[i_d] if gate_mask is not None else None
        can_buy = lambda x: (mrow is None or mrow[colpos[x]]) and x not in pending and cool.get(x, -1) < i0
        must_sell = (lambda x: mrow is not None and gate_hold and not mrow[colpos[x]])
        boost = set()
        if addon:        # holdings below (or above) cost by dd: size them by addon["mult"] if still ranked
            for x in w.index:
                st = pos.get(x)
                p = Pv[i_d, colpos[x]]
                if st is not None and np.isfinite(p) and st["entry_px"] > 0:
                    rt = p/st["entry_px"] - 1
                    if (addon["when"] == "under" and rt <= -addon["dd"]) or (addon["when"] == "over" and rt >= addon["dd"]):
                        boost.add(x)
        m_new = 1.0
        w_in = w
        if expo is not None:           # market-level EXPOSURE overlay: scale every position, the rest earns cash_r
            raw = float(expo(i_d, nav) if callable(expo) else expo.iloc[i_d])
            m_new = raw
            if abs(m_new - m_prev) > 1e-9: switches += 1
            m_prev = m_new
            sw = float(w.sum())
            w_in = w/sw if sw > 1e-9 else w          # decide on RELATIVE weights so the band logic is unaffected
        target = decide2(by_date[d], w_in, Rres.iloc[max(0, i_d - corr_win + 1): i_d + 1], top_n, exit_rank,
                         band, max_corr, can_buy, must_sell, min_slots, depth, winners, boost,
                         addon["mult"] if addon else 1.0, addon["rank"] if addon else 20)
        if expo is not None:
            target = target*m_new
        dwi = target.sub(w, fill_value=0).abs()
        ci = cost.iloc[i0].reindex(dwi.index).fillna(0.01)
        c = (dwi*ci).sum()
        turnover += dwi.sum()/2
        for x in list(w.index):
            if x not in target.index:
                close_trip(x, i0, "rank")
        for x in target.index:
            if x not in w.index:
                open_trip(x, i0)
        w = target
        pending = {x: r_ for x, r_ in pending.items() if x in w.index}
        v = nav[-1]*(1 - c)
        i1 = idx.get_loc(sdates[k + 1]) + 1 if k + 1 < len(sdates) else end_i
        cols = [colpos[x] for x in w.index]
        for i in range(i0 + 1, i1 + 1):
            r = pd.Series(Rv[i, cols], index=w.index)
            died = [x for x in w.index if np.isnan(r[x]) and last_valid[x] < idx[i]
                    and last_valid[x] < idx[-1] - pd.Timedelta(days=30)]
            r = r.fillna(0.0)
            for x in died:
                r[x] = -SM.DEAD_HAIRCUT
            cashw = max(1.0 - float(w.sum()), 0.0)
            expo_hist.append(float(w.sum()))
            pr = float((w*r).sum()) + cashw*cash_d[i]
            if hold_log is not None:                       # (day index, symbol, weight, return, NAV before the day)
                for x in w.index:
                    hold_log.append((i, x, float(w[x]), float(r[x]), v))
            v *= 1 + pr
            w = w*(1 + r)/(1 + pr) if len(w) else w
            for x in died:
                close_trip(x, i, "dead", dead=True)
                pending.pop(x, None)
            if died:
                w = w.drop(died)
            for x in w.index:                                  # path statistics
                p = Pv[i, colpos[x]]
                if np.isfinite(p):
                    st = pos[x]
                    st["last"] = p
                    if p > st["peak"]: st["peak"] = p
                    if p < st["trough"]: st["trough"] = p
            # execute yesterday's flags at today's close
            crow = cost.iloc[i]
            for x in [x for x in w.index if x in pending]:
                f = float(w[x]*(crow.get(x, 0.01) if np.isfinite(crow.get(x, np.nan)) else 0.01))
                turnover += float(w[x])/2
                v *= 1 - f
                w = w.drop(x)/(1 - f)
                close_trip(x, i, pending[x])
                cool[x] = i + cooldown
            pending = {}
            cols = [colpos[x] for x in w.index]
            if rules:                                          # evaluate rules at today's close
                for x in w.index:
                    j = colpos[x]; p = Pv[i, j]
                    if not np.isfinite(p):
                        continue
                    st = pos[x]
                    ret = p/st["entry_px"] - 1; dd = p/st["peak"] - 1
                    if sma is not None:
                        st["brk"] = st["brk"] + 1 if p < sma[i, j] else 0
                    why = None
                    if rules.get("stop") and ret <= -rules["stop"]: why = "stop"
                    elif rules.get("trail") and dd <= -rules["trail"]: why = "trail"
                    elif rules.get("trail_vol") and np.isfinite(vol[i, j]) \
                            and dd <= -rules["trail_vol"]*vol[i, j]*np.sqrt(21): why = "trail_vol"
                    elif rules.get("shock") and Rv[i, j] <= -rules["shock"]: why = "shock"
                    elif rules.get("time") and (i - st["entry_i"]) >= rules["time"][0] \
                            and ret < rules["time"][1]: why = "time"
                    elif rules.get("trend_n") and st["brk"] >= rules["trend_n"]: why = "trend"
                    elif rules.get("tp") and ret >= rules["tp"]: why = "tp"
                    elif rules.get("spike") and i >= 5 and p/Pv[i - 5, j] - 1 >= rules["spike"]: why = "spike"
                    if why:
                        pending[x] = why
            nav.append(v); navd.append(idx[i])
    nav = pd.Series(nav, index=navd)
    yrs = (nav.index[-1] - nav.index[0]).days/365.25
    out = dict(nav=nav, turnover=turnover/yrs, exposure=float(np.mean(expo_hist)) if expo_hist else 1.0, switches=switches)
    if want_trips:
        out["trips"] = pd.DataFrame(trips)
        out["idx"] = idx
    return out


# ------------------------------------------------------------------ inputs
def prep(a):
    from scripts import research_smart_audit as A
    from scripts import research_smart_book as B
    P, X, Xr, C, univ, mkt, cm = A.setup(a.capital)
    A.MOM_W = 0.5
    sc = A.best_score(Xr)
    sc = sc[sc.index.get_level_values("date") <= IT.DEV_END]
    nav, _ = B.simulate_book(sc, C, mkt, cm, **B.FINAL)
    ref = float(nav[:IT.DEV_END].iloc[-1])           # the final book is held to 2026: compare at DEV end
    reg = pd.read_parquet(BASE/"data_store/smart_features_s5.parquet", columns=["m_breadth", "m_r63"])
    reg = reg.groupby(level="date").first()
    pickle.dump(dict(C=C.astype("float32"), mkt=mkt, cm=cm.astype("float32"), sc=sc,
                     regime=reg, ref_nav_end=ref), open(INPUTS, "wb"))
    print(f"saved {INPUTS.name}: C {C.shape}, scores {len(sc):,}, reference NAV at DEV end {ref:.4f}")


_G = {}


def inputs():
    if not _G:
        _G.update(pickle.load(open(INPUTS, "rb")))
    return _G


def gate_matrix(name, C, mkt):
    Cf = C.ffill(limit=5)
    sma50 = Cf.rolling(50, min_periods=40).mean(); sma200 = Cf.rolling(200, min_periods=150).mean()
    r63 = Cf/Cf.shift(63) - 1; r126 = Cf/Cf.shift(126) - 1
    hi = Cf.rolling(252, min_periods=200).max()
    mc = (1 + mkt.fillna(0)).cumprod(); m63 = (mc/mc.shift(63) - 1)
    m = {"sma50": Cf > sma50,
         "trend": (Cf > sma50) & (Cf > sma200),
         "mom": (r126 > 0) & (Cf > sma50),
         "rs": r63.gt(m63, axis=0) & (Cf > sma50),
         "near_high": Cf >= 0.80*hi,
         "strong": (Cf > sma50) & (Cf > sma200) & (r126 > 0) & (Cf >= 0.75*hi)}[name]
    return m.fillna(False).values


_SC = {}


def run_cfg(cfg):
    G = inputs()
    if cfg.get("scfile"):                                   # daily-prediction score (see `dailygrid`)
        if "sc" not in _SC:
            _SC["sc"] = pickle.load(open(cfg["scfile"], "rb"))
        G = dict(G, sc=_SC["sc"])
    kw = dict(cfg["kw"])
    if "gate" in kw:
        kw["gate_mask"] = gate_matrix(kw.pop("gate"), G["C"], G["mkt"])
    if kw.pop("park", False):
        kw["cash_r"] = G["mkt"]
    t0 = time.time()
    res = simulate_rules(G["sc"], G["C"], G["mkt"], G["cm"], want_trips=True, end_date=IT.DEV_END, **kw)
    nav = res["nav"]; tr = res["trips"]
    st = IT.split_stats(nav)["dev"]
    m = nav.resample("ME").last().pct_change().dropna()
    yr = nav.resample("YE").last().pct_change(); yr.iloc[0] = nav.resample("YE").last().iloc[0]/nav.iloc[0] - 1
    hold = (tr["exit_i"] - tr["entry_i"]).mean() if len(tr) else np.nan
    reasons = tr["reason"].value_counts().to_dict() if len(tr) else {}
    return dict(label=cfg["label"], cagr=st[0], dd=st[1], sh=st[2], up=float((m > 0).mean()), worst=float(m.min()),
                to=res["turnover"], expo=res["exposure"], hold=float(hold), reasons=reasons,
                final=float(nav[:IT.DEV_END].iloc[-1]), yr={int(k.year): float(v) for k, v in yr.items()}, secs=time.time() - t0)


def fmt_row(r, base_cagr=None):
    rs = " ".join(f"{k}{v}" for k, v in sorted(r["reasons"].items()) if k not in ("rank",))
    return (f"  {r['label']:<40} {r['cagr']:+6.1%} / {r['dd']:6.1%} / {r['sh']:4.2f}  up {r['up']:.0%} "
            f"worst {r['worst']:+5.1%}  TO {r['to']:4.0%}  invested {r['expo']:.0%}  hold {r['hold']:.0f}d  {rs}")


def run_grid(cfgs, workers=6):
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=ctx) as ex:
        res = list(ex.map(run_cfg, cfgs))
    print(f"({len(cfgs)} runs in {time.time() - t0:.0f}s)")
    return res


# ------------------------------------------------------------------ commands
def cmd_exits(a):
    base = dict(FINAL)
    cfgs = [dict(label="BASELINE (live rules)", kw=dict(base))]
    for er in (30, 50, 200, 400):
        cfgs.append(dict(label=f"exit rank {er} (live 100)", kw=dict(base, exit_rank=er)))
    for ev, nm in ((2, "10d"), (4, "20d")):
        cfgs.append(dict(label=f"decide every {nm} (live 5d)", kw=dict(base, every=ev)))
    cfgs.append(dict(label="LET WINNERS RUN (no trim), no cap", kw=dict(base, winners=dict())))
    for cp in (0.20, 0.15, 0.10):
        cfgs.append(dict(label=f"LET WINNERS RUN, cap {cp:.0%}", kw=dict(base, winners=dict(cap=cp))))
    for bd in (1.0, 2.0):
        cfgs.append(dict(label=f"band {bd:.0%} (live 25%)", kw=dict(base, band=bd)))
    for s in (0.08, 0.12, 0.18):
        cfgs.append(dict(label=f"hard stop -{s:.0%} from entry", kw=dict(base, rules=dict(stop=s))))
    for t in (0.12, 0.20, 0.30):
        cfgs.append(dict(label=f"trailing stop -{t:.0%} from peak", kw=dict(base, rules=dict(trail=t))))
    for k in (1.5, 2.5):
        cfgs.append(dict(label=f"vol-scaled trail {k} monthly sigmas", kw=dict(base, rules=dict(trail_vol=k))))
    for s in (0.08, 0.12):
        cfgs.append(dict(label=f"shock exit: 1-day drop -{s:.0%}", kw=dict(base, rules=dict(shock=s))))
    cfgs.append(dict(label="time stop: 40d and return < 0", kw=dict(base, rules=dict(time=(40, 0.0)))))
    cfgs.append(dict(label="time stop: 60d and return < +5%", kw=dict(base, rules=dict(time=(60, 0.05)))))
    for n in (3, 10):
        cfgs.append(dict(label=f"trend break: {n}d below 50DMA", kw=dict(base, rules=dict(trend_n=n))))
    res = run_grid(cfgs, a.workers)
    G = inputs()
    print(f"\nbaseline reproduces simulate_book at DEV end: NAV {res[0]['final']:.4f} vs {G['ref_nav_end']:.4f} "
          f"({'OK' if abs(res[0]['final']/G['ref_nav_end'] - 1) < 1e-6 else 'MISMATCH'})")
    print("\nDEV 2013-2022, Rs2L per-stock costs   CAGR / maxDD / Sharpe\n")
    for r in res:
        print(fmt_row(r))
    print("\nby year (%):")
    print(pd.DataFrame({r["label"][:26]: r["yr"] for r in res[:1] + res[6:10]}).mul(100).round(0).to_string())
    pickle.dump(res, open(BASE/"data_store/smart_exit_results.pkl", "wb"))


def cmd_timing(a):
    """Is the decide-every-5d vs 10d/20d gap real, or just which weekday we happen to use?"""
    base = dict(FINAL)
    cfgs = [dict(label=f"every {5*ev}d, start offset {o}", kw=dict(base, every=ev, offset=o))
            for ev in (1, 2, 4) for o in range(ev)]
    res = run_grid(cfgs, a.workers)
    print("\nDEV CAGR / maxDD / Sharpe by decision frequency and starting offset:\n")
    for r in res:
        print(fmt_row(r))
    for ev in (1, 2, 4):
        c = [r["cagr"] for r in res if r["label"].startswith(f"every {5*ev}d")]
        print(f"  every {5*ev:>2}d: mean {np.mean(c):+.1%}  range {min(c):+.1%} .. {max(c):+.1%}")


def cmd_scores(a):
    """Different ways to combine the same model horizons into one ranking (DEV)."""
    G = inputs()
    C, mkt, cm = G["C"], G["mkt"], G["cm"]
    tags = {"h10": "h10_s5_ev_pr_ni_px_x3", "h21": "h21_s5_ev_pr_ni_px_x3", "h63": "h63_s5_ev_pr_ni_px_x3",
            "h126": "h126_s5_ev_pr_ni_px"}
    pct = lambda s: s.groupby(level="date").rank(pct=True)
    comp = {k: pct(pd.read_parquet(BASE/f"data_store/smart_ml_pred_{t}.parquet")["pred"]) for k, t in tags.items()}
    X = pd.read_parquet(BASE/"data_store/smart_features_s5.parquet", columns=["mom_riskadj"])["mom_riskadj"]
    mom = pct(X)
    ix = comp["h21"].index
    for k in comp: comp[k] = comp[k].reindex(ix)
    mom = mom.reindex(ix)
    dev = ix.get_level_values("date") <= IT.DEV_END
    D = pd.DataFrame({k: v for k, v in comp.items()}); D["mom"] = mom
    D = D[dev]
    def run(lab, score, **kw):
        out = simulate_rules(score, C, mkt, cm, end_date=IT.DEV_END, **dict(FINAL, **kw))
        st = IT.split_stats(out["nav"])["dev"]; m = out["nav"][:IT.DEV_END].resample("ME").last().pct_change().dropna()
        print(f"  {lab:<46} {st[0]:+6.1%} / {st[1]:6.1%} / {st[2]:4.2f}  up {(m > 0).mean():.0%} worst {m.min():+5.1%}  TO {out['turnover']:4.0%}", flush=True)
    ml = lambda cols: D[cols].mean(axis=1)
    print("DEV 2013-22: CAGR / maxDD / Sharpe\n  -- which model horizons? (50% momentum)")
    for lab, cols in (("live: h10+h21+h63", ["h10", "h21", "h63"]), ("h21+h63", ["h21", "h63"]), ("h63 only", ["h63"]),
                      ("h126 only", ["h126"]), ("h21+h63+h126", ["h21", "h63", "h126"]), ("all four", ["h10", "h21", "h63", "h126"]),
                      ("h10 only", ["h10"]), ("h10+h21", ["h10", "h21"])):
        run(lab, 0.5*ml(cols) + 0.5*D["mom"])
    print("  -- agreement across horizons (h10,h21,h63), 50% momentum")
    H = D[["h10", "h21", "h63"]]
    run("mean (live)", 0.5*H.mean(axis=1) + 0.5*D["mom"])
    run("minimum over horizons", 0.5*H.min(axis=1) + 0.5*D["mom"])
    run("mean - 0.5 x disagreement", 0.5*(H.mean(axis=1) - 0.5*H.std(axis=1)) + 0.5*D["mom"])
    run("geometric mean (ML) x momentum", np.sqrt(H.mean(axis=1)*D["mom"]))
    run("product of all four (ML horizons x mom)", pct((H.prod(axis=1)*D["h126"]*D["mom"]).rename("p")))
    print("  -- momentum weight with the 4-model ML")
    for w in (0.3, 0.4, 0.5, 0.6, 0.7):
        run(f"4-model ML, momentum weight {w:.0%}", (1 - w)*ml(["h10", "h21", "h63", "h126"]) + w*D["mom"])
    print("  -- take-profit rules on the live score")
    live = 0.5*ml(["h10", "h21", "h63"]) + 0.5*D["mom"]
    for lab, r in (("take profit +50%", dict(tp=0.5)), ("take profit +100%", dict(tp=1.0)), ("sell parabolic: 5d +25%", dict(spike=0.25)),
                   ("sell parabolic: 5d +40%", dict(spike=0.40))):
        run(lab, live, rules=r)


def cmd_stagger(a):
    """Split the money into sleeves that decide on different days; compare with one weekly book."""
    G = inputs(); sc, C, mkt, cm = G["sc"], G["C"], G["mkt"], G["cm"]
    cache = {}
    def sleeve(every, offset):
        k = (every, offset)
        if k not in cache:
            cache[k] = simulate_rules(sc, C, mkt, cm, end_date=IT.DEV_END, **dict(FINAL, every=every, offset=offset))["nav"][:IT.DEV_END]
        return cache[k]
    def combine(navs):
        r = pd.concat([n.pct_change() for n in navs], axis=1).dropna().mean(axis=1)   # sleeves re-equalised daily
        return (1 + r).cumprod()
    def line(lab, nav, extra=""):
        st = IT.split_stats(nav)["dev"]; m = nav.resample("ME").last().pct_change().dropna()
        print(f"  {lab:<46} {st[0]:+6.1%} / {st[1]:6.1%} / {st[2]:4.2f}  up {(m > 0).mean():.0%} worst {m.min():+5.1%}  {extra}", flush=True)
    print("DEV 2013-22 (live rules, per-stock costs): CAGR / maxDD / Sharpe")
    line("ONE weekly book (live)", sleeve(1, 0))
    for every, n_s, lab in ((2, 2, "2 sleeves, each decides every 10d (alternate weeks)"),
                            (3, 3, "3 sleeves, each decides every 15d"),
                            (4, 4, "4 sleeves, each decides every 20d"),
                            (5, 5, "5 sleeves, each decides every 25d")):
        navs = [sleeve(every, o) for o in range(n_s)]
        cg = [IT.split_stats(n)["dev"][0] for n in navs]
        line(lab, combine(navs), f"sleeves alone: {min(cg):+.1%}..{max(cg):+.1%} (mean {np.mean(cg):+.1%})")


def cmd_addiag(a):
    """When we hold a stock below our cost that the model STILL ranks highly, does it do better?"""
    G = inputs(); sc, C, mkt, cm = G["sc"], G["C"], G["mkt"], G["cm"]
    idx = C.index; mc = (1 + mkt.fillna(0)).cumprod().values
    Pv = C.ffill(limit=5).values; colpos = {c: j for j, c in enumerate(C.columns)}
    out = simulate_rules(sc, C, mkt, cm, want_trips=True, end_date=IT.DEV_END, **FINAL)
    tr = out["trips"]
    rk = sc.groupby(level="date").rank(ascending=False, method="first")
    dates = sc.index.get_level_values("date").unique()
    dpos = idx.get_indexer(dates)
    rkd = {d: g.droplevel(0) for d, g in rk.groupby(level="date")}
    rows = []
    for r in tr.itertuples():
        j = colpos[r.sym]
        for d, i in zip(dates, dpos):
            if not (r.entry_i <= i < r.exit_i - 1):
                continue
            rank = rkd[d].get(r.sym)
            p = Pv[i, j]
            if rank is None or not np.isfinite(p) or rank > 100:
                continue
            f = {}
            for h in (21, 63):
                if i + 1 + h < len(idx) and np.isfinite(Pv[i + 1, j]) and np.isfinite(Pv[i + 1 + h, j]):
                    f[h] = Pv[i + 1 + h, j]/Pv[i + 1, j] - 1 - (mc[i + 1 + h]/mc[i + 1] - 1)
            rows.append((rank, p/r.entry_px - 1, f.get(21, np.nan), f.get(63, np.nan)))
    D = pd.DataFrame(rows, columns=["rank", "ret", "f21", "f63"])
    D["rb"] = pd.cut(D["rank"], [0, 10, 20, 50, 100], labels=["rank 1-10", "11-20", "21-50", "51-100"])
    D["pb"] = pd.cut(D["ret"], [-1, -0.15, -0.05, 0.05, 0.20, 10], labels=["<-15%", "-15..-5%", "-5..+5%", "+5..+20%", ">+20%"])
    print(f"{len(D):,} (holding, week) observations from {len(tr):,} positions; excess return over the NEXT 21 / 63 days")
    for h, col in ((21, "f21"), (63, "f63")):
        print(f"\n  mean +{h}d excess by (current model rank) x (position vs our cost):  [n]")
        T = D.pivot_table(index="rb", columns="pb", values=col, aggfunc="mean", observed=True)*100
        N = D.pivot_table(index="rb", columns="pb", values=col, aggfunc="count", observed=True)
        print("  " + f"{'':<10}" + "".join(f"{c:>16}" for c in T.columns))
        for rb_ in T.index:
            print("  " + f"{rb_:<10}" + "".join(f"{T.loc[rb_, c]:>+9.2f}% [{int(N.loc[rb_, c]):>4}]" for c in T.columns))


def cmd_addon(a):
    base = dict(FINAL)
    cfgs = [dict(label="BASELINE (live)", kw=dict(base))]
    for dd, rk_, m in ((0.10, 20, 2.0), (0.10, 50, 2.0), (0.20, 20, 2.0), (0.10, 20, 1.5)):
        cfgs.append(dict(label=f"ADD to losers: <-{dd:.0%} vs cost, rank<={rk_}, x{m}", kw=dict(base, addon=dict(when="under", dd=dd, rank=rk_, mult=m))))
    for dd, rk_, m in ((0.20, 20, 2.0), (0.40, 20, 2.0)):
        cfgs.append(dict(label=f"ADD to winners: >+{dd:.0%} vs cost, rank<={rk_}, x{m}", kw=dict(base, addon=dict(when="over", dd=dd, rank=rk_, mult=m))))
    for dd in (0.10, 0.20):
        cfgs.append(dict(label=f"CUT losers: <-{dd:.0%} vs cost to half size", kw=dict(base, addon=dict(when="under", dd=dd, rank=100, mult=0.5))))
    res = run_grid(cfgs, a.workers)
    print("\nDEV 2013-2022   CAGR / maxDD / Sharpe\n")
    for r in res:
        print(fmt_row(r))


def cmd_daily(a):
    """Predictions for EVERY trading day (models trained on the same 5-day grid as v1.1):
    sanity check vs the live numbers, weekday-by-weekday luck, weekday sleeves, daily decisions."""
    G = inputs(); C, mkt, cm = G["C"], G["mkt"], G["cm"]
    pct = lambda x: x.groupby(level="date").rank(pct=True)
    ml = [pct(pd.read_parquet(BASE/f"data_store/smart_ml_pred_h{h}_s5_ev_pr_ni_x3_px_d1.parquet")["pred"]) for h in (10, 21, 63)]
    mom = pct(pd.read_parquet(BASE/"data_store/smart_features_s1.parquet", columns=["mom_riskadj"])["mom_riskadj"])
    ix = ml[0].index
    sc = 0.5*(ml[0] + ml[1].reindex(ix) + ml[2].reindex(ix))/3 + 0.5*mom.reindex(ix)
    sc = sc[sc.index.get_level_values("date") <= IT.DEV_END].dropna()
    dates = sc.index.get_level_values("date").unique()
    idx = C.index
    grid = set(idx[260::5])
    off0 = next(k for k, d in enumerate(dates) if d in grid)           # alignment of the live 5-day grid
    print(f"{len(dates)} daily decision dates ({dates[0].date()}..{dates[-1].date()}); live grid = offset {off0}\n")
    cache = {}
    def run(every, offset, **kw):
        k = (every, offset, tuple(sorted(kw.items())))
        if k not in cache:
            cache[k] = simulate_rules(sc, C, mkt, cm, end_date=IT.DEV_END, **dict(FINAL, every=every, offset=offset, **kw))
        return cache[k]
    def line(lab, nav, extra=""):
        st = IT.split_stats(nav)["dev"]; m = nav.resample("ME").last().pct_change().dropna()
        print(f"  {lab:<46} {st[0]:+6.1%} / {st[1]:6.1%} / {st[2]:4.2f}  up {(m > 0).mean():.0%} worst {m.min():+5.1%} {extra}", flush=True)
    def combine(navs):
        r = pd.concat([n[:IT.DEV_END].pct_change() for n in navs], axis=1).dropna().mean(axis=1)
        return (1 + r).cumprod()
    print("DEV 2013-22 (live rules, per-stock costs): CAGR / maxDD / Sharpe")
    out = run(5, off0); line("sanity: weekly on the live grid (live was +47.6%)", out["nav"][:IT.DEV_END], f"TO {out['turnover']:.0%}")
    print("\n  -- weekly decisions on each of the 5 weekday alignments (the live system's luck range)")
    wk = []
    for o in range(5):
        r_ = run(5, (off0 + o) % 5 if False else off0 + o)
        wk.append(r_["nav"][:IT.DEV_END]); line(f"weekly, alignment +{o}", wk[-1], f"TO {r_['turnover']:.0%}")
    cg = [IT.split_stats(n)["dev"][0] for n in wk]
    print(f"  weekly: mean {np.mean(cg):+.1%}  range {min(cg):+.1%} .. {max(cg):+.1%}")
    print("\n  -- weekday sleeves (each sleeve decides WEEKLY on its own weekday; capital split equally)")
    line("2 sleeves (alignments +0,+2)", combine([wk[0], wk[2]]))
    line("3 sleeves (alignments +0,+2,+4)", combine([wk[0], wk[2], wk[4]]))
    line("5 sleeves (every weekday)", combine(wk))
    print("\n  -- how often to decide? (one book)")
    for ev in (1, 2, 3):
        r_ = run(ev, 0); line(f"decide every {ev} trading day(s)", r_["nav"][:IT.DEV_END], f"TO {r_['turnover']:.0%}")
    r_ = run(1, 0, exit_rank=150); line("daily, exit rank 150", r_["nav"][:IT.DEV_END], f"TO {r_['turnover']:.0%}")
    r_ = run(1, 0, exit_rank=200, band=0.5); line("daily, exit rank 200, band 50%", r_["nav"][:IT.DEV_END], f"TO {r_['turnover']:.0%}")


def cmd_dailygrid(a):
    """Is 'decide daily + wide exit band' a plateau or a lucky pick? Neighbours + all alignments."""
    pct = lambda x: x.groupby(level="date").rank(pct=True)
    ml = [pct(pd.read_parquet(BASE/f"data_store/smart_ml_pred_h{h}_s5_ev_pr_ni_x3_px_d1.parquet")["pred"]) for h in (10, 21, 63)]
    mom = pct(pd.read_parquet(BASE/"data_store/smart_features_s1.parquet", columns=["mom_riskadj"])["mom_riskadj"])
    ix = ml[0].index
    sc = 0.5*(ml[0] + ml[1].reindex(ix) + ml[2].reindex(ix))/3 + 0.5*mom.reindex(ix)
    sc = sc[sc.index.get_level_values("date") <= IT.DEV_END].dropna()
    f = BASE/"data_store/smart_exit_daily_sc.pkl"; pickle.dump(sc, open(f, "wb"))
    base = dict(FINAL)
    cfgs = []
    for er in (100, 150, 200, 300):
        for bd in (0.25, 0.5, 1.0):
            cfgs.append(dict(label=f"DAILY exit rank {er}, band {bd:.0%}", scfile=str(f), kw=dict(base, every=1, exit_rank=er, band=bd)))
    for o in range(5):
        cfgs.append(dict(label=f"WEEKLY align +{o}, exit 200, band 50%", scfile=str(f), kw=dict(base, every=5, offset=o, exit_rank=200, band=0.5)))
    for o in range(5):
        cfgs.append(dict(label=f"WEEKLY align +{o}, exit 100, band 25% (live)", scfile=str(f), kw=dict(base, every=5, offset=o)))
    for ev in (2, 3):
        cfgs.append(dict(label=f"every {ev}d, exit 200, band 50%", scfile=str(f), kw=dict(base, every=ev, exit_rank=200, band=0.5)))
    res = run_grid(cfgs, a.workers)
    print("\nDEV 2013-2022, daily-prediction models, Rs2L per-stock costs   CAGR / maxDD / Sharpe\n")
    for r in res:
        print(fmt_row(r))
    for key in ("WEEKLY align +", ):
        for tag in ("exit 200, band 50%", "exit 100, band 25% (live)"):
            c = [r["cagr"] for r in res if r["label"].startswith(key) and tag in r["label"]]
            sh = [r["sh"] for r in res if r["label"].startswith(key) and tag in r["label"]]
            print(f"  weekly, {tag:<26} mean {np.mean(c):+.1%} (range {min(c):+.1%}..{max(c):+.1%})  mean Sharpe {np.mean(sh):.2f}")
    pickle.dump(res, open(BASE/"data_store/smart_dailygrid_results.pkl", "wb"))


WIN = {"A 2013-2020": ("2013-01-01", "2020-12-31"), "B 2021-2026": ("2021-01-01", "2026-12-31")}
COMP = BASE/"data_store/smart_split_components.pkl"
_CP = {}


def split_cfg(cfg):
    """One run: weekly decisions at alignment `off`, blended score weight `w`, over window `win`."""
    G = inputs()
    if "c" not in _CP:
        _CP["c"] = pickle.load(open(COMP, "rb"))
    ml, mom = _CP["c"]["ml"], _CP["c"]["mom"]
    a0, a1 = pd.Timestamp(cfg["start"]), pd.Timestamp(cfg["end"])
    d = ml.index.get_level_values("date")
    keep = (d >= a0) & (d <= a1)
    sc = ((1 - cfg["w"])*ml[keep] + cfg["w"]*mom.reindex(ml.index[keep])).dropna()
    out = simulate_rules(sc, G["C"], G["mkt"], G["cm"], end_date=a1, **dict(FINAL, every=5, offset=cfg["off"], **cfg["kw"]))
    nav = out["nav"][:a1]
    return dict(label=cfg["label"], win=cfg["win"], off=cfg["off"], nav=nav.astype("float32"), to=out["turnover"])


def nstats(nav):
    nav = nav/nav.iloc[0]
    yrs = (nav.index[-1] - nav.index[0]).days/365.25
    r = nav.pct_change().dropna()
    return dict(cagr=nav.iloc[-1]**(1/yrs) - 1, dd=float((nav/nav.cummax() - 1).min()), sh=float(r.mean()/r.std()*np.sqrt(252)))


def cmd_split(a):
    pct = lambda x: x.groupby(level="date").rank(pct=True)
    ml = [pct(pd.read_parquet(BASE/f"data_store/smart_ml_pred_h{h}_s5_ev_pr_ni_x3_px_d1.parquet")["pred"]) for h in (10, 21, 63)]
    ix = ml[0].index
    mlx = (ml[0] + ml[1].reindex(ix) + ml[2].reindex(ix))/3
    mom = pct(pd.read_parquet(BASE/"data_store/smart_features_s1.parquet", columns=["mom_riskadj"])["mom_riskadj"])
    pickle.dump(dict(ml=mlx, mom=mom), open(COMP, "wb"))
    grid = [("LIVE: top20 exit100 band.25 corr.5 mom50%", {}, 0.5)]
    for er in (50, 150, 200, 300):
        grid.append((f"exit rank {er}", dict(exit_rank=er), 0.5))
    for n_, er in ((10, 50), (15, 75), (30, 150), (40, 200)):
        grid.append((f"{n_} stocks (exit rank {er})", dict(top_n=n_, exit_rank=er), 0.5))
    for w in (0.0, 0.25, 0.75, 1.0):
        grid.append((f"momentum weight {w:.0%}", {}, w))
    grid.append(("no correlation cap", dict(max_corr=None), 0.5))
    grid.append(("band 0 (rebalance fully)", dict(band=0.0), 0.5))
    grid.append(("band 50%", dict(band=0.5), 0.5))
    cfgs = [dict(label=l, kw=kw, w=w, win=wn, start=st, end=en, off=o)
            for l, kw, w in grid for wn, (st, en) in WIN.items() for o in range(5)]
    # full-period series for the period table (live / ML only / momentum only)
    full = [dict(label=l, kw={}, w=w, win="FULL", start="2013-01-01", end="2026-12-31", off=o)
            for l, w in (("smart", 0.5), ("ML only", 0.0), ("momentum", 1.0)) for o in range(5)]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as ex:
        res = list(ex.map(split_cfg, cfgs + full))
    print(f"({len(res)} runs in {time.time() - t0:.0f}s)")
    R = pd.DataFrame([dict(label=r["label"], win=r["win"], off=r["off"], **nstats(r["nav"]), to=r["to"]) for r in res[:len(cfgs)]])
    M = R.groupby(["label", "win"]).agg(cagr=("cagr", "mean"), lo=("cagr", "min"), hi=("cagr", "max"), dd=("dd", "mean"), sh=("sh", "mean")).unstack("win")
    order = [g[0] for g in grid]
    print("\nRULES CHOSEN ON ONE PERIOD, TESTED ON THE OTHER (mean over 5 weekday alignments; per-stock costs Rs2L)")
    print(f"{'':<46} {'A 2013-2020':^38} {'B 2021-Oct 2026':^38}")
    print(f"{'':<46} {'CAGR (min..max)':>22} {'DD':>7} {'Sh':>5}   {'CAGR (min..max)':>22} {'DD':>7} {'Sh':>5}")
    for l in order:
        row = M.loc[l]
        cell = lambda w: f"{row[('cagr', w)]:+6.1%} ({row[('lo', w)]:+5.0%}..{row[('hi', w)]:+4.0%})  {row[('dd', w)]:6.1%} {row[('sh', w)]:5.2f}"
        print(f"  {l:<44} {cell('A 2013-2020')}   {cell('B 2021-2026')}")
    bA = M[("sh", "A 2013-2020")].idxmax(); bB = M[("sh", "B 2021-2026")].idxmax()
    live = grid[0][0]
    print(f"\n  best on A by Sharpe: {bA}  -> on B: {M.loc[bA, ('cagr', 'B 2021-2026')]:+.1%} / Sh {M.loc[bA, ('sh', 'B 2021-2026')]:.2f}")
    print(f"  best on B by Sharpe: {bB}  -> on A: {M.loc[bB, ('cagr', 'A 2013-2020')]:+.1%} / Sh {M.loc[bB, ('sh', 'A 2013-2020')]:.2f}")
    print(f"  live config:  A {M.loc[live, ('cagr', 'A 2013-2020')]:+.1%} / Sh {M.loc[live, ('sh', 'A 2013-2020')]:.2f}   "
          f"B {M.loc[live, ('cagr', 'B 2021-2026')]:+.1%} / Sh {M.loc[live, ('sh', 'B 2021-2026')]:.2f}")
    # ---- period table: average-alignment NAV for smart / ML / momentum vs indices
    F = {}
    for r in res[len(cfgs):]:
        F.setdefault(r["label"], []).append(r["nav"].pct_change())
    nav = {k: (1 + pd.concat(v, axis=1).mean(axis=1).fillna(0)).cumprod() for k, v in F.items()}
    ix_ = pd.read_parquet(BASE/"data_store/nse_index.parquet")
    def idx_(nm):
        return ix_[ix_["index"].str.lower() == nm.lower()].set_index("date")["close"].sort_index()
    G = inputs()
    nav["equal-wt 800"] = (1 + G["mkt"].fillna(0)).cumprod()
    for nm in ("Nifty 500", "Nifty Midcap 150"):
        nav[nm] = idx_(nm)
    periods = [("2013-2016", "2013-01-01", "2016-12-31"), ("2017-2020", "2017-01-01", "2020-12-31"),
               ("2021-Oct 2026", "2021-01-01", "2026-12-31"), ("2023-Oct 2026", "2023-01-01", "2026-12-31"),
               ("2025-Oct 2026", "2025-01-01", "2026-12-31")]
    print("\nPERIOD TABLE (average over the 5 weekday alignments)  CAGR / maxDD / Sharpe")
    print(f"  {'':<18}" + "".join(f"{n:>30}" for n, _, _ in periods))
    for k in ("smart", "ML only", "momentum", "equal-wt 800", "Nifty 500", "Nifty Midcap 150"):
        cells = []
        for n, s0, s1 in periods:
            x = nav[k].loc[s0:s1].dropna()
            if len(x) < 100 or x.index[0] > pd.Timestamp(s0) + pd.Timedelta(days=20):
                cells.append(f"{'n/a':>30}"); continue
            st = nstats(x); cells.append(f"{st['cagr']:>+11.1%} {st['dd']:>8.1%} {st['sh']:>7.2f}   ")
        print(f"  {k:<18}" + "".join(cells))
    yr = pd.DataFrame({k: nav[k].resample("YE").last().pct_change()*100 for k in ("smart", "ML only", "momentum", "equal-wt 800", "Nifty 500")})
    yr.iloc[0] = [ (nav[k].resample("YE").last().iloc[0]/nav[k].iloc[0] - 1)*100 for k in yr.columns]
    print("\nBY YEAR (%):\n" + yr.round(1).rename(index=lambda d: d.year).to_string())
    pickle.dump(dict(nav=nav, R=R), open(BASE/"data_store/smart_split_results.pkl", "wb"))


SIGF = BASE/"data_store/smart_overlay_signals.pkl"
CASH = 0.065            # liquid-fund yield assumed for uninvested money (approximation: 3.5-8% over the years)


def make_expo(spec, SG):
    """Exposure rule -> Series (known at each close) or callable(i, nav_list). Levels are 0 / .25 / .5 / .75 / 1."""
    if spec is None:
        return None
    kind = spec[0]
    if kind == "breadth":                       # market breadth: share of the universe above its 200DMA
        v = pd.Series(1.0, index=SG.index)
        for thr, lvl in sorted(spec[1], reverse=True):      # e.g. [(0.35, .5), (0.25, 0.)]: deepest threshold wins
            v[SG["breadth"] < thr] = lvl
        return v
    if kind == "mkt":                           # equal-weight market vs its own moving average
        ma, lvl = spec[1], spec[2]
        v = pd.Series(1.0, index=SG.index); v[SG[f"below{ma}"]] = lvl
        if len(spec) > 3:                       # deeper step: below MA AND 63d return negative
            v[SG[f"below{ma}"] & (SG["r63"] < 0)] = spec[3]
        return v
    if kind == "sig_lt":                        # any precomputed signal column: below thr -> level (deepest wins)
        v = pd.Series(1.0, index=SG.index)
        for thr, lvl in sorted(spec[2], reverse=True):
            v[SG[spec[1]] < thr] = lvl
        return v
    if kind == "combo":                         # minimum of several rules (Series and/or callables)
        parts = [make_expo(x, SG) for x in spec[1:]]
        def f(i, nav):
            return min(float(p_(i, nav)) if callable(p_) else float(p_.iloc[i]) for p_ in parts)
        return f
    if kind == "voltarget":                     # scale to a target volatility of the BOOK itself (trailing 63d)
        tgt = spec[1]
        def f(i, nav):
            if len(nav) < 70: return 1.0
            r = np.diff(np.log(np.array(nav[-64:], dtype=float)))
            vol = r.std()*np.sqrt(252)
            return float(np.clip(np.floor(tgt/max(vol, 1e-6)*4)/4, 0.0, 1.0))
        return f
    if kind == "dd":                            # equity-curve filter: drawdown of the book from its peak
        steps = sorted(spec[1], reverse=True)
        def f(i, nav):
            a = np.array(nav, dtype=float); dd = a[-1]/a.max() - 1; lvl = 1.0
            for thr, l in steps:
                if dd <= -thr: lvl = l
            return lvl
        return f
    if kind == "navsma":                        # book NAV below its own n-day average
        n, lvl = spec[1], spec[2]
        def f(i, nav):
            if len(nav) < n: return 1.0
            a = np.array(nav[-n:], dtype=float); return lvl if a[-1] < a.mean() else 1.0
        return f
    raise ValueError(kind)


def ov_cfg(cfg):
    G = inputs()
    if "c" not in _CP:
        _CP["c"] = pickle.load(open(COMP, "rb")); _CP["sig"] = pickle.load(open(SIGF, "rb"))
    ml, mom, SG = _CP["c"]["ml"], _CP["c"]["mom"], _CP["sig"]
    a0, a1 = pd.Timestamp(cfg["start"]), pd.Timestamp(cfg["end"])
    d = ml.index.get_level_values("date"); keep = (d >= a0) & (d <= a1)
    mlk = ml[keep]; momk = mom.reindex(mlk.index)
    if cfg.get("wreg"):                          # momentum weight depends on the market state
        flag, w_bad, w_good = cfg["wreg"]
        wv = np.where(SG[flag].reindex(d[keep]).fillna(False).values, w_bad, w_good)
    else:
        wv = cfg["w"]
    sc = ((1 - wv)*mlk + wv*momk).dropna()
    out = simulate_rules(sc, G["C"], G["mkt"], G["cm"], end_date=a1, expo=make_expo(cfg.get("expo"), SG), cash_r=CASH,
                         **dict(FINAL, every=5, offset=cfg["off"]))
    return dict(label=cfg["label"], win=cfg["win"], off=cfg["off"], nav=out["nav"][:a1].astype("float32"),
                expo=out["exposure"], sw=out["switches"], to=out["turnover"])


def cmd_overlay(a):
    C, mkt = inputs()["C"], inputs()["mkt"]
    idx = C.index; mc = (1 + mkt.fillna(0)).cumprod()
    SG = pd.DataFrame(index=idx)
    SG["below200"] = (mc < mc.rolling(200).mean()).fillna(False)
    SG["below100"] = (mc < mc.rolling(100).mean()).fillna(False)
    SG["r63"] = (mc/mc.shift(63) - 1).fillna(0)
    f1 = pd.read_parquet(BASE/"data_store/smart_features_s1.parquet", columns=["m_breadth"]).groupby(level="date").first()["m_breadth"]
    SG["breadth"] = f1.reindex(idx).ffill().fillna(0.5)
    SG["b35"] = SG["breadth"] < 0.35
    for t_ in (25, 30, 40, 45, 50):
        SG[f"b{t_}"] = SG["breadth"] < t_/100
    if SIGF.exists():                          # pick-quality signals (cand_strong, picks_trail) were added by hand to this file
        old = pickle.load(open(SIGF, "rb"))
        for c in ("cand_strong", "picks_trail"):
            if c in old.columns: SG[c] = old[c]
    pickle.dump(SG, open(SIGF, "wb"))
    pct = lambda x: x.groupby(level="date").rank(pct=True)
    if not COMP.exists():
        raise SystemExit("run `split` first (builds the score components)")
    grid = [("BASELINE (live): always 100% invested", None, None, 0.5),
            ("breadth<35% -> 50%", ("breadth", [(0.35, 0.5)]), None, 0.5),
            ("breadth<30% -> 0% (cash)", ("breadth", [(0.30, 0.0)]), None, 0.5),
            ("breadth<35% -> 50%, <25% -> 0%", ("breadth", [(0.35, 0.5), (0.25, 0.0)]), None, 0.5),
            ("breadth<40% -> 50%", ("breadth", [(0.40, 0.5)]), None, 0.5),
            ("market<200DMA -> 50%", ("mkt", 200, 0.5), None, 0.5),
            ("market<200DMA -> 0% (cash)", ("mkt", 200, 0.0), None, 0.5),
            ("market<100DMA -> 50%", ("mkt", 100, 0.5), None, 0.5),
            ("market<200DMA 50%; +63d<0 -> 0%", ("mkt", 200, 0.5, 0.0), None, 0.5),
            ("book vol-target 20%", ("voltarget", 0.20), None, 0.5),
            ("book vol-target 25%", ("voltarget", 0.25), None, 0.5),
            ("book vol-target 30%", ("voltarget", 0.30), None, 0.5),
            ("book drawdown>12% -> 50%", ("dd", [(0.12, 0.5)]), None, 0.5),
            ("book DD>12% 50%, >20% 0%", ("dd", [(0.12, 0.5), (0.20, 0.0)]), None, 0.5),
            ("book below 60d avg -> 50%", ("navsma", 60, 0.5), None, 0.5),
            ("combo: mkt<200DMA 50% + breadth<25% 0%", ("combo", ("mkt", 200, 0.5), ("breadth", [(0.25, 0.0)])), None, 0.5),
            ("combo: breadth<35% 50% + book DD>20% 0%", ("combo", ("breadth", [(0.35, 0.5)]), ("dd", [(0.20, 0.0)])), None, 0.5),
            ("PICKS WORKING? trailing top-20 excess < 0.4% -> 50%", ("sig_lt", "picks_trail", [(0.004, 0.5)]), None, 0.5),
            ("PICKS WORKING? <0% -> 50%, < -1% -> 0%", ("sig_lt", "picks_trail", [(0.0, 0.5), (-0.01, 0.0)]), None, 0.5),
            ("PICKS WORKING? < -0.5% -> 0% (cash)", ("sig_lt", "picks_trail", [(-0.005, 0.0)]), None, 0.5),
            ("CANDIDATES STRONG? <50% of top-30 in uptrend -> 50%", ("sig_lt", "cand_strong", [(0.50, 0.5)]), None, 0.5),
            ("CANDIDATES STRONG? <50% -> 50%, <30% -> 0%", ("sig_lt", "cand_strong", [(0.50, 0.5), (0.30, 0.0)]), None, 0.5),
            ("CANDIDATES STRONG? <30% -> 0% (cash)", ("sig_lt", "cand_strong", [(0.30, 0.0)]), None, 0.5),
            ("combo: picks<0% OR candidates<50% -> 50%", ("combo", ("sig_lt", "picks_trail", [(0.0, 0.5)]), ("sig_lt", "cand_strong", [(0.50, 0.5)])), None, 0.5),
            ("combo: picks<-0.5% AND cand<50% -> 0%, either -> 50%", ("combo", ("sig_lt", "picks_trail", [(0.0, 0.5)]), ("sig_lt", "cand_strong", [(0.50, 0.5)]), ("sig_lt", "picks_trail", [(-0.005, 0.5)])), None, 0.5),
            ("NO CASH: momentum weight 25% when market<200DMA", None, ("below200", 0.25, 0.5), 0.5),
            ("NO CASH: ML-only when market<200DMA", None, ("below200", 0.0, 0.5), 0.5),
            ("NO CASH: ML-only when breadth<35%", None, ("b35", 0.0, 0.5), 0.5)]
    if a.which == "regime":                 # recipe switching by market state (no cash): robustness of the winning family
        grid = [grid[0], ("momentum weight 75% always", None, None, 0.75)]
        for t_ in (25, 30, 35, 40, 45, 50):
            for wb in (0.0, 0.25):
                for wg in (0.5, 0.75):
                    grid.append((f"breadth<{t_}%: momentum {wb:.0%} (else {wg:.0%})", None, (f"b{t_}", wb, wg), 0.5))
    cfgs = [dict(label=l, expo=e, wreg=wr, w=w, win=wn, start=st, end=en, off=o)
            for l, e, wr, w in grid for wn, (st, en) in WIN.items() for o in range(5)]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as ex:
        res = list(ex.map(ov_cfg, cfgs))
    print(f"({len(res)} runs in {time.time() - t0:.0f}s)   cash earns {CASH:.1%}; overlay decided at weekly closes, executed next close with costs\n")
    rows = []
    for r in res:
        st = nstats(r["nav"]); n = r["nav"]
        yrs = {}
        for y in (2018, 2020, 2022, 2025, 2026):
            if n.index[0] <= pd.Timestamp(f"{y}-01-05") and n.index[-1] >= pd.Timestamp(f"{y}-12-20" if y < 2026 else "2026-09-25"):
                a0_ = n.loc[:f"{y-1}-12-31"]; a0_ = a0_.iloc[-1] if len(a0_) else n.iloc[0]
                yrs[y] = float(n.loc[:f"{y}-12-31"].iloc[-1]/a0_ - 1)*100
        rows.append(dict(label=r["label"], win=r["win"], expo=r["expo"], sw=r["sw"], **st, **{f"y{k}": v for k, v in yrs.items()}))
    R = pd.DataFrame(rows)
    M = R.groupby(["label", "win"]).mean(numeric_only=True)
    order = [g[0] for g in grid]
    print(f"{'':<50}{'A 2013-2020: CAGR / DD / Sh':^30}{'invested':>9}   {'B 2021-Oct26: CAGR / DD / Sh':^30}{'invested':>9}   2018 | 2020 | 2022 | 2025 | 2026*")
    for l in order:
        A_, B_ = M.loc[(l, "A 2013-2020")], M.loc[(l, "B 2021-2026")]
        ys = " ".join(f"{(A_.get('y2018') if y == 2018 else A_.get('y2020') if y == 2020 else B_.get(f'y{y}')):>+5.0f}" for y in (2018, 2020, 2022, 2025, 2026))
        print(f"  {l:<48}{A_['cagr']*100:>+8.1f}% {A_['dd']*100:>7.1f}% {A_['sh']:>5.2f}   {A_['expo']:>6.0%}   {B_['cagr']*100:>+8.1f}% {B_['dd']*100:>7.1f}% {B_['sh']:>5.2f}   {B_['expo']:>6.0%}   {ys}")
    # the user's criterion: what does each rule COST in strong years and what does it SAVE in weak years?
    def yearly(n):
        out = {}
        for y in range(2013, 2027):
            if n.index[0] <= pd.Timestamp(f"{y}-01-05") and n.index[-1] >= pd.Timestamp(f"{y}-12-20" if y < 2026 else "2026-09-25"):
                a0_ = n.loc[:f"{y-1}-12-31"]; a0_ = a0_.iloc[-1] if len(a0_) else n.iloc[0]
                out[y] = float(n.loc[:f"{y}-12-31"].iloc[-1]/a0_ - 1)*100
        return out
    Yr = {}
    for lab, win, off, nv in [(r["label"], r["win"], r["off"], r["nav"]) for r in res]:
        Yr.setdefault(lab, {}).setdefault(off, {}).update(yearly(nv))
    Ym = {lab: pd.DataFrame(v).mean(axis=1) for lab, v in Yr.items()}
    base = Ym[grid[0][0]]
    strong = base[base >= 40].index; weak = base[base < 15].index
    print(f"\nWHAT EACH RULE COSTS IN STRONG YEARS vs SAVES IN WEAK YEARS  (strong = baseline >= +40%: {list(strong)};  weak = baseline < +15%: {list(weak)})")
    print(f"  {'':<56}{'strong yrs: mean change':>24}{'weak yrs: mean change':>24}   net over all 14 yrs")
    for lab in order[1:]:
        d = Ym[lab] - base
        print(f"  {lab:<56}{d[strong].mean():>+22.1f}pp{d[weak].mean():>+22.1f}pp   {d.mean():>+6.1f}pp/yr")
    pickle.dump(dict(R=R, Ym=Ym, res=[(r["label"], r["win"], r["off"], r["nav"]) for r in res]), open(BASE/"data_store/smart_overlay_results.pkl", "wb"))


def cmd_gates(a):
    base = dict(FINAL)
    cfgs = [dict(label="BASELINE (live rules)", kw=dict(base))]
    for g in ("sma50", "trend", "mom", "rs", "near_high", "strong"):
        cfgs.append(dict(label=f"gate {g}: entries only, cash 6.5%", kw=dict(base, gate=g, min_slots=20, depth=15, cash_r=0.065)))
        cfgs.append(dict(label=f"gate {g}: entries+holds, cash 6.5%", kw=dict(base, gate=g, gate_hold=True, min_slots=20, depth=15, cash_r=0.065)))
    for g in ("trend", "strong"):
        cfgs.append(dict(label=f"gate {g}: entries+holds, park in market", kw=dict(base, gate=g, gate_hold=True, min_slots=20, depth=15, park=True)))
        cfgs.append(dict(label=f"gate {g}: entries+holds, >=10 names concentrate", kw=dict(base, gate=g, gate_hold=True, min_slots=10, depth=15)))
    res = run_grid(cfgs, a.workers)
    print("\nDEV 2013-2022, Rs2L per-stock costs   CAGR / maxDD / Sharpe\n")
    for r in res:
        print(fmt_row(r))
    pickle.dump(res, open(BASE/"data_store/smart_gate_results.pkl", "wb"))


def cmd_diag(a):
    G = inputs()
    sc, C, mkt, cm = G["sc"], G["C"], G["mkt"], G["cm"]
    idx = C.index
    mc = (1 + mkt.fillna(0)).cumprod().values
    Pv = C.ffill(limit=5).values
    colpos = {c: j for j, c in enumerate(C.columns)}
    dates = sc.index.get_level_values("date").unique()

    # ---- 1. what is each score rank worth?
    rk = sc.groupby(level="date").rank(ascending=False, method="first")
    labels = ["1-5", "6-10", "11-20", "21-40", "41-100", "101-200", "201-400", "401+"]
    b = pd.cut(rk, bins=[0, 5, 10, 20, 40, 100, 200, 400, 10000], labels=labels)
    print("== 1. forward return by blended-score rank (DEV; market-adjusted; entry = next close) ==")
    print(f"{'rank':<9}" + "".join(f"{'+%dd excess / hit%%' % h:>22}" for h in (5, 10, 21, 63)))
    tab = {}
    f21 = None
    for h in (5, 10, 21, 63):
        y = SM.forward_returns(C.astype("float64"), dates, h)
        d = y.index.get_level_values("date"); p = idx.get_indexer(d)
        m = mc[np.minimum(p + 1 + h, len(idx) - 1)]/mc[p + 1] - 1
        df = pd.DataFrame({"raw": y.values, "ex": y.values - m, "b": b.reindex(y.index).values}, index=y.index).dropna(subset=["b"])
        g = df.groupby("b", observed=True)
        tab[h] = pd.DataFrame({"ex": g["ex"].mean(), "hit": g["ex"].apply(lambda s: (s > 0).mean()), "raw": g["raw"].mean()})
        if h == 21: f21 = df.copy()
    for lab in labels:
        print(f"{lab:<9}" + "".join(f"{tab[h].loc[lab, 'ex']*100:>+14.2f}% /{tab[h].loc[lab, 'hit']*100:>4.0f}%" for h in (5, 10, 21, 63)))
    print("  (a swap costs roughly 0.4-1.2% round trip: a bucket must beat the one it replaces by more than that)")

    print("\n== 2. does the top of the list still earn money when the market is weak? (+21d) ==")
    reg = G["regime"].reindex(dates)
    br = pd.qcut(reg["m_breadth"], 3, labels=["weak breadth", "mid breadth", "strong breadth"])
    f21["date"] = f21.index.get_level_values("date")
    f21["breadth"] = br.reindex(f21["date"]).values
    f21["m63"] = np.where(reg["m_r63"].reindex(f21["date"]).values < 0, "market 63d DOWN", "market 63d UP")
    f21["top"] = np.where(b.reindex(f21.index).values == "1-5", "1-5", np.where(b.reindex(f21.index).isin(["6-10", "11-20"]), "6-20", "other"))
    for col in ("breadth", "m63"):
        print(f"  by {col}:")
        for k, g in f21[f21["top"] != "other"].groupby([col, "top"], observed=True):
            print(f"    {k[0]:<16} ranks {k[1]:<5} raw {g['raw'].mean()*100:+6.2f}%  excess {g['ex'].mean()*100:+6.2f}%  hit {(g['ex'] > 0).mean()*100:3.0f}%  n={len(g):,}")

    # ---- baseline simulation with round trips
    out = simulate_rules(sc, C, mkt, cm, want_trips=True, end_date=IT.DEV_END, **FINAL)
    tr = out["trips"].copy()
    tr["ret"] = tr["exit_px"]/tr["entry_px"] - 1
    tr["days"] = tr["exit_i"] - tr["entry_i"]
    tr["mae"] = tr["trough"]/tr["entry_px"] - 1
    tr["mfe"] = tr["peak"]/tr["entry_px"] - 1
    print(f"\n== 3. the {len(tr):,} round trips of the live rules (DEV) ==")
    w_, l_ = tr[tr.ret > 0], tr[tr.ret <= 0]
    print(f"  hold days: median {tr.days.median():.0f}, mean {tr.days.mean():.0f}, 10th/90th pct {tr.days.quantile(.1):.0f}/{tr.days.quantile(.9):.0f}")
    print(f"  win rate {len(w_)/len(tr):.0%}; avg win {w_.ret.mean():+.1%}, avg loss {l_.ret.mean():+.1%}; median trip {tr.ret.median():+.1%}")
    srt = tr.sort_values("ret", ascending=False)
    tot = tr.ret.clip(lower=-1).sum()
    print(f"  profit concentration: best 10% of trips contribute {srt.ret.head(len(tr)//10).sum()/tot:.0%} of the summed return; worst 10% lose {srt.ret.tail(len(tr)//10).sum()/tot:+.0%}")
    print(f"  exit reasons: {tr.reason.value_counts().to_dict()}")
    print("  worst-excursion vs outcome (MAE = deepest point below entry):")
    mb = pd.cut(tr.mae, [-1, -0.25, -0.15, -0.10, -0.05, 0.0001], labels=["< -25%", "-25..-15%", "-15..-10%", "-10..-5%", "> -5%"])
    for k, g in tr.groupby(mb, observed=True):
        print(f"    MAE {k:<10} n={len(g):>5} ({len(g)/len(tr):4.0%})  final return avg {g.ret.mean():+6.1%}  finished above entry {(g.ret > 0).mean():3.0f}%".replace("3.0f", "") if False else
              f"    MAE {k:<10} n={len(g):>5} ({len(g)/len(tr):4.0%})  final return avg {g.ret.mean():+6.1%}  finished above entry {(g.ret > 0).mean()*100:3.0f}%")

    # ---- 4. net value of the weekly swaps
    print("\n== 4. what the weekly SWAPS earn (bought vs sold the same week; from that close) ==")
    sells = tr[(tr.reason == "rank")]
    buys = tr
    def fwd(i, sym, h):
        j = colpos[sym]
        if i + h >= len(idx): return np.nan
        a_, b_ = Pv[i, j], Pv[i + h, j]
        if not (np.isfinite(a_) and np.isfinite(b_) and a_ > 0): return np.nan
        return b_/a_ - 1 - (mc[i + h]/mc[i] - 1)
    for h in (5, 10, 21, 63):
        fb = np.array([fwd(i, s, h) for i, s in zip(buys.entry_i, buys.sym)])
        fs = np.array([fwd(i, s, h) for i, s in zip(sells.exit_i, sells.sym)])
        print(f"  +{h:>2}d: bought {np.nanmean(fb)*100:+6.2f}%   sold {np.nanmean(fs)*100:+6.2f}%   edge per swap {(np.nanmean(fb) - np.nanmean(fs))*100:+6.2f}%")
    cb = np.nanmean([cm.iloc[i].get(s, np.nan) for i, s in zip(buys.entry_i, buys.sym)])
    cs = np.nanmean([cm.iloc[i].get(s, np.nan) for i, s in zip(sells.exit_i, sells.sym)])
    print(f"  one-way cost: buys {cb*100:.2f}%  sells {cs*100:.2f}%  -> a swap must earn > {(cb + cs)*100:.2f}% to pay for itself")

    # ---- 5. after a stock falls X% (from entry / from peak): what happens next?
    print("\n== 5. after a position falls, does it keep falling? (hypothetical: keep holding, +21d from next close) ==")
    base_all = []
    for r in tr.itertuples():
        for i in range(r.entry_i + 5, r.exit_i, 5):
            v = fwd(i + 1, r.sym, 21)
            if np.isfinite(v): base_all.append(v)
    print(f"  baseline: every held stock-week          +21d excess {np.mean(base_all)*100:+.2f}%  (n={len(base_all):,})")
    for kind, ths in (("below ENTRY", (0.08, 0.12, 0.18, 0.25)), ("below PEAK", (0.12, 0.20, 0.30))):
        for th in ths:
            f5, f21_, fin = [], [], []
            for r in tr.itertuples():
                j = colpos[r.sym]
                path = Pv[r.entry_i: r.exit_i + 1, j]
                ref = np.full(len(path), path[0]) if kind == "below ENTRY" else np.fmax.accumulate(path)
                hit = np.where(path/ref - 1 <= -th)[0]
                if len(hit):
                    i = r.entry_i + hit[0]
                    f5.append(fwd(i + 1, r.sym, 5)); f21_.append(fwd(i + 1, r.sym, 21)); fin.append(r.ret)
            f5, f21_, fin = np.array(f5), np.array(f21_), np.array(fin)
            print(f"  first time {th:.0%} {kind:<11} n={len(f21_):>5}  next +5d {np.nanmean(f5)*100:+6.2f}%  next +21d {np.nanmean(f21_)*100:+6.2f}%  trip ended {np.nanmean(fin)*100:+6.1f}% avg")
    pickle.dump(tr, open(BASE/"data_store/smart_exit_trips.pkl", "wb"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "diag", "exits", "gates", "timing", "scores", "stagger", "addiag", "addon", "daily", "dailygrid", "split", "overlay"])
    ap.add_argument("--capital", type=float, default=2e5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--which", default="main")
    a = ap.parse_args()
    {"prep": prep, "diag": cmd_diag, "exits": cmd_exits, "gates": cmd_gates, "timing": cmd_timing, "scores": cmd_scores,
     "stagger": cmd_stagger, "addiag": cmd_addiag, "addon": cmd_addon, "daily": cmd_daily, "dailygrid": cmd_dailygrid, "split": cmd_split, "overlay": cmd_overlay}[a.cmd](a)
