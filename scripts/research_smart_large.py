"""
Large-cap SPECIALIST sleeve (user, 2026-10-04: "pick the best stocks at that time" - in 2018 and 2025 small caps
crashed while large caps held up, and the general model is weak in large caps).

Specialist = the same pipeline trained ONLY on the 300 most liquid stocks (ranks and targets within that set;
research_smart_ml.py --liq-max 300). Questions, judged on A=2013-2020 and tested on B=2021-Oct 2026, mean of 5
weekday alignments, per-stock costs:
  1  does specialisation help INSIDE large caps? (general model restricted to the same names vs the specialist)
  2  does a large-cap sleeve beside the main book improve the total? (fixed mixes, daily-rebalanced, no timing)
  3  what does it do in the bad years (2018, 2022, 2025, 2026 YTD) and in the market falls?

Run:  python scripts/research_smart_large.py
"""
from __future__ import annotations
import multiprocessing as mp, pickle, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))
from scripts import research_smart_exit as E          # noqa: E402
from scripts import research_smart_verify as V        # noqa: E402

LC = BASE/"data_store/smart_large_components.pkl"
_S = {}
VARIANTS = [  # (label, universe/model, momentum weight, top_n, exit_rank)
    ("general model in the same 300 names, 50/50", "general", 0.5, 20, 60),
    ("SPECIALIST ML only, top 20", "spec", 0.0, 20, 60),
    ("SPECIALIST ML only, top 15", "spec", 0.0, 15, 45),
    ("SPECIALIST 50/50 with momentum, top 20", "spec", 0.5, 20, 60),
    ("SPECIALIST 50/50 with momentum, top 15", "spec", 0.5, 15, 45),
    ("momentum only in the 300 names, top 20", "spec", 1.0, 20, 60),
]
MIXES = (1.0, 0.8, 0.7, 0.6, 0.5)           # weight of the MAIN (800-stock smart) book


def build_components():
    pct = lambda x: x.groupby(level="date").rank(pct=True)
    ml = [pct(pd.read_parquet(BASE/f"data_store/smart_ml_pred_h{h}_s5_ev_pr_ni_L300_x3_px_d1.parquet")["pred"]) for h in (10, 21, 63)]
    ix = ml[0].index
    mlx = (ml[0] + ml[1].reindex(ix) + ml[2].reindex(ix))/3
    raw = pd.read_parquet(BASE/"data_store/smart_features_s1.parquet", columns=["mom_riskadj"])["mom_riskadj"].reindex(ix)
    pickle.dump(dict(ml=mlx, mom=pct(raw)), open(LC, "wb"))


def task(cfg):
    G = E.inputs()
    if "m" not in _S:
        _S["m"] = pickle.load(open(E.COMP, "rb")); _S["l"] = pickle.load(open(LC, "rb"))
    a0, a1 = pd.Timestamp(cfg["start"]), pd.Timestamp(cfg["end"])
    def window(x):
        d = x.index.get_level_values("date"); return x[(d >= a0) & (d <= a1)]
    mm = _S["m"]; sc_main = window((0.5*mm["ml"] + 0.5*mm["mom"].reindex(mm["ml"].index)).dropna())
    out_m = E.simulate_rules(sc_main, G["C"], G["mkt"], G["cm"], end_date=a1, **dict(E.FINAL, every=5, offset=cfg["off"]))
    ll = _S["l"]
    if cfg["model"] == "spec":
        sc = window(((1 - cfg["w"])*ll["ml"] + cfg["w"]*ll["mom"]).dropna())
    else:                                  # general model, but only the same 300 names, re-ranked within them
        idx_l = ll["ml"].index
        gm = (0.5*mm["ml"] + 0.5*mm["mom"].reindex(mm["ml"].index)).reindex(idx_l).dropna()
        sc = window(gm.groupby(level="date").rank(pct=True))
    out_l = E.simulate_rules(sc, G["C"], G["mkt"], G["cm"], end_date=a1,
                             **dict(E.FINAL, every=5, offset=cfg["off"], top_n=cfg["top_n"], exit_rank=cfg["exit_rank"]))
    return dict(label=cfg["label"], win=cfg["win"], off=cfg["off"], main=out_m["nav"][:a1].astype("float32"),
                large=out_l["nav"][:a1].astype("float32"))


def main():
    build_components()
    cfgs = [dict(label=l, model=m, w=w, top_n=n, exit_rank=er, win=wn, start=st, end=en, off=o)
            for l, m, w, n, er in VARIANTS for wn, (st, en) in E.WIN.items() for o in range(5)]
    t0 = time.time()
    with ProcessPoolExecutor(5, mp_context=mp.get_context("spawn")) as ex:
        res = list(ex.map(task, cfgs))
    print(f"({len(res)} runs in {time.time() - t0:.0f}s)")
    def mix(r, wm):
        a, b = r["main"].pct_change(), r["large"].pct_change()
        j = pd.concat([a, b], axis=1).dropna(); x = (1 + wm*j.iloc[:, 0] + (1 - wm)*j.iloc[:, 1]).cumprod()
        return x
    def yrs(n):
        o = {}
        for y in (2018, 2022, 2025, 2026):
            if n.index[0] <= pd.Timestamp(f"{y}-01-05") and n.index[-1] >= pd.Timestamp(f"{y}-12-20" if y < 2026 else "2026-09-25"):
                a0 = n.loc[:f"{y-1}-12-31"]; a0 = a0.iloc[-1] if len(a0) else n.iloc[0]
                o[y] = float(n.loc[:f"{y}-12-31"].iloc[-1]/a0 - 1)*100
        return o
    # ---- 1. standalone sleeve vs main book
    rows = []
    for r in res:
        s = E.nstats(r["large"]); sm = E.nstats(r["main"]); j = pd.concat([r["main"].pct_change(), r["large"].pct_change()], axis=1).dropna()
        mo = j.resample("ME").apply(lambda x: (1 + x).prod() - 1) if False else (1 + j).resample("ME").prod() - 1
        rows.append(dict(label=r["label"], win=r["win"], **{f"L_{k}": v for k, v in s.items()}, **{f"M_{k}": v for k, v in sm.items()},
                         corr=float(mo.iloc[:, 0].corr(mo.iloc[:, 1])), **{f"y{k}": v for k, v in yrs(r["large"]).items()}))
    R = pd.DataFrame(rows); M = R.groupby(["label", "win"]).mean(numeric_only=True)
    print(f"\n1. THE LARGE-CAP SLEEVE ON ITS OWN (300 most liquid stocks); main 800-stock smart book for reference\n")
    print(f"  {'':<44}{'A 2013-2020 CAGR/DD/Sh':>26}{'B 2021-Oct26 CAGR/DD/Sh':>28}   corr vs main (A/B)   2018 | 2022 | 2025 | 2026*")
    mA, mB = M.iloc[0][("M_cagr")] if False else None, None
    first = VARIANTS[0][0]
    print(f"  {'MAIN smart book (800 stocks)':<44}{M.loc[(first, 'A 2013-2020'), 'M_cagr']*100:>+9.1f}% {M.loc[(first, 'A 2013-2020'), 'M_dd']*100:>7.1f}% {M.loc[(first, 'A 2013-2020'), 'M_sh']:>5.2f}"
          f"   {M.loc[(first, 'B 2021-2026'), 'M_cagr']*100:>+9.1f}% {M.loc[(first, 'B 2021-2026'), 'M_dd']*100:>7.1f}% {M.loc[(first, 'B 2021-2026'), 'M_sh']:>5.2f}")
    for l, *_ in VARIANTS:
        a, b = M.loc[(l, "A 2013-2020")], M.loc[(l, "B 2021-2026")]
        ys = " ".join(f"{(a.get('y2018') if y == 2018 else b.get(f'y{y}')):>+5.0f}" for y in (2018, 2022, 2025, 2026))
        print(f"  {l:<44}{a['L_cagr']*100:>+9.1f}% {a['L_dd']*100:>7.1f}% {a['L_sh']:>5.2f}   {b['L_cagr']*100:>+9.1f}% {b['L_dd']*100:>7.1f}% {b['L_sh']:>5.2f}"
              f"      {a['corr']:>+5.2f} / {b['corr']:>+5.2f}    {ys}")
    # ---- 2. mixes, for the variant chosen on A only (best Sharpe among specialists)
    spec = [l for l, m, *_ in VARIANTS if m == "spec" and "momentum only" not in l]
    best = max(spec, key=lambda l: M.loc[(l, "A 2013-2020"), "L_sh"])
    print(f"\n2. MIXING THE MAIN BOOK WITH THE LARGE-CAP SLEEVE (fixed weights, no timing). Sleeve chosen on 2013-2020 only: '{best}'\n")
    print(f"  {'main / large-cap':<18}{'A 2013-2020 CAGR / DD / Sh':>30}{'B 2021-Oct26 CAGR / DD / Sh':>31}   2018 | 2022 | 2025 | 2026*")
    for wm in MIXES:
        st = {}
        for win in E.WIN:
            xs = [mix(r, wm) for r in res if r["label"] == best and r["win"] == win]
            st[win] = (np.mean([E.nstats(x)["cagr"] for x in xs]), np.mean([E.nstats(x)["dd"] for x in xs]), np.mean([E.nstats(x)["sh"] for x in xs]),
                       pd.DataFrame([yrs(x) for x in xs]).mean().to_dict())
        a, b = st["A 2013-2020"], st["B 2021-2026"]
        ys = " ".join(f"{(a[3].get(2018) if y == 2018 else b[3].get(y)):>+5.0f}" for y in (2018, 2022, 2025, 2026))
        print(f"  {f'{wm:.0%} / {1 - wm:.0%}':<18}{a[0]*100:>+12.1f}% {a[1]*100:>8.1f}% {a[2]:>5.2f}   {b[0]*100:>+12.1f}% {b[1]*100:>8.1f}% {b[2]:>5.2f}      {ys}")
    pickle.dump(dict(res=[(r["label"], r["win"], r["off"], r["main"], r["large"]) for r in res], best=best), open(BASE/"data_store/smart_large_results.pkl", "wb"))


if __name__ == "__main__":
    main()
