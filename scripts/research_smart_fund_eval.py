"""
Do fundamentals (growth, margin change, earnings surprise from quarterly XBRL, now continuous to 2026 via NSE's integrated
filing feed) improve the ranking? Baseline = the live 3-horizon x 3-seed daily models; fd = identical but with the
fundamental features (`--fund --ptag fd`). Judged on windows where fundamentals exist for training:
  B 2021-Oct 2026 | C 2023-Oct 2026 | D 2025-Oct 2026 (the stretch the old feed could not cover)
50/50 with momentum and ML-only, mean of 5 weekday alignments, per-stock costs; plus rank-IC of the ML score.

Run:  python scripts/research_smart_fund_eval.py
"""
from __future__ import annotations
import multiprocessing as mp, pickle, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))
from scripts import research_smart_exit as E          # noqa: E402
from scripts import research_smart_ml as SM           # noqa: E402

SUFFIX = sys.argv[1] if len(sys.argv) > 1 else "fd"          # prediction-file suffix to compare with the baseline (fd, mc, ...)
WINS = {"A 2013-2020": ("2013-01-01", "2020-12-31"), "B 2021-Oct26": ("2021-01-01", "2026-12-31"),
        "C 2023-Oct26": ("2023-01-01", "2026-12-31"), "D 2025-Oct26": ("2025-01-01", "2026-12-31")}
if SUFFIX == "fd":
    WINS.pop("A 2013-2020")                                   # fundamentals do not exist before 2019
FDF = BASE/f"data_store/smart_{SUFFIX}_components.pkl"
_S = {}


def build():
    pct = lambda x: x.groupby(level="date").rank(pct=True)
    ml = [pct(pd.read_parquet(BASE/f"data_store/smart_ml_pred_h{h}_s5_ev_pr_ni_x3_px_d1_{SUFFIX}.parquet")["pred"]) for h in (10, 21, 63)]
    ix = ml[0].index
    pickle.dump(ml[0].to_frame("x").assign(x=(ml[0] + ml[1].reindex(ix) + ml[2].reindex(ix))/3)["x"], open(FDF, "wb"))


def task(cfg):
    G = E.inputs()
    if "b" not in _S:
        c = pickle.load(open(E.COMP, "rb")); _S["b"] = c["ml"]; _S["m"] = c["mom"]; _S["f"] = pickle.load(open(FDF, "rb"))
    a0, a1 = pd.Timestamp(cfg["start"]), pd.Timestamp(cfg["end"])
    ml = _S["f"] if cfg["model"] == "fd" else _S["b"]
    d = ml.index.get_level_values("date"); keep = (d >= a0) & (d <= a1)
    mlk = ml[keep]; w = cfg["w"]
    sc = ((1 - w)*mlk + w*_S["m"].reindex(mlk.index)).dropna()
    out = E.simulate_rules(sc, G["C"], G["mkt"], G["cm"], end_date=a1, **dict(E.FINAL, every=5, offset=cfg["off"]))
    return dict(**{k: cfg[k] for k in ("model", "w", "win", "off")}, nav=out["nav"][:a1].astype("float32"))


def rank_ic():
    G = E.inputs(); C = G["C"].astype("float64")
    c = pickle.load(open(E.COMP, "rb")); base = c["ml"]; fd = pickle.load(open(FDF, "rb"))
    dates = base.index.get_level_values("date").unique()[::5]
    y = SM.forward_returns(C, dates, 21)
    out = {}
    for lab, ml in (("baseline", base), ("fundamentals", fd)):         # 'fundamentals' = the variant under test
        d = pd.concat([ml.rename("s"), y], axis=1, join="inner")
        d = d[d.index.get_level_values("date").isin(dates)]
        ic = d.groupby(level="date").apply(lambda g: g["s"].corr(g["fwd"], method="spearman"))
        out[lab] = ic
    return out


def main():
    build()
    cfgs = [dict(model=m, w=w, win=wn, start=st, end=en, off=o) for m in ("base", "fd") for w in (0.5, 0.0) for wn, (st, en) in WINS.items() for o in range(5)]
    t0 = time.time()
    with ProcessPoolExecutor(5, mp_context=mp.get_context("spawn")) as ex:
        res = list(ex.map(task, cfgs))
    print(f"({len(res)} runs in {time.time() - t0:.0f}s)\n")
    R = pd.DataFrame([dict(model=r["model"], w=r["w"], win=r["win"], **E.nstats(r["nav"])) for r in res])
    M = R.groupby(["w", "model", "win"]).mean(numeric_only=True)
    for w, nm in ((0.5, "50% ML + 50% momentum"), (0.0, "ML only")):
        print(f"{nm}   (mean of 5 weekday alignments)   CAGR / DD / Sharpe   [change vs baseline]")
        print(f"  {'':<14}" + "".join(f"{wn:^42}" for wn in WINS))
        for m, lab in (("base", "baseline"), ("fd", f"+ {SUFFIX}")):
            cells = ""
            for wn in WINS:
                r = M.loc[(w, m, wn)]; b = M.loc[(w, "base", wn)]
                d = f"[{(r['cagr'] - b['cagr'])*100:+.1f}pp, Sh {r['sh'] - b['sh']:+.2f}]" if m == "fd" else ""
                cells += f"{r['cagr']*100:>+8.1f}% {r['dd']*100:>7.1f}% {r['sh']:>5.2f} {d:<18}"
            print(f"  {lab:<14}{cells}")
        print()
    ic = rank_ic()
    print("rank-IC of the ML score vs the next 21 days (every 5th decision date):")
    for wn, (a0, a1) in WINS.items():
        s = {k: v[(v.index >= a0) & (v.index <= a1)] for k, v in ic.items()}
        print(f"  {wn}: baseline IC {s['baseline'].mean():+.4f}   + {SUFFIX} {s['fundamentals'].mean():+.4f}   "
              f"(difference {s['fundamentals'].mean() - s['baseline'].mean():+.4f}, t={(s['fundamentals'] - s['baseline']).mean()/(s['fundamentals'] - s['baseline']).std()*np.sqrt(len(s['baseline'])):+.1f})")


if __name__ == "__main__":
    main()
