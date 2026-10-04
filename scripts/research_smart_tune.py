"""
Model-tuning screen (user, 2026-10-04: "fundamentals and others"). The LightGBM settings were never tuned and the label
was always the plain cross-sectional return rank. Variants (h21 only, 1 seed, DAILY predictions so all 5 weekday
alignments can be averaged; compare to the matching baseline 'b1', not to the 3-seed ensemble):
  b1 baseline | lv15 shallower trees | lv63 deeper trees | slow lr 0.015 x 800 rounds | reg stronger regularisation |
  vs   RISK-ADJUSTED label (rank of return / trailing 63d volatility)
Judged on A=2013-2020 and B=2021-Oct 2026, 50/50 with momentum and ML-only, per-stock costs.

Run:  python scripts/research_smart_tune.py
"""
from __future__ import annotations
import multiprocessing as mp, pickle, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))
from scripts import research_smart_exit as E          # noqa: E402

TAGS = {"b1": "baseline (defaults)", "lv15": "shallower trees (15 leaves, min 600)", "lv63": "deeper trees (63 leaves, min 300)",
        "slow": "slow learning (lr .015, 800 rounds)", "reg": "stronger regularisation (ff .4, l2 30)", "vs": "RISK-ADJUSTED label (return / volatility)"}
_S = {}


def task(cfg):
    G = E.inputs()
    if "mom" not in _S:
        _S["mom"] = pickle.load(open(E.COMP, "rb"))["mom"]
    ml = pd.read_parquet(BASE/f"data_store/smart_ml_pred_h21_s5_ev_pr_ni_px_d1_{cfg['tag']}.parquet")["pred"]
    ml = ml.groupby(level="date").rank(pct=True)
    a0, a1 = pd.Timestamp(cfg["start"]), pd.Timestamp(cfg["end"])
    d = ml.index.get_level_values("date"); keep = (d >= a0) & (d <= a1)
    mlk = ml[keep]; w = cfg["w"]
    sc = ((1 - w)*mlk + w*_S["mom"].reindex(mlk.index)).dropna()
    out = E.simulate_rules(sc, G["C"], G["mkt"], G["cm"], end_date=a1, **dict(E.FINAL, every=5, offset=cfg["off"]))
    return dict(**{k: cfg[k] for k in ("tag", "w", "win", "off")}, nav=out["nav"][:a1].astype("float32"), to=out["turnover"])


def main():
    cfgs = [dict(tag=t, w=w, win=wn, start=st, end=en, off=o) for t in TAGS for w in (0.5, 0.0)
            for wn, (st, en) in E.WIN.items() for o in range(5)
            if (BASE/f"data_store/smart_ml_pred_h21_s5_ev_pr_ni_px_d1_{t}.parquet").exists()]
    t0 = time.time()
    with ProcessPoolExecutor(5, mp_context=mp.get_context("spawn")) as ex:
        res = list(ex.map(task, cfgs))
    print(f"({len(res)} runs in {time.time() - t0:.0f}s)\n")
    R = pd.DataFrame([dict(tag=r["tag"], w=r["w"], win=r["win"], **E.nstats(r["nav"]), to=r["to"]) for r in res])
    M = R.groupby(["w", "tag", "win"]).mean(numeric_only=True).unstack("win")
    for w, nm in ((0.5, "50% ML + 50% momentum"), (0.0, "ML only")):
        print(f"{nm}   (mean of 5 weekday alignments; h21 model, 1 seed)")
        print(f"  {'':<42}{'A 2013-2020 CAGR / DD / Sh':>30}{'B 2021-Oct26 CAGR / DD / Sh':>32}   turnover")
        base = M.loc[(w, "b1")] if (w, "b1") in M.index else None
        for t, lab in TAGS.items():
            if (w, t) not in M.index: continue
            r = M.loc[(w, t)]
            dA = f"{(r[('cagr', 'A 2013-2020')] - base[('cagr', 'A 2013-2020')])*100:+.1f}" if base is not None else ""
            dB = f"{(r[('cagr', 'B 2021-2026')] - base[('cagr', 'B 2021-2026')])*100:+.1f}" if base is not None else ""
            print(f"  {lab:<42}{r[('cagr', 'A 2013-2020')]*100:>+9.1f}% {r[('dd', 'A 2013-2020')]*100:>7.1f}% {r[('sh', 'A 2013-2020')]:>5.2f} ({dA:>5})"
                  f"{r[('cagr', 'B 2021-2026')]*100:>+10.1f}% {r[('dd', 'B 2021-2026')]*100:>7.1f}% {r[('sh', 'B 2021-2026')]:>5.2f} ({dB:>5})   "
                  f"{(R[(R.w == w) & (R.tag == t)].to.mean()):.0%}")
        print()


if __name__ == "__main__":
    main()
