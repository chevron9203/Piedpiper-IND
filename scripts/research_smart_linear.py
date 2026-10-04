"""
A linear second opinion for the tree ensemble (diversity test). Ridge regression on the SAME ranked features and the SAME
cross-sectional rank target, walk-forward with the same purge, trained on the 5-day grid, predicting every trading day.
Then: does mixing a little of it into the LightGBM score help (A 2013-2020 / B 2021-Oct26, 5 weekday alignments)?

Run:  python scripts/research_smart_linear.py train | eval
"""
from __future__ import annotations
import multiprocessing as mp, pickle, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))
from scripts import research_smart_ml as SM           # noqa: E402
from scripts import research_smart_exit as E          # noqa: E402

ALPHA = 1000.0
_S = {}


def train():
    from sklearn.linear_model import Ridge
    t0 = time.time()
    P = SM.load_panel(); X, C, univ, mkt = SM.compute_features(P, 1)
    X = X.join(pd.read_parquet(BASE/"data_store/smart_events_s1.parquet")).join(pd.read_parquet(BASE/"data_store/smart_peers_s1.parquet"))
    X = X.drop(columns=[c for c in ("prom_net", "ins_net", "ins_buyers") if c in X.columns])
    Xr = SM.rank_features(X).astype("float32"); feats = list(Xr.columns)
    cal = C.index; grid = set(cal[260::5]); dates = Xr.index.get_level_values("date")
    print(f"features {Xr.shape} in {time.time() - t0:.0f}s", flush=True)
    for h in (10, 21, 63):
        y = SM.forward_returns(C, Xr.index.get_level_values("date").unique(), h)
        df = Xr.join(y, how="inner"); df["t"] = df.groupby(level="date")["fwd"].rank(pct=True)
        dd = df.index.get_level_values("date"); preds = []
        for yr in range(2013, dates.max().year + 1):
            start = pd.Timestamp(f"{yr}-01-01"); first = cal.searchsorted(start); purge = cal[max(first - h - 2, 0)]
            tr = df[(dd < purge) & dd.isin(grid)]
            mu = np.nanmean(tr[feats].values, axis=0); sd = np.nanstd(tr[feats].values, axis=0) + 1e-6
            prep = lambda M: np.nan_to_num((M - mu)/sd, nan=0.0)
            m = Ridge(alpha=ALPHA).fit(prep(tr[feats].values), tr["t"].values)
            te = Xr[(dates >= start) & (dates < pd.Timestamp(f"{yr + 1}-01-01"))]
            preds.append(pd.Series(m.predict(prep(te[feats].values)), index=te.index, name="pred"))
        pr = pd.concat(preds)
        pr.to_frame().to_parquet(BASE/f"data_store/smart_ml_pred_h{h}_ridge_d1.parquet")
        print(f"ridge h{h}: {len(pr):,} predictions, {time.time() - t0:.0f}s", flush=True)


def task(cfg):
    G = E.inputs()
    if "b" not in _S:
        c = pickle.load(open(E.COMP, "rb")); _S["b"] = c["ml"]; _S["m"] = c["mom"]
        pct = lambda x: x.groupby(level="date").rank(pct=True)
        r = [pct(pd.read_parquet(BASE/f"data_store/smart_ml_pred_h{h}_ridge_d1.parquet")["pred"]) for h in (10, 21, 63)]
        ix = r[0].index; _S["r"] = (r[0] + r[1].reindex(ix) + r[2].reindex(ix))/3
    a0, a1 = pd.Timestamp(cfg["start"]), pd.Timestamp(cfg["end"])
    base = _S["b"]; d = base.index.get_level_values("date"); keep = (d >= a0) & (d <= a1)
    ml = (1 - cfg["v"])*base[keep] + cfg["v"]*_S["r"].reindex(base.index[keep])
    sc = ((1 - cfg["w"])*ml + cfg["w"]*_S["m"].reindex(ml.index)).dropna()
    out = E.simulate_rules(sc, G["C"], G["mkt"], G["cm"], end_date=a1, **dict(E.FINAL, every=5, offset=cfg["off"]))
    return dict(**{k: cfg[k] for k in ("v", "w", "win", "off")}, nav=out["nav"][:a1].astype("float32"))


def evaluate():
    cfgs = [dict(v=v, w=w, win=wn, start=st, end=en, off=o) for v in (0.0, 0.15, 0.3, 0.5, 1.0) for w in (0.5, 0.0)
            for wn, (st, en) in E.WIN.items() for o in range(5)]
    with ProcessPoolExecutor(5, mp_context=mp.get_context("spawn")) as ex:
        res = list(ex.map(task, cfgs))
    R = pd.DataFrame([dict(v=r["v"], w=r["w"], win=r["win"], **E.nstats(r["nav"])) for r in res])
    M = R.groupby(["w", "v", "win"]).mean(numeric_only=True)
    for w, nm in ((0.5, "50% ML + 50% momentum"), (0.0, "ML only")):
        print(f"{nm}   (ML = (1-v) LightGBM + v Ridge; mean of 5 alignments)   CAGR / DD / Sharpe  [change vs v=0]")
        print(f"  {'ridge share v':<14}{'A 2013-2020':^42}{'B 2021-Oct26':^42}")
        for v in (0.0, 0.15, 0.3, 0.5, 1.0):
            cells = ""
            for wn in E.WIN:
                r = M.loc[(w, v, wn)]; b = M.loc[(w, 0.0, wn)]
                cells += f"{r['cagr']*100:>+8.1f}% {r['dd']*100:>7.1f}% {r['sh']:>5.2f} [{(r['cagr'] - b['cagr'])*100:+5.1f}pp]  "
            print(f"  {v:<14.2f}{cells}")
        print()


if __name__ == "__main__":
    {"train": train, "eval": evaluate}[sys.argv[1]]()
