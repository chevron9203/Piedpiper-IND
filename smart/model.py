"""Train / predict the v1 models exactly as researched (research_smart_ml.walk_forward
params, 3 seeds averaged, cross-sectional rank target), but one final fit on all rows
whose label is COMPLETE (the research label clamps at the panel end; live must not train
on a 63-day return that has only run 10 days)."""
from __future__ import annotations
import json, sys, time
import numpy as np, pandas as pd

from smart.config import BASE, HORIZONS, LABELS, MODELS, MOM_W, SEEDS, STORE, THREADS

sys.path.insert(0, str(BASE))
from scripts import research_smart_ml as SM          # noqa: E402

PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=31,
              min_data_in_leaf=400, feature_fraction=0.7, bagging_fraction=0.7,
              bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=THREADS)
ROUNDS = 400


def complete_labels(C, dates, h):
    """Forward return close[t+1] -> close[t+1+h] for dates whose window has fully elapsed;
    names that stop trading for good get the research haircut."""
    idx = C.index
    pos = idx.get_indexer(dates)
    ok = (pos >= 0) & (pos + 1 + h <= len(idx) - 1)
    if not ok.any():
        return pd.Series(dtype="float32", name=f"y{h}")
    y = SM.forward_returns(C, pd.DatetimeIndex(dates[ok]), h)
    return y.rename(f"y{h}").astype("float32")


def train(store=None, labels=None, horizons=HORIZONS, seeds=SEEDS, log=print):
    import lightgbm as lgb
    X = pd.read_parquet(STORE) if store is None else store
    Y = pd.read_parquet(LABELS) if labels is None else labels
    Xr = SM.rank_features(X)
    feats = list(X.columns)
    MODELS.mkdir(parents=True, exist_ok=True)
    meta = {"features": feats, "trained": str(pd.Timestamp.now()), "horizons": {}}
    for h in horizons:
        y = Y[f"y{h}"].dropna()
        df = Xr.join(y, how="inner")
        tgt = df.groupby(level="date")[f"y{h}"].rank(pct=True)
        t0 = time.time()
        for sd in seeds:
            m = lgb.train({**PARAMS, "seed": sd}, lgb.Dataset(df[feats], tgt), num_boost_round=ROUNDS)
            m.save_model(str(MODELS/f"h{h}_s{sd}.txt"))
        last = df.index.get_level_values("date").max()
        meta["horizons"][str(h)] = {"rows": int(len(df)), "last_label_date": str(last.date())}
        log(f"trained h{h}: {len(df):,} rows through {last.date()}, {len(seeds)} seeds, {time.time()-t0:.0f}s")
    (MODELS/"meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def score_parts(X_today):
    """(ml, momentum) percentile scores for one decision date, each a Series indexed by symbol:
    ml = mean rank of the 3 horizons x 3 seeds; momentum = risk-adjusted multi-timeframe momentum rank."""
    import lightgbm as lgb
    meta = json.loads((MODELS/"meta.json").read_text())
    Xr = SM.rank_features(X_today)[meta["features"]]
    comp = []
    for h in meta["horizons"]:
        p = np.mean([lgb.Booster(model_file=str(MODELS/f"h{h}_s{sd}.txt")).predict(Xr) for sd in SEEDS], axis=0)
        comp.append(pd.Series(p, index=Xr.index).rank(pct=True))
    ml = sum(comp)/len(comp)
    mom = Xr["mom_riskadj"].rank(pct=True)
    return ml.droplevel("date"), mom.droplevel("date")


def score(X_today):
    """Blend score for one decision date: (1-MOM_W) x ML rank + MOM_W x momentum rank."""
    ml, mom = score_parts(X_today)
    return ((1 - MOM_W)*ml + MOM_W*mom).sort_values(ascending=False)
