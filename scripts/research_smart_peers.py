"""
Peer-group features for the smart system -- sector/theme rotation without a sector map.

A static sector map only covers today's survivors (survivorship leak), so peers are
learned point-in-time from the tape: every REFRESH trading days, for every stock in the
universe, its K most correlated names on market-residual daily returns over the past
year. Between refreshes the peer lists are frozen; features use closes up to t only.

  peer_r21 / peer_r63   how strong the stock's group is (rotation)
  peer_mom              group's risk-adjusted momentum
  rel_r21 / rel_r63     stock minus its group (leader vs laggard within the theme)
  peer_d52h             group distance from 52w high (is the theme breaking out)
  peer_corr             how tight the group is (a real theme vs noise)

Output: data_store/smart_peers_s{step}.parquet keyed (date, sym) on decision dates.
Run:  python scripts/research_smart_peers.py [--step 5]
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))
from scripts import research_smart_ml as SM

K = 15; REFRESH = 21; LOOKBACK = 252; MIN_OBS = 150
SRC = ["r21", "r63", "mom_riskadj", "d52h"]


def build(step, P=None, feats=None, save=True):
    """feats: (X, C, univ, mkt) from compute_features -- the live service passes its own;
    every date in X then gets fresh peer lists if it is a refresh date (live: always)."""
    if feats is None:
        P = SM.load_panel()
        X, C, univ, mkt = SM.compute_features(P, step)
    else:
        X, C, univ, mkt = feats
    R = C.pct_change(fill_method=None)
    res = R.sub(mkt, axis=0)                      # market-residual: what is left is group + own
    idx = C.index
    dates = X.index.get_level_values("date").unique()
    refresh = dates[::max(REFRESH//step, 1)]
    Xs = X[SRC]
    out = []
    peers = None
    for d in dates:
        if d in refresh:
            i = idx.get_loc(d)
            cols = univ.loc[d][univ.loc[d]].index
            M = res.iloc[max(0, i - LOOKBACK + 1): i + 1][cols]
            ok = M.notna().sum() >= MIN_OBS
            M = M.loc[:, ok]
            Z = (M - M.mean())/M.std()
            Zv = Z.fillna(0.0).values
            n = M.notna().astype(float).values
            cor = (Zv.T @ Zv)/np.maximum(n.T @ n - 1, 1)
            np.fill_diagonal(cor, -np.inf)
            top = np.argpartition(-cor, K, axis=1)[:, :K]
            names = M.columns.values
            peers = {names[j]: (names[top[j]], float(np.mean(cor[j, top[j]]))) for j in range(len(names))}
        if peers is None:
            continue
        xd = Xs.xs(d, level="date")
        rows = {}
        for s in xd.index:
            if s not in peers:
                continue
            pn, pc = peers[s]
            g = xd.reindex(pn)
            if g["r21"].notna().sum() < 5:
                continue
            m = g.mean()
            rows[s] = (m["r21"], m["r63"], m["mom_riskadj"], m["d52h"],
                       xd.at[s, "r21"] - m["r21"], xd.at[s, "r63"] - m["r63"], pc)
        f = pd.DataFrame.from_dict(rows, orient="index",
                                   columns=["peer_r21", "peer_r63", "peer_mom", "peer_d52h",
                                            "rel_r21", "rel_r63", "peer_corr"])
        f.index = pd.MultiIndex.from_product([[d], f.index], names=["date", "sym"])
        out.append(f)
    F = pd.concat(out).astype("float32").reindex(X.index)
    if save:
        F.to_parquet(BASE/f"data_store/smart_peers_s{step}.parquet")
    return F


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--step", type=int, default=5)
    a = ap.parse_args()
    F = build(a.step)
    print("coverage:", {c: f"{F[c].notna().mean():.0%}" for c in F.columns})
    print(F.describe().T[["mean", "50%", "min", "max"]].round(4))
