"""
Fundamental features from quarterly XBRL results (fetch_nse_xbrl.py), point-in-time.

One series per company: consolidated if the company mostly files consolidated, else
standalone -- never mixed, so year-on-year compares like with like. First filing per
quarter only (revisions arrive later and would otherwise leak into the past).

Features on decision dates, using filings with broadcast time before that day's close:
  rev_yoy / rev_accel   revenue growth vs same quarter last year, and its change
  pat_yoy               profit growth, (P - P4)/|P4| (works through losses), clipped
  opm / opm_chg         operating margin (PBT + finance cost + dep - other income)/revenue
                        and its year-on-year change
  sue                   standardised unexpected earnings: (P - P4) scaled by the std of
                        that change over the previous 8 quarters -- surprise vs the
                        company's own history, no analyst consensus needed
  exc_share             exceptional items / |PBT| (one-offs flatter or hide results)
  fund_age              trading days since the latest result became public

Coverage: XBRL starts mid-2018, so these exist from ~2019 (SUE from ~2020).
Run:  python scripts/research_smart_fund.py [--step 5]
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))


def quarterly():
    T = pd.read_parquet(BASE/"data_store/fundamentals.parquet").dropna(subset=["known", "period_end", "revenue"])
    T = T.sort_values("known")
    pref = T.groupby("sym")["cons"].mean().ge(0.5)
    T = T[T["cons"] == T["sym"].map(pref)]
    T = T.drop_duplicates(["sym", "period_end"], keep="first")     # original filing only
    # quarter key so "4 quarters ago" is exact even with a missing filing
    T["qk"] = T["period_end"].dt.year*4 + (T["period_end"].dt.month - 1)//3
    T = T.sort_values(["sym", "qk"]).reset_index(drop=True)
    g = T.groupby("sym")
    def lag(col, n):
        prev = T[["sym", "qk", col]].copy(); prev["qk"] += n
        return T[["sym", "qk"]].merge(prev, on=["sym", "qk"], how="left")[col].values
    T["opm"] = ((T["pbt"] + T["fin_cost"].fillna(0) + T["dep"].fillna(0) - T["other_inc"].fillna(0))
                / T["revenue"].where(T["revenue"] > 0))
    r4, p4, o4 = lag("revenue", 4), lag("pat", 4), lag("opm", 4)
    T["rev_yoy"] = (T["revenue"]/np.where(r4 > 0, r4, np.nan) - 1).clip(-1, 5)
    T["pat_yoy"] = ((T["pat"] - p4)/np.abs(np.where(p4 != 0, p4, np.nan))).clip(-5, 5)
    T["opm_chg"] = (T["opm"] - o4).clip(-1, 1)
    T["opm"] = T["opm"].clip(-1, 1)
    T["d4"] = T["pat"] - p4
    prev_rev_yoy = T[["sym", "qk", "rev_yoy"]].copy(); prev_rev_yoy["qk"] += 1
    T["rev_accel"] = T["rev_yoy"] - T[["sym", "qk"]].merge(prev_rev_yoy, on=["sym", "qk"], how="left")["rev_yoy"].values
    sd = g["d4"].transform(lambda s: s.shift(1).rolling(8, min_periods=4).std())
    T["sue"] = (T["d4"]/sd.replace(0, np.nan)).clip(-10, 10)
    T["exc_share"] = (T["exceptional"].fillna(0)/T["pbt"].abs().replace(0, np.nan)).clip(-5, 5)
    return T


def build(step):
    from scripts import research_smart_ml as SM
    T = quarterly()
    P = SM.load_panel()
    X, C, univ, mkt = SM.compute_features(P, step)
    idx = C.index
    # knowable at the close of the first trading day whose 15:30 close is after the broadcast
    after_close = T["known"].dt.hour*60 + T["known"].dt.minute > 15*60 + 30
    day = T["known"].dt.normalize() + pd.to_timedelta(after_close.astype(int), unit="D")
    pos = np.minimum(idx.searchsorted(day.values), len(idx) - 1)
    T["avail"] = idx[pos]
    feats = ["rev_yoy", "rev_accel", "pat_yoy", "opm", "opm_chg", "sue", "exc_share"]
    K = X.index.to_frame(index=False).sort_values("date")
    E = pd.merge_asof(K, T[["sym", "avail"] + feats].sort_values("avail"), left_on="date",
                      right_on="avail", by="sym", direction="backward")
    E["fund_age"] = idx.get_indexer(E["date"]) - idx.get_indexer(E["avail"].fillna(idx[0]))
    stale = E["avail"].isna() | (E["fund_age"] > 150)            # > ~2 quarters old: drop
    E.loc[stale, feats + ["fund_age"]] = np.nan
    out = E.set_index(["date", "sym"])[feats + ["fund_age"]].reindex(X.index).astype("float32")
    out.to_parquet(BASE/f"data_store/smart_fund_s{step}.parquet")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--step", type=int, default=5)
    a = ap.parse_args()
    out = build(a.step)
    yr = out.index.get_level_values("date").year
    print("coverage by year:\n", out.notna().groupby(yr).mean().round(2)[["rev_yoy", "pat_yoy", "sue"]].T.to_string())
    print(out.describe().T[["mean", "50%", "min", "max"]].round(3))
