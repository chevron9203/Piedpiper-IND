"""
Cross-asset sleeve beside the smart book (no timing, fixed weights): Indian gold ETF (GOLDBEES) + Nasdaq-100 ETF (MON100).
DATA NOTE: both ETFs have UNIT SPLITS that are missing from the price panel's corporate-action data - GOLDBEES 1:100 on
2019-12-19 (3,360 -> 33.55) and MON100 1:10 on 2021-06-17 (1,016 -> 100.9). Unfixed they show fake -99% / -90% days.
(The stock model is unaffected: ETFs are excluded from its universe.) Fixed here by scaling the earlier prices.

Run:  python scripts/research_smart_assets.py       (needs data_store/smart_split_results.pkl from `research_smart_exit.py split`)
"""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))
from scripts import research_smart_exit as E          # noqa: E402


def sleeves_returns(idx):
    C = pd.read_parquet(BASE/"data_store/nse_panel.parquet")["close"]
    gold = C["GOLDBEES"].ffill(limit=5).copy(); us = C["MON100"].ffill(limit=5).copy()
    gold.loc[:"2019-12-18"] *= 0.01
    us.loc[:"2021-06-16"] *= 0.1
    return gold.pct_change().reindex(idx), us.pct_change().reindex(idx)


def main():
    smart = pickle.load(open(BASE/"data_store/smart_split_results.pkl", "rb"))["nav"]["smart"].dropna()
    r_g, r_u = sleeves_returns(smart.index)
    j = pd.concat([smart.pct_change(), r_g, r_u], axis=1, keys=["smart", "gold", "us"]).dropna()
    sleeve = {"gold+US 50/50": 0.5*j.gold + 0.5*j.us, "gold only": j.gold, "US only": j.us}
    nav = lambda r: (1 + r).cumprod()
    def st(x):
        x = x/x.iloc[0]; yrs = (x.index[-1] - x.index[0]).days/365.25; r = x.pct_change().dropna()
        return x.iloc[-1]**(1/yrs) - 1, (x/x.cummax() - 1).min(), r.mean()/r.std()*np.sqrt(252)
    for k, v in sleeve.items():
        m = pd.concat([j.smart, v], axis=1).resample("ME").apply(lambda x: (1 + x).prod() - 1).corr().iloc[0, 1]
        print(f"{k}: monthly correlation with smart book {m:+.2f}")
    for key, v in sleeve.items():
        print(f"\nsmart + {key}")
        for w in (1.0, 0.9, 0.8, 0.7, 0.6):
            n = nav(w*j.smart + (1 - w)*v); a, b = st(n.loc[:"2020-12-31"]), st(n.loc["2021-01-01":])
            print(f"  {w:.0%}/{1 - w:.0%}: A {a[0]*100:+.1f}% / {a[1]*100:.1f}% / {a[2]:.2f}   B {b[0]*100:+.1f}% / {b[1]*100:.1f}% / {b[2]:.2f}")


if __name__ == "__main__":
    main()
