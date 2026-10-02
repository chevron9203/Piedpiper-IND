"""
Freeze the per-system backtest return distributions that system_health.py scores live
returns against.

Why per-system: S1 is the regime-conditional multi-asset book (gold/US/cash/momentum)
and S5 is pure always-invested mid-cap momentum. Their return distributions are nothing
alike -- S5's monthly sd is roughly 3x S1's. Scoring S1's live returns against a
pure-momentum reference would call a perfectly normal month an outlier (or, worse, miss
a real breakdown because the band is far too wide).

Run:  python scripts/build_health_reference.py
"""
from __future__ import annotations
import sys, json
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np, pandas as pd

BASE = Path(__file__).parent.parent
OUT = BASE/"data_store/backtest_reference.json"


def describe(r, label):
    r = pd.Series(r).dropna()
    return dict(label=label, n=int(len(r)),
                mean=float(r.mean()), sd=float(r.std()),
                p05=float(r.quantile(0.05)), p50=float(r.quantile(0.50)),
                p95=float(r.quantile(0.95)),
                worst_month=float(r.min()), best_month=float(r.max()))


def pure_momentum():
    """S4/S5 family — from the already-computed intra-month baseline."""
    f = BASE/"data_store/intramonth_results.csv"
    if not f.exists():
        print("  ! intramonth_results.csv missing — run scripts/research_intramonth.py")
        return None
    return describe(pd.read_csv(f, index_col=0)["BASELINE monthly (no stop)"].values,
                    "pure_momentum_monthly_2010_2026")


def regime_conditional():
    """S1 — the deployed book. Reuses the validated backtest's own components."""
    try:
        from scripts.backtest_regime_conditional import load, components, regime_conditional as rc
    except Exception as e:
        print(f"  ! could not import regime-conditional backtest: {e!r}")
        return None
    close, turn, nifty, gold, us = load()
    D = components(close, turn, nifty, gold, us)
    return describe(rc(D, 0.70).values, "regime_conditional_70mom_monthly")


def main():
    refs = {"built": str(datetime.now()), "systems": {}}
    print("building per-system reference distributions ...")
    for key, fn in [("s1", regime_conditional), ("s5", pure_momentum)]:
        r = fn()
        if r:
            refs["systems"][key] = r
            print(f"  {key}: n={r['n']}  mean={r['mean']*100:+.2f}%/mo  sd={r['sd']*100:.2f}pp  "
                  f"worst={r['worst_month']*100:+.1f}%")
    if not refs["systems"]:
        raise SystemExit("no references built")
    json.dump(refs, open(OUT, "w"), indent=2)
    print(f"\nsaved -> {OUT.relative_to(BASE)}")


if __name__ == "__main__":
    main()
