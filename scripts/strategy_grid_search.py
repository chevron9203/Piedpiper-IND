"""
Grid search across Monthly Momentum parameters — IS 2008-2019, OOS 2020-2026.

Objective: find configurations that are robust across both periods.
Robustness metric: score = 0.4*IS_CAGR + 0.4*OOS_CAGR - 0.1*|IS_MDD| - 0.1*|OOS_MDD|

Usage:
    python scripts/strategy_grid_search.py
    python scripts/strategy_grid_search.py --quick   # 24 combos, ~2 min
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.backtest_momentum_monthly import (
    _load_adjusted_ohlcv, _load_index, _load_goldbees,
    run_monthly_momentum,
)
from data.universe import get_sector_map
from config import settings
from storage import store

IS_END = 2020


def _metrics(eq: pd.Series, start_capital: float) -> tuple[float, float]:
    """Returns (CAGR %, MaxDD %)."""
    if eq.empty:
        return 0.0, 0.0
    extended = pd.concat([pd.Series([start_capital],
                          index=[eq.index[0] - pd.DateOffset(months=1)]), eq])
    years = (extended.index[-1] - extended.index[0]).days / 365.25
    cagr = ((extended.iloc[-1] / extended.iloc[0]) ** (1 / years) - 1) * 100 if years > 0 else 0.0
    roll_max = extended.cummax()
    mdd = float(((extended - roll_max) / roll_max).min()) * 100
    return round(cagr, 2), round(mdd, 2)


def _run_one(universe_symbols, ohlcv_dict, nifty_close, vix_close, gold_close,
             sector_map, params: dict, capital: float) -> dict | None:
    try:
        eq, rebalances = run_monthly_momentum(
            ohlcv_dict=ohlcv_dict,
            nifty_close=nifty_close,
            universe_symbols=universe_symbols,
            vix_close=vix_close,
            start_capital=capital,
            gold_close=gold_close,
            sector_map=sector_map,
            **params,
        )
    except Exception as exc:
        print(f"  ERR: {exc}")
        return None

    if eq.empty:
        return None

    is_eq  = eq[eq.index.year <= IS_END]
    oos_eq = eq[eq.index.year >  IS_END]

    is_start  = capital
    oos_start = is_eq.iloc[-1] if not is_eq.empty else capital

    is_cagr,  is_mdd  = _metrics(is_eq,  is_start)
    oos_cagr, oos_mdd = _metrics(oos_eq, oos_start)

    score = 0.4 * is_cagr + 0.4 * oos_cagr - 0.1 * abs(is_mdd) - 0.1 * abs(oos_mdd)

    n_invested = sum(1 for r in rebalances if r.get("in_market", False))
    pct_in = n_invested / len(rebalances) * 100 if rebalances else 0

    return {
        "is_cagr":  is_cagr,
        "is_mdd":   is_mdd,
        "oos_cagr": oos_cagr,
        "oos_mdd":  oos_mdd,
        "score":    score,
        "pct_in":   pct_in,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="Smaller grid, faster run")
    ap.add_argument("--from-year", type=int, default=2008)
    ap.add_argument("--to-year",   type=int, default=2026)
    ap.add_argument("--capital",   type=float, default=100_000)
    args = ap.parse_args()

    from_dt = date(args.from_year, 1, 1)
    to_dt   = date(args.to_year,   12, 31)

    print(f"\nLoading data {args.from_year}–{args.to_year} ...")
    ohlcv_dict  = _load_adjusted_ohlcv(from_dt, to_dt)
    nifty_close = _load_index(settings.NIFTY50_SYMBOL,   from_dt, to_dt)
    vix_close   = _load_index(settings.INDIA_VIX_SYMBOL, from_dt, to_dt)
    print("Fetching GOLDBEES.NS ...")
    gold_close  = _load_goldbees(from_dt, to_dt)

    universe_symbols = [s for s in ohlcv_dict
                        if s not in (settings.NIFTY50_SYMBOL, settings.INDIA_VIX_SYMBOL)]

    sector_df  = get_sector_map()
    sector_map = dict(zip(sector_df["symbol"], sector_df["sector"])) if not sector_df.empty else None

    print(f"Universe: {len(universe_symbols)} symbols | Data: {args.from_year}–{args.to_year}")
    print(f"IS: {args.from_year}–{IS_END-1}  |  OOS: {IS_END}–{args.to_year}\n")

    # ── Grid definition ──────────────────────────────────────────────────────
    if args.quick:
        grid = {
            "top_n":               [5, 10, 15],
            "lookback_months":     [12],
            "skip_months":         [1],
            "abs_momentum":        [True],
            "inv_vol_weight":      [True],
            "require_200d_ema":    [True],
            "require_nifty_200ema":[False, True],
            "defensive_mode":      ["gold", "cash"],
            "vix_reduce":          [25.0],
            "vix_exit":            [35.0],
            "max_per_sector":      [3],
        }
    else:
        grid = {
            "top_n":               [5, 10, 15, 20],
            "lookback_months":     [9, 12],
            "skip_months":         [1],
            "abs_momentum":        [True],
            "inv_vol_weight":      [True, False],
            "require_200d_ema":    [True],
            "require_nifty_200ema":[False, True],
            "defensive_mode":      ["gold", "cash", "split"],
            "vix_reduce":          [20.0, 25.0],
            "vix_exit":            [30.0, 35.0],
            "max_per_sector":      [2, 3],
        }

    keys   = list(grid.keys())
    combos = list(product(*[grid[k] for k in keys]))
    print(f"Running {len(combos)} parameter combinations ...\n")

    rows = []
    for i, vals in enumerate(combos, 1):
        params = dict(zip(keys, vals))

        result = _run_one(universe_symbols, ohlcv_dict, nifty_close, vix_close,
                          gold_close, sector_map, params, args.capital)
        if result is None:
            continue

        label = (
            f"top{params['top_n']}"
            f"|lk{params['lookback_months']}"
            f"|{'iv' if params['inv_vol_weight'] else 'ew'}"
            f"|{'n200' if params['require_nifty_200ema'] else '----'}"
            f"|def={params['defensive_mode']:<5}"
            f"|vx{int(params['vix_reduce'])}/{int(params['vix_exit'])}"
        )

        rows.append({"label": label, **params, **result})

        if i % 20 == 0 or i == len(combos):
            print(f"  {i}/{len(combos)} done ...", flush=True)

    if not rows:
        print("No valid results.")
        return

    df = pd.DataFrame(rows).sort_values("score", ascending=False)

    # ── Results table ────────────────────────────────────────────────────────
    WIDTH = 105
    print(f"\n{'='*WIDTH}")
    print(f"{'Config':<52} {'IS CAGR':>8} {'IS MDD':>8} {'OOS CAGR':>9} {'OOS MDD':>8} {'%In':>5} {'Score':>7}")
    print(f"{'─'*WIDTH}")
    for _, row in df.head(25).iterrows():
        print(
            f"{row['label']:<52} "
            f"{row['is_cagr']:>7.1f}% "
            f"{row['is_mdd']:>7.1f}% "
            f"{row['oos_cagr']:>8.1f}% "
            f"{row['oos_mdd']:>7.1f}% "
            f"{row['pct_in']:>4.0f}% "
            f"{row['score']:>7.1f}"
        )

    print(f"{'='*WIDTH}")

    # ── Best config ──────────────────────────────────────────────────────────
    best = df.iloc[0]
    print(f"\nBEST CONFIG: {best['label']}")
    print(f"  IS  {args.from_year}–{IS_END-1}: CAGR {best['is_cagr']:+.1f}%  MaxDD {best['is_mdd']:.1f}%")
    print(f"  OOS {IS_END}–{args.to_year}:  CAGR {best['oos_cagr']:+.1f}%  MaxDD {best['oos_mdd']:.1f}%")
    print(f"  Invested: {best['pct_in']:.0f}% of months")

    # ── Consistency check: how many top-10 combos beat IS+OOS thresholds ────
    robust = df[(df["is_cagr"] > 12) & (df["oos_cagr"] > 15) & (df["is_mdd"] > -35)]
    print(f"\nRobust configs (IS>12% AND OOS>15% AND IS_MDD>-35%): {len(robust)} / {len(df)}")
    if not robust.empty:
        print(f"  IS CAGR range: {robust['is_cagr'].min():.1f}% – {robust['is_cagr'].max():.1f}%")
        print(f"  OOS CAGR range: {robust['oos_cagr'].min():.1f}% – {robust['oos_cagr'].max():.1f}%")


if __name__ == "__main__":
    main()
