"""
Piedpiper — Full Combined System Backtest
==========================================
Integrates all three strategies over their full available history:

  ₹3L → Monthly Momentum (2010–2026, longest available daily data)
  ₹2L → ORB LONG intraday  (2020–2026, limited by 15-min data availability)
  ₹2L → LIQUIDBEES yield   (2010–2019, idle capital earns ~6.5% p.a.)

Shows what ₹5L deployed in the full system from 2010 would have become.

Run:
  python scripts/backtest_combined.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from storage import store
from backtest_momentum_monthly import (
    run_monthly_momentum,
    _load_adjusted_ohlcv,
    _load_index,
    _load_goldbees,
)
from config import settings

# ── Constants ─────────────────────────────────────────────────────────────────

MOM_CAPITAL   = 300_000   # ₹3L in monthly momentum
ORB_CAPITAL   = 200_000   # ₹2L in ORB LONG intraday
TOTAL_CAPITAL = 500_000   # ₹5L total system

ORB_FIRST_YEAR = 2020     # 15-min data available from 2020

# ORB LONG year-by-year P&L (from real backtest, ₹1L basis → scale to ₹2L)
ORB_PNL_1L = {
    2020: 104_127, 2021: 170_004, 2022:  31_560,
    2023: 109_976, 2024: 122_232, 2025:  10_211,
    2026:  49_810,   # partial year (Jan–Aug)
}

# Scale ORB P&L from ₹1L basis to ₹2L: linear (3% risk, so scales exactly)
ORB_PNL_2L = {yr: pnl * 2 for yr, pnl in ORB_PNL_1L.items()}

LIQUID_RATE  = 0.065   # LIQUIDBEES idle yield when ORB not running
IS_END_YEAR  = 2016    # clean IS/OOS split
FROM_YEAR    = 2010
TO_YEAR      = 2026


# ── Load data ─────────────────────────────────────────────────────────────────

def _load_universe(ohlcv_dict: dict[str, pd.DataFrame]) -> list[str]:
    """Return symbols that have ≥5 years of data (avoid thin coverage)."""
    return [s for s, df in ohlcv_dict.items()
            if len(df) >= 1250 and s not in (settings.NIFTY50_SYMBOL, settings.INDIA_VIX_SYMBOL)]


# ── Main backtest ─────────────────────────────────────────────────────────────

def main() -> None:
    from_dt = date(FROM_YEAR, 1, 1)
    to_dt   = date(TO_YEAR,   12, 31)

    print(f"\nLoading data {FROM_YEAR}–{TO_YEAR} ...")
    ohlcv_dict  = _load_adjusted_ohlcv(from_dt, to_dt)
    nifty_close = _load_index(settings.NIFTY50_SYMBOL,   from_dt, to_dt)
    vix_close   = _load_index(settings.INDIA_VIX_SYMBOL, from_dt, to_dt)
    gold_close  = _load_goldbees(date(2018, 1, 1), to_dt)  # GOLDBEES from 2018 for defensive

    universe = _load_universe(ohlcv_dict)
    print(f"Universe: {len(universe)} symbols | Nifty: {len(nifty_close)} days")

    # ── 1. Run monthly momentum on ₹3L ───────────────────────────────────────
    print(f"\nRunning Monthly Momentum (₹{MOM_CAPITAL:,.0f}) {FROM_YEAR}–{TO_YEAR} ...")
    mom_eq, _ = run_monthly_momentum(
        ohlcv_dict=ohlcv_dict,
        nifty_close=nifty_close,
        universe_symbols=universe,
        vix_close=vix_close,
        gold_close=gold_close,
        start_capital=float(MOM_CAPITAL),
        top_n=10, lookback_months=12, skip_months=1,
        abs_momentum=True, abs_lookback_months=6,
        require_200d_ema=True, max_per_sector=3,
        inv_vol_weight=False,
        require_nifty_200ema=True,
        defensive_mode="liquid",
        liquid_rate=LIQUID_RATE,
        vix_reduce=25.0, vix_exit=35.0,
    )
    mom_eq.index = pd.to_datetime(mom_eq.index)

    # ── 2. Build ORB + LIQUIDBEES equity for the ₹2L portion ─────────────────
    # Before 2020: ₹2L earns LIQUIDBEES yield
    # From 2020: ₹2L deployed in ORB LONG (year-by-year actual P&L)
    orb_eq_by_year = _build_orb_equity(mom_eq.index)

    # ── 3. Combine into one equity curve ─────────────────────────────────────
    combined = _combine(mom_eq, orb_eq_by_year)

    # ── 4. Print results ─────────────────────────────────────────────────────
    _print_report(combined, mom_eq, orb_eq_by_year)


def _build_orb_equity(dates: pd.DatetimeIndex) -> pd.Series:
    """
    Build daily equity for the ₹2L ORB portion.
    Pre-2020: grows at LIQUIDBEES rate.
    2020+: adds ORB P&L lump-sum at year-end (approximation for combining).
    """
    vals   = {}
    equity = float(ORB_CAPITAL)

    prev_date = None
    for dt in sorted(dates):
        yr = dt.year
        # Daily yield for LIQUIDBEES
        if prev_date is not None:
            days = (dt - prev_date).days
            if yr < ORB_FIRST_YEAR:
                equity *= (1 + LIQUID_RATE) ** (days / 365.25)

        # At year-end, add ORB P&L (for years with ORB data)
        if prev_date is not None and dt.year != prev_date.year:
            prev_yr = prev_date.year
            if prev_yr >= ORB_FIRST_YEAR and prev_yr in ORB_PNL_2L:
                equity += ORB_PNL_2L[prev_yr]

        vals[dt]  = equity
        prev_date = dt

    # Add 2026 partial year ORB P&L at the last date
    if 2026 in ORB_PNL_2L and ORB_FIRST_YEAR <= 2026:
        last = sorted(vals)[-1]
        if last.year == 2026:
            vals[last] += ORB_PNL_2L[2026]

    return pd.Series(vals)


def _combine(mom_eq: pd.Series, orb_eq: pd.Series) -> pd.Series:
    """Add momentum equity and ORB equity, aligned by date."""
    aligned = pd.concat([mom_eq.rename("mom"), orb_eq.rename("orb")], axis=1).ffill()
    return aligned["mom"] + aligned["orb"]


def _cagr(series: pd.Series) -> float:
    if len(series) < 2:
        return 0.0
    years = (series.index[-1] - series.index[0]).days / 365.25
    if years < 0.1:
        return 0.0
    return (series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1


def _maxdd(series: pd.Series) -> float:
    roll_max = series.cummax()
    return float(((series - roll_max) / roll_max).min())


def _annual_return(eq: pd.Series, yr: int) -> float | None:
    s = eq[eq.index.year == yr]
    if len(s) < 2:
        return None
    # Find previous year-end
    prev = eq[eq.index.year == yr - 1]
    start = float(prev.iloc[-1]) if len(prev) > 0 else float(s.iloc[0])
    return float(s.iloc[-1]) / start - 1


def _print_report(combined: pd.Series, mom_eq: pd.Series, orb_eq: pd.Series) -> None:
    is_mask  = combined.index.year <= IS_END_YEAR
    oos_mask = combined.index.year >  IS_END_YEAR

    is_eq  = combined[is_mask]
    oos_eq = combined[oos_mask]

    sep = "═" * 68

    print(f"\n{sep}")
    print(f"  PIEDPIPER COMBINED SYSTEM  |  ₹{TOTAL_CAPITAL:,.0f} total capital")
    print(f"  Momentum ₹{MOM_CAPITAL:,.0f}  +  ORB LONG ₹{ORB_CAPITAL:,.0f}  +  LIQUIDBEES (idle)")
    print(sep)

    if len(is_eq) > 10:
        print(f"  IS  {FROM_YEAR}–{IS_END_YEAR}: CAGR {_cagr(is_eq)*100:+.1f}%  |  MaxDD {_maxdd(is_eq)*100:.1f}%")
    if len(oos_eq) > 10:
        print(f"  OOS {IS_END_YEAR+1}–{TO_YEAR}: CAGR {_cagr(oos_eq)*100:+.1f}%  |  MaxDD {_maxdd(oos_eq)*100:.1f}%")

    print(f"\n  {'Year':<6} {'Momentum':>12} {'ORB/Liquid':>12} {'Combined':>12} {'% on ₹5L':>10}")
    print(f"  {'────':<6} {'──────────':>12} {'──────────':>12} {'──────────':>12} {'─────────':>10}")

    for yr in range(FROM_YEAR, TO_YEAR + 1):
        mom_r  = _annual_return(mom_eq,  yr)
        orb_r  = _annual_return(orb_eq,  yr)
        comb_r = _annual_return(combined, yr)
        if mom_r is None:
            continue

        mom_tag  = f"{mom_r*100:+.1f}%"
        orb_tag  = f"{orb_r*100:+.1f}%" if orb_r is not None else "—"
        comb_tag = f"{comb_r*100:+.1f}%" if comb_r is not None else "—"
        bar      = "█" * int(abs(comb_r * 100) / 5) if comb_r else ""
        is_oos   = "IS " if yr <= IS_END_YEAR else "OOS"

        print(f"  {yr} {is_oos}  {mom_tag:>10}   {orb_tag:>10}   {comb_tag:>10}  {bar}")

    # Nifty benchmark
    print(f"\n  ── Benchmark comparison ──")
    print(f"  Nifty 50 TRI (buy-hold): ~12% CAGR long-run  |  MaxDD -56% (COVID)")
    print(f"  This system OOS CAGR   : {_cagr(oos_eq)*100:.1f}%             |  MaxDD {_maxdd(oos_eq)*100:.1f}%")

    # Final value
    print(f"\n  ₹{TOTAL_CAPITAL:,.0f} invested {FROM_YEAR} → ₹{combined.iloc[-1]:,.0f} today ({TO_YEAR})")
    total_mult = combined.iloc[-1] / TOTAL_CAPITAL
    print(f"  Total multiplier: {total_mult:.1f}x")
    print(sep)


if __name__ == "__main__":
    main()
