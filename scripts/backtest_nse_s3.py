"""
NSE-S3 backtest: 52-Week HIGH breakout with EMA stack alignment.

Research basis: stocks breaking their 52-week HIGH (not just 63-day close) with
volume show ~70% 10-session continuation probability (IntradayLab, BacktestIndia).
The EMA alignment requirement (50>100>200) filters to structurally trending stocks.

Uses the same simulation engine as NSE-S1 (ATR-based stops, same portfolio rules).

Usage:
    python scripts/backtest_nse_s3.py
    python scripts/backtest_nse_s3.py --slot-pct 0.25 --max-gap 0.003
    python scripts/backtest_nse_s3.py --from-year 2015 --to-year 2026
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from data.universe import get_sector_map, get_nifty100_symbols
from storage import store
from signals.nse_s3_generator import generate_signals

from backtest_nse_s1 import (
    _load_universe_ohlcv, _load_index,
    run_backtest, _print_report,
    MAX_POSITIONS, MAX_DEPLOYED, MAX_HOLD_DAYS, BASE_RISK_PCT, EQ_FILTER_DAYS,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="NSE-S3 backtest (52-week HIGH + EMA stack)")
    ap.add_argument("--capital",        type=float, default=settings.STARTING_VIRTUAL_CAPITAL)
    ap.add_argument("--risk",           type=float, default=BASE_RISK_PCT)
    ap.add_argument("--eq-filter",      type=int,   default=EQ_FILTER_DAYS)
    ap.add_argument("--is-end",         type=int,   default=2020)
    ap.add_argument("--from-year",      type=int,   default=2015)
    ap.add_argument("--to-year",        type=int,   default=2025)
    ap.add_argument("--vix-threshold",  type=float, default=20.0)
    ap.add_argument("--max-hold",       type=int,   default=MAX_HOLD_DAYS)
    ap.add_argument("--sl-mult",        type=float, default=2.5)
    ap.add_argument("--tp-mult",        type=float, default=5.0)
    ap.add_argument("--slot-pct",       type=float, default=MAX_DEPLOYED / MAX_POSITIONS,
                    help=f"Per-position cap as fraction of equity (default {MAX_DEPLOYED/MAX_POSITIONS:.2f})")
    ap.add_argument("--max-gap",        type=float, default=0.003,
                    help="Skip entry if open gaps > this above signal close (default 0.003)")
    # S3 signal parameters
    ap.add_argument("--bo-window",      type=int,   default=252,
                    help="HIGH-based breakout lookback in days (default 252 = 52 weeks)")
    ap.add_argument("--vol-mult",       type=float, default=2.0)
    ap.add_argument("--rsi-low",        type=float, default=55.0)
    ap.add_argument("--rsi-high",       type=float, default=78.0)
    ap.add_argument("--min-score",      type=int,   default=2,
                    help="Min of 3 scored gates (default 2 — 52w HIGH is already the hard filter)")
    ap.add_argument("--nifty-ema-mid",  type=int,   default=50)
    args = ap.parse_args()

    from_dt = date(args.from_year, 1, 1)
    to_dt   = date(args.to_year, 12, 31)

    print(f"\nLoading data {args.from_year}–{args.to_year} ...")
    ohlcv_dict, symbols = _load_universe_ohlcv(from_dt, to_dt)
    nifty_df = _load_index(settings.NIFTY50_SYMBOL,   from_dt, to_dt)
    vix_df   = _load_index(settings.INDIA_VIX_SYMBOL, from_dt, to_dt)

    sector_df     = get_sector_map()
    sector_map    = dict(zip(sector_df["symbol"], sector_df["sector"]))
    nifty100_syms = get_nifty100_symbols()

    if nifty_df.empty:
        print("ERROR: Nifty 50 data missing. Run: python scripts/ingest_data.py")
        sys.exit(1)
    if vix_df.empty:
        import numpy as np
        vix_df = pd.DataFrame(
            {"open": 0, "high": 0, "low": 0, "close": 0, "volume": 0},
            index=nifty_df.index,
        )

    print(f"Generating NSE-S3 signals (252d HIGH breakout + EMA stack) for {len(symbols)} symbols ...")
    signals = generate_signals(
        ohlcv_dict=ohlcv_dict,
        nifty_df=nifty_df,
        vix_df=vix_df,
        universe_symbols=symbols,
        vix_threshold=args.vix_threshold,
        nifty_ema_mid=args.nifty_ema_mid,
        breakout_window=args.bo_window,
        vol_mult=args.vol_mult,
        rsi_low=args.rsi_low,
        rsi_high=args.rsi_high,
        min_score=args.min_score,
    )

    if signals.empty:
        print("No signals generated. Check data or relax filters.")
        return

    n_sig   = len(signals)
    n_dates = signals["date"].nunique()
    print(f"  {n_sig} signals across {n_dates} signal dates "
          f"(avg {n_sig/max(n_dates,1):.1f}/day)")

    atr_pct = (signals["atr"] / signals["close"]).mean()
    print(f"\nSignal quality:")
    print(f"  Avg ATR%:   {atr_pct*100:.2f}%  "
          f"(stop {args.sl_mult*atr_pct*100:.2f}%, target {args.tp_mult*atr_pct*100:.2f}%)")
    print(f"  Score dist: {dict(signals['score'].value_counts().sort_index())}")

    print("Running simulation ...")
    eq_curve, trades = run_backtest(
        signals=signals,
        ohlcv_dict=ohlcv_dict,
        sector_map=sector_map,
        nifty100_syms=nifty100_syms,
        start_capital=args.capital,
        base_risk_pct=args.risk,
        eq_filter_days=args.eq_filter,
        max_hold_days=args.max_hold,
        sl_mult=args.sl_mult,
        tp_mult=args.tp_mult,
        max_gap_pct=args.max_gap,
        slot_pct=args.slot_pct,
    )

    label = (
        f"[S3] 52w-HIGH+EMA-stack | slot={args.slot_pct*100:.0f}% "
        f"| hold={args.max_hold}d | bo={args.bo_window}d-HIGH "
        f"| score≥{args.min_score} | gap<{args.max_gap*100:.1f}%"
    )
    _print_report(eq_curve, trades, args.capital, args.is_end, label)


if __name__ == "__main__":
    main()
