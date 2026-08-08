"""
Collect intraday OHLCV candles from Angel One SmartAPI and store in DuckDB.

SmartAPI historical limits (as of 2026):
  1h  — up to 6 months lookback (chunks of 30 days)
  15m — up to 3 months lookback
  30m — up to 3 months lookback

Usage:
    # 1hr candles (maximum history) — run once initially
    python scripts/ingest_intraday.py --interval 1h --days 180

    # Daily top-up (run after market close each day)
    python scripts/ingest_intraday.py --interval 1h --days 5

    # 15-min candles (shorter history available)
    python scripts/ingest_intraday.py --interval 15m --days 90

    # Single symbol test
    python scripts/ingest_intraday.py --interval 1h --symbol RELIANCE-EQ --days 30
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from data.universe import get_universe
from data.smartapi_client import fetch_candles, INTERVAL_MAP
from data.instrument_master import symbol_to_token
from storage import store

# Max history per interval (conservative)
MAX_DAYS: dict[str, int] = {
    "1h":  180,
    "30m": 90,
    "15m": 90,
    "5m":  30,
    "1m":  30,
}


def ingest_symbol(
    symbol: str,
    interval: str,
    from_date: date,
    to_date: date,
) -> int:
    """
    Fetch intraday candles for one symbol and insert into DuckDB.
    Returns number of new rows inserted.
    """
    try:
        token = symbol_to_token(symbol)
    except Exception as exc:
        logger.warning("No token for {}: {}", symbol, exc)
        return 0

    if not token:
        logger.warning("Empty token for {} — skipping", symbol)
        return 0

    try:
        df = fetch_candles(
            symbol_token=token,
            exchange="NSE",
            interval=interval,
            from_date=from_date,
            to_date=to_date,
        )
    except Exception as exc:
        logger.error("Fetch failed for {} @ {}: {}", symbol, interval, exc)
        return 0

    if df.empty:
        return 0

    # Filter to market hours only (09:15–15:30 IST)
    df = df.between_time("09:15", "15:30")
    if df.empty:
        return 0

    inserted = store.upsert_intraday_ohlcv(symbol, df, interval)
    return inserted


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest intraday OHLCV from SmartAPI into DuckDB")
    ap.add_argument("--interval", type=str, default="1h",
                    choices=list(INTERVAL_MAP.keys()) + ["1h", "15m", "30m"],
                    help="Candle interval (default: 1h)")
    ap.add_argument("--days",     type=int, default=180,
                    help="How many calendar days of history to fetch (default: 180)")
    ap.add_argument("--symbol",   type=str, default=None,
                    help="Fetch only this symbol (default: all universe symbols)")
    ap.add_argument("--limit",    type=int, default=None,
                    help="Process only first N symbols (for testing)")
    args = ap.parse_args()

    interval = args.interval
    if interval not in INTERVAL_MAP:
        logger.error("Unknown interval {}. Available: {}", interval, list(INTERVAL_MAP.keys()))
        sys.exit(1)

    max_days = MAX_DAYS.get(interval, 30)
    if args.days > max_days:
        logger.warning(
            "Requested {} days but SmartAPI limit for {} is {} days. Clamping.",
            args.days, interval, max_days
        )
        args.days = max_days

    to_date   = date.today()
    from_date = to_date - timedelta(days=args.days)

    # Ensure schema has intraday table
    store.init_schema()

    print(f"\n{'='*60}")
    print(f"  PiedPiper — Intraday Ingestion")
    print(f"{'='*60}")
    print(f"  Interval : {interval}")
    print(f"  Range    : {from_date} → {to_date} ({args.days} days)")

    if args.symbol:
        symbols = [args.symbol]
    else:
        universe = get_universe()
        symbols  = universe["symbol"].tolist()
        if args.limit:
            symbols = symbols[:args.limit]

    print(f"  Symbols  : {len(symbols)}")
    print()

    total_rows = 0
    ok = 0
    skipped = 0

    for i, sym in enumerate(symbols, 1):
        rows = ingest_symbol(sym, interval, from_date, to_date)
        if rows > 0:
            total_rows += rows
            ok += 1
        else:
            skipped += 1

        if i % 20 == 0:
            logger.info("Progress: {}/{} symbols | {} rows inserted so far", i, len(symbols), total_rows)

        # SmartAPI rate limit: fetch_candles already sleeps 1.2s per chunk
        # Add a small extra buffer between symbols
        if i < len(symbols):
            time.sleep(0.3)

    # Summary
    with store.db_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*), MIN(dt), MAX(dt) FROM intraday_ohlcv WHERE interval = ?",
            [interval]
        ).fetchone()

    print(f"\n{'='*60}")
    print(f"  Done. Inserted {total_rows} new rows.")
    print(f"  Symbols: {ok} with data, {skipped} skipped/empty")
    print(f"  DB total [{interval}]: {row[0]} rows, {row[1]} → {row[2]}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
