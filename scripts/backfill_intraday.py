"""
Backfill 2 years of 15-min candle data into DuckDB for all Nifty 200 symbols.

Fetches via Angel One (if token map available) then falls back to yfinance
(yfinance only covers the last ~60 days at 15-min — Angel One covers 2+ years).

Usage:
    python scripts/backfill_intraday.py              # 2-year backfill, all Nifty 200
    python scripts/backfill_intraday.py --from-year 2023  # from Jan 1 2023
    python scripts/backfill_intraday.py --quick           # Nifty 50 only
    python scripts/backfill_intraday.py --symbol RELIANCE # single symbol

After this runs, backtest_orb_multi.py will use stored data (60+ days) automatically.
"""
from __future__ import annotations

import argparse
import sys
import time as time_module
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fetch_historical_candles import fetch_and_store
from storage import store

NIFTY50 = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "SBIN",
    "BHARTIARTL", "ITC", "KOTAKBANK", "LT", "AXISBANK", "ASIANPAINT", "MARUTI",
    "SUNPHARMA", "TITAN", "BAJFINANCE", "WIPRO", "HCLTECH", "ULTRACEMCO",
    "NTPC", "POWERGRID", "TECHM", "NESTLEIND", "M&M", "JSWSTEEL", "TATASTEEL",
    "INDUSINDBK", "BAJAJ-AUTO", "HDFCLIFE", "BRITANNIA", "GRASIM", "DIVISLAB",
    "CIPLA", "DRREDDY", "ONGC", "COALINDIA", "ADANIPORTS", "BPCL", "EICHERMOT",
    "APOLLOHOSP", "HINDALCO", "BAJAJFINSV", "HEROMOTOCO", "SHRIRAMFIN",
    "TATACONSUM", "SBILIFE", "VEDL",
]


def _get_universe() -> list[str]:
    try:
        from data.universe import fetch_nifty200
        df = fetch_nifty200()
        return df["symbol"].tolist()
    except Exception as exc:
        logger.warning("Nifty 200 fetch failed ({}), using Nifty 50", exc)
        return NIFTY50


def _get_token_map() -> dict[str, str]:
    try:
        from data.instrument_master import get_nse_equity_master
        master = get_nse_equity_master()
        return dict(zip(master["symbol"].str.replace("-EQ", ""), master["token"].astype(str)))
    except Exception:
        return {}


def _backfill_one(sym: str, token: str, from_date: date, to_date: date) -> tuple[str, int]:
    try:
        rows = fetch_and_store(sym, token=token, from_date=from_date, to_date=to_date)
        return sym, rows
    except Exception as exc:
        logger.warning("Backfill failed for {}: {}", sym, exc)
        return sym, 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill 15-min intraday candles")
    ap.add_argument("--from-year", type=int, default=None,
                    help="Start year (default: 2 years ago)")
    ap.add_argument("--quick",   action="store_true", help="Nifty 50 only")
    ap.add_argument("--symbol",  type=str,  default=None, help="Single symbol to backfill")
    ap.add_argument("--workers", type=int,  default=1,    help="Parallel threads (default 1 — Angel One rate limits)")
    args = ap.parse_args()

    store.init_schema()

    today = date.today()
    if args.from_year:
        from_date = date(args.from_year, 1, 1)
    else:
        from_date = date(today.year - 2, today.month, today.day)
    to_date = today

    logger.info("Backfill 15-min candles: {} → {}", from_date, to_date)

    if args.symbol:
        symbols = [args.symbol.upper()]
    elif args.quick:
        symbols = NIFTY50
    else:
        symbols = _get_universe()

    token_map = _get_token_map()
    if not token_map:
        logger.warning("No Angel One token map — will use yfinance (covers last ~60 days only)")
    else:
        logger.info("Token map loaded: {} symbols", len(token_map))

    logger.info("Backfilling {} symbols with {} workers ...", len(symbols), args.workers)

    total_rows = 0
    done       = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_backfill_one, sym, token_map.get(sym, ""), from_date, to_date): sym
            for sym in symbols
        }
        for fut in as_completed(futures):
            sym, rows = fut.result()
            total_rows += rows
            done += 1
            if done % 20 == 0 or done == len(symbols):
                logger.info("  {}/{} done | {} total rows stored", done, len(symbols), total_rows)
            # Pause between symbols to stay under Angel One rate limits
            time_module.sleep(0.5)

    logger.info("Backfill complete. {} symbols | {} rows stored", len(symbols), total_rows)

    # Summary
    try:
        with store.db_conn() as conn:
            stats = conn.execute("""
                SELECT symbol, COUNT(*) AS bars,
                       MIN(dt) AS from_dt, MAX(dt) AS to_dt
                FROM intraday_ohlcv
                WHERE interval = '15m'
                GROUP BY symbol
                ORDER BY bars DESC
                LIMIT 10
            """).df()
        print("\nTop 10 symbols by bar count (15m):")
        print(stats.to_string(index=False))
    except Exception:
        pass


if __name__ == "__main__":
    main()
