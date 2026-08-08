"""
Historical 15-min candle fetcher for NSE symbols.

Fetches via Angel One SmartAPI getCandleData (primary) or yfinance (fallback).
Angel One limits one call to 30 days of 15-min data — loops in 30-day chunks.
Stores in DuckDB intraday_ohlcv table via store.upsert_intraday_ohlcv.

Usage (called by backfill_intraday.py or standalone):
    from data.fetch_historical_candles import fetch_and_store
    rows = fetch_and_store("RELIANCE", token="99926000", from_date=date(2024,1,1))
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone, time
from pathlib import Path
import sys

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

IST = timezone(timedelta(hours=5, minutes=30))
CHUNK_DAYS        = 29    # Angel One max window per call (30 days inclusive)
INTERVAL          = "15m"
COVERAGE_THRESHOLD = 0.70  # skip API call if ≥70% of expected bars already stored


def _chunk_already_stored(symbol: str, from_date: date, to_date: date) -> bool:
    """
    True if DuckDB already has ≥70% of expected 15m bars for this date range.
    Avoids redundant API calls on repeated backfill runs.
    """
    try:
        from storage import store
        from_dt = datetime.combine(from_date, time(9, 0))
        to_dt   = datetime.combine(to_date,   time(16, 0))
        df = store.load_intraday_ohlcv(symbol, INTERVAL, from_dt=from_dt, to_dt=to_dt)
        if df.empty:
            return False
        actual_days = df.index.normalize().nunique()
        weekdays = sum(
            1 for i in range((to_date - from_date).days + 1)
            if (from_date + timedelta(days=i)).weekday() < 5
        )
        return actual_days >= max(1, int(weekdays * COVERAGE_THRESHOLD))
    except Exception:
        return False


def _fetch_chunk_angel(token: str, symbol_nse: str,
                       from_dt: datetime, to_dt: datetime) -> pd.DataFrame:
    """
    Fetch one 30-day chunk of 15-min candles from Angel One SmartAPI.
    Returns empty DataFrame on failure.
    """
    try:
        from data.smartapi_client import fetch_candles
        df = fetch_candles(token, "NSE", INTERVAL, from_dt, to_dt)
        if df.empty:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index)
        if df.index.tzinfo is not None:
            df.index = df.index.tz_convert("Asia/Kolkata").tz_localize(None)
        df = df.between_time("09:15", "15:30")
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception as exc:
        logger.debug("Angel One fetch failed for {} {}: {}", symbol_nse, from_dt.date(), exc)
        return pd.DataFrame()


def _fetch_chunk_yfinance(symbol_nse: str, from_dt: date, to_dt: date) -> pd.DataFrame:
    """
    Fetch one chunk from yfinance. Works only for last ~60 days at 15-min.
    Returns empty DataFrame on failure or if data is too old.
    """
    cutoff = date.today() - timedelta(days=59)
    if to_dt < cutoff:
        return pd.DataFrame()   # yfinance won't have this data

    try:
        import yfinance as yf
        sym_yf = symbol_nse + ".NS"
        df = yf.download(
            sym_yf,
            start=from_dt.isoformat(),
            end=(to_dt + timedelta(days=1)).isoformat(),
            interval="15m", progress=False, auto_adjust=True,
        )
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        df.index = pd.to_datetime(df.index)
        if df.index.tzinfo is not None:
            df.index = df.index.tz_convert("Asia/Kolkata").tz_localize(None)
        return df.between_time("09:15", "15:30").dropna(subset=["close"])
    except Exception as exc:
        logger.debug("yfinance fetch failed for {} {}: {}", symbol_nse, from_dt, exc)
        return pd.DataFrame()


def fetch_and_store(
    symbol:    str,         # NSE clean symbol, e.g. "RELIANCE"
    token:     str = "",    # Angel One token (optional — falls back to yfinance if empty)
    from_date: date | None = None,
    to_date:   date | None = None,
) -> int:
    """
    Fetch 15-min historical candles for `symbol` and store in DuckDB.
    Loops in 29-day chunks from from_date to to_date.
    Returns total rows inserted.

    Args:
        symbol:    clean NSE ticker, e.g. "RELIANCE"
        token:     Angel One instrument token (leave empty to use yfinance only)
        from_date: start of history (default: 2 years ago)
        to_date:   end (default: yesterday)
    """
    from storage import store

    if to_date is None:
        to_date = date.today() - timedelta(days=1)
    if from_date is None:
        from_date = to_date - timedelta(days=730)   # 2 years

    total_rows = 0
    cursor     = from_date

    while cursor <= to_date:
        chunk_end = min(cursor + timedelta(days=CHUNK_DAYS), to_date)

        # Skip API call if this chunk is already well-covered in the DB
        if _chunk_already_stored(symbol, cursor, chunk_end):
            logger.debug("{}: chunk {}-{} already stored, skipping", symbol, cursor, chunk_end)
            cursor = chunk_end + timedelta(days=1)
            continue

        # Try Angel One first (works for any date if token valid)
        from_dt = datetime.combine(cursor,    time(9, 15)).replace(tzinfo=IST)
        to_dt   = datetime.combine(chunk_end, time(15, 30)).replace(tzinfo=IST)

        df = pd.DataFrame()
        if token:
            df = _fetch_chunk_angel(token, symbol, from_dt, to_dt)

        # Fall back to yfinance for recent data
        if df.empty:
            df = _fetch_chunk_yfinance(symbol, cursor, chunk_end)

        if not df.empty:
            rows = store.upsert_intraday_ohlcv(symbol, df, INTERVAL)
            total_rows += rows
            logger.debug("{}: {} rows stored ({} → {})", symbol, rows, cursor, chunk_end)

        cursor = chunk_end + timedelta(days=1)

    return total_rows
