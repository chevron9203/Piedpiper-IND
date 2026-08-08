"""
Backfill 2008–2014 daily OHLCV from yfinance into DuckDB.

Why: eod2 data starts 2015. Monthly momentum IS period extends to 2008
for a proper 7-year in-sample + 6-year OOS split.

What it downloads:
  - All current Nifty 500 stocks (.NS symbols) — survivorship-biased pre-2015
  - Nifty 50 index (^NSEI) → stored as "Nifty 50"
  - India VIX (^INDIAVIX) → stored as "India VIX"

Stored in adjusted_ohlcv with source='yfinance'. The backtest queries both
eod2 and yfinance rows, eod2 wins on overlap.

Usage:
    python scripts/backfill_historical.py
    python scripts/backfill_historical.py --from-year 2005 --to-year 2014
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from storage import store


def _download_symbol(sym_yf: str, sym_db: str, from_dt: date, to_dt: date) -> pd.DataFrame:
    """Download one symbol from yfinance. Returns cleaned DataFrame (not written to DB yet)."""
    try:
        import yfinance as yf
        df = yf.download(sym_yf, start=from_dt.isoformat(), end=to_dt.isoformat(),
                         progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]

        df = df[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
        df.index = pd.to_datetime(df.index).normalize()
        df.index.name = "dt"
        df = df.reset_index()
        df["symbol"] = sym_db
        df["source"] = "yfinance"
        df["dt"] = df["dt"].dt.date
        return df
    except Exception as exc:
        logger.debug("Failed {}: {}", sym_yf, exc)
        return pd.DataFrame()


def _get_nifty500_yf_symbols() -> list[tuple[str, str]]:
    """Return [(yf_symbol, db_symbol)] for Nifty 500 stocks."""
    try:
        import requests
        url = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
        resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        import io
        df = pd.read_csv(io.StringIO(resp.text))
        syms = df["Symbol"].dropna().tolist()
        # db_symbol must match eod2 format: plain ticker, no suffix (e.g. "RELIANCE")
        pairs = [(s.strip() + ".NS", s.strip()) for s in syms if s.strip()]
        logger.info("Loaded {} Nifty 500 symbols from NSE", len(pairs))
        return pairs
    except Exception as exc:
        logger.warning("Could not load Nifty 500 from NSE ({}), using hardcoded core set", exc)
        core = [
            "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR",
            "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "LT", "AXISBANK",
            "ASIANPAINT", "MARUTI", "SUNPHARMA", "TITAN", "BAJFINANCE",
            "WIPRO", "HCLTECH", "ULTRACEMCO", "NTPC", "POWERGRID", "TECHM",
            "NESTLEIND", "M&M", "JSWSTEEL", "TATASTEEL", "BAJAJ-AUTO",
            "HDFCLIFE", "BRITANNIA", "GRASIM", "DIVISLAB", "CIPLA",
            "DRREDDY", "ONGC", "COALINDIA", "ADANIPORTS", "BPCL", "EICHERMOT",
            "APOLLOHOSP", "HINDALCO", "BAJAJFINSV", "HEROMOTOCO", "TATACONSUM",
            "TATAMOTORS", "ZOMATO", "BAJAJHLDNG", "BOSCHLTD", "ABBOTINDIA",
            "PIDILITIND", "MUTHOOTFIN", "PAGEIND", "CHOLAFIN", "TORNTPHARM",
            "BERGEPAINT", "DABUR", "GODREJCP", "MARICO", "COLPAL", "HAVELLS",
        ]
        # No suffix — matches eod2 naming
        return [(s + ".NS", s) for s in core]


def _ensure_source_column() -> None:
    """Add 'source' column to adjusted_ohlcv if missing (older schema)."""
    with store.db_conn() as conn:
        # DuckDB: query information_schema, not PRAGMA
        cols_df = conn.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'adjusted_ohlcv'
        """).df()
        if "source" not in cols_df["column_name"].tolist():
            conn.execute("ALTER TABLE adjusted_ohlcv ADD COLUMN source TEXT DEFAULT 'eod2'")
            logger.info("Added 'source' column to adjusted_ohlcv")


def _bulk_insert(frames: list[pd.DataFrame]) -> int:
    """Concatenate all frames and insert into DuckDB in one transaction."""
    if not frames:
        return 0
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["symbol", "dt"])

    with store.db_conn() as conn:
        # DuckDB ON CONFLICT syntax (not INSERT OR IGNORE)
        conn.execute("""
            INSERT INTO adjusted_ohlcv (symbol, dt, open, high, low, close, volume, source)
            SELECT symbol, dt, open, high, low, close, volume, source
            FROM combined
            ON CONFLICT (symbol, dt) DO NOTHING
        """)
    return len(combined)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-year", type=int, default=2008)
    ap.add_argument("--to-year",   type=int, default=2014)
    ap.add_argument("--max-workers", type=int, default=5)
    ap.add_argument("--skip-index", action="store_true")
    args = ap.parse_args()

    from_dt = date(args.from_year, 1, 1)
    to_dt   = date(args.to_year, 12, 31)

    store.init_schema()
    _ensure_source_column()

    all_frames: list[pd.DataFrame] = []

    # 1. Index data
    if not args.skip_index:
        for yf_sym, db_sym in [("^NSEI", "Nifty 50"), ("^INDIAVIX", "India VIX")]:
            df = _download_symbol(yf_sym, db_sym, from_dt, to_dt)
            if not df.empty:
                all_frames.append(df)
                logger.info("{} → {} rows", db_sym, len(df))

    # 2. Download equities in parallel — collect into memory, insert in one shot
    pairs = _get_nifty500_yf_symbols()
    logger.info("Downloading {} equity symbols ({} to {}) with {} threads ...",
                len(pairs), from_dt, to_dt, args.max_workers)

    done = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(_download_symbol, yf_sym, db_sym, from_dt, to_dt): db_sym
            for yf_sym, db_sym in pairs
        }
        for fut in as_completed(futures):
            db_sym = futures[fut]
            df = fut.result()
            if not df.empty:
                all_frames.append(df)
            done += 1
            if done % 100 == 0 or done == len(pairs):
                logger.info("  {}/{} downloaded | {} frames collected",
                            done, len(pairs), len(all_frames))

    # 3. Single-threaded bulk insert
    logger.info("Inserting {} symbol-frames into DuckDB ...", len(all_frames))
    rows = _bulk_insert(all_frames)
    logger.info("Backfill complete — {} rows written (source=yfinance)", rows)

    # Summary
    with store.db_conn() as conn:
        stats = conn.execute("""
            SELECT source,
                   MIN(dt)              AS min_dt,
                   MAX(dt)              AS max_dt,
                   COUNT(DISTINCT symbol) AS n_symbols,
                   COUNT(*)             AS n_rows
            FROM adjusted_ohlcv
            GROUP BY source
            ORDER BY source
        """).df()
    print("\nadjusted_ohlcv coverage:")
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
