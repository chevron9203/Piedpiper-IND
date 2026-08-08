"""
Populate DuckDB with adjusted OHLCV from EOD2 for all universe symbols
and index data (Nifty 50, India VIX).

Run once before the first backtest; re-run after EOD2 daily updates.

Usage:
    python scripts/ingest_data.py
    python scripts/ingest_data.py --refresh-universe
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
from storage import store
from data.universe import get_universe
from data.eod2_manager import EOD2_DAILY_DIR, load_adjusted_ohlcv, is_eod2_available

# ── Index symbol mapping: EOD2 filename stem → DuckDB symbol name ────────────
INDEX_MAP = {
    "nifty 50":  settings.NIFTY50_SYMBOL,   # "Nifty 50"
    "india vix": settings.INDIA_VIX_SYMBOL, # "India VIX"
}


def _load_index_from_eod2(eod2_stem: str) -> pd.DataFrame:
    """Load index CSV from EOD2 daily dir by exact (lowercase) filename stem."""
    path = EOD2_DAILY_DIR / f"{eod2_stem}.csv"
    if not path.exists():
        logger.warning("Index file not found: {}", path)
        return pd.DataFrame()

    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index).normalize()

    # Keep only standard OHLCV columns
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df["volume"] = df["volume"].fillna(0)
    df = df.dropna(subset=["close"])
    return df.sort_index()


def ingest_universe(universe: pd.DataFrame, from_year: int = 2015) -> None:
    """Load all universe symbols from EOD2 into DuckDB adjusted_ohlcv."""
    from_date = date(from_year, 1, 1)
    symbols = universe["symbol"].tolist()
    ok = 0
    skipped = 0
    missing = []

    logger.info("Ingesting {} universe symbols from EOD2 (from {})...", len(symbols), from_year)

    for sym in symbols:
        bare = sym.upper().replace("-EQ", "")
        try:
            df = load_adjusted_ohlcv(bare, from_date=from_date)
            if df.empty:
                logger.warning("Empty data for {}", sym)
                skipped += 1
                missing.append(sym)
                continue
            rows = store.upsert_adjusted_ohlcv(sym, df, source="eod2")
            ok += 1
            if ok % 20 == 0:
                logger.info("  Progress: {}/{} symbols ingested", ok, len(symbols))
        except FileNotFoundError:
            logger.warning("  {} not in EOD2 — skipping", sym)
            skipped += 1
            missing.append(sym)
        except Exception as exc:
            logger.error("  {} failed: {}", sym, exc)
            skipped += 1
            missing.append(sym)

    logger.info("Universe ingestion complete: {} ok, {} skipped", ok, skipped)
    if missing:
        logger.warning("Missing from EOD2 ({} symbols): {}", len(missing), missing[:10])


def ingest_indices(from_year: int = 2015) -> None:
    """Load Nifty 50 and India VIX from EOD2 into DuckDB."""
    from_date = date(from_year, 1, 1)
    for eod2_stem, db_symbol in INDEX_MAP.items():
        df = _load_index_from_eod2(eod2_stem)
        if df.empty:
            logger.warning("No data for index {} — skipping", db_symbol)
            continue
        df = df[df.index.date >= from_date]
        rows = store.upsert_adjusted_ohlcv(db_symbol, df, source="eod2_index")
        logger.info("  {} → {} rows in DuckDB", db_symbol, rows)


def save_nifty100_symbols(universe: pd.DataFrame) -> None:
    """Write nifty100_symbols.txt used by backtest for slippage tier assignment."""
    # Best proxy: stocks that appear in Nifty 100 (need separate fetch or use universe sector info)
    # For now, read from niftyindices.com same as universe.py does
    try:
        nifty100_df = pd.read_csv(
            settings.NIFTY100_CSV_URL,
            storage_options={"User-Agent": "Mozilla/5.0"},
        )
        col = next((c for c in nifty100_df.columns if "symbol" in c.lower()), None)
        if col:
            n100_syms = [s.strip() + "-EQ" for s in nifty100_df[col].dropna()]
            path = Path(settings.DATA_DIR) / "nifty100_symbols.txt"
            path.write_text("\n".join(n100_syms))
            logger.info("Nifty 100 symbol list saved: {} symbols → {}", len(n100_syms), path)
            return
    except Exception as exc:
        logger.warning("Could not fetch Nifty 100 list: {}. Using universe-based proxy.", exc)

    # Fallback: assume first 100 universe stocks sorted by symbol are Nifty100
    n100_syms = sorted(universe["symbol"].tolist())[:100]
    path = Path(settings.DATA_DIR) / "nifty100_symbols.txt"
    path.write_text("\n".join(n100_syms))
    logger.warning("Used fallback Nifty100 list (first 100 sorted) — slippage tiers may be approximate")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest EOD2 data into DuckDB")
    parser.add_argument("--refresh-universe", action="store_true",
                        help="Force re-fetch universe from niftyindices.com")
    parser.add_argument("--from-year", type=int, default=2015,
                        help="Start year for historical data (default: 2015)")
    args = parser.parse_args()

    if not is_eod2_available():
        logger.error("EOD2 not initialised. Run: python -c \"from data.eod2_manager import setup_eod2; setup_eod2()\"")
        sys.exit(1)

    store.init_schema()

    print("\n" + "="*60)
    print("  PiedPiper — Data Ingestion")
    print("="*60)

    # Step 1: Universe
    print("\n[1] Building universe cache...")
    universe = get_universe(force_refresh=args.refresh_universe)
    print(f"  Universe: {len(universe)} stocks, {universe['sector'].nunique()} sectors")

    # Step 2: Nifty 100 symbols file
    print("\n[2] Saving Nifty 100 symbol list...")
    save_nifty100_symbols(universe)

    # Step 3: Universe OHLCV → DuckDB
    print(f"\n[3] Loading universe OHLCV from EOD2 (from {args.from_year})...")
    ingest_universe(universe, from_year=args.from_year)

    # Step 4: Index data → DuckDB
    print(f"\n[4] Loading index data (Nifty 50, India VIX) from EOD2...")
    ingest_indices(from_year=args.from_year)

    # Step 5: Verify
    print("\n[5] Verification...")
    with store.db_conn() as conn:
        syms_in_db = conn.execute(
            "SELECT COUNT(DISTINCT symbol), MIN(dt), MAX(dt) FROM adjusted_ohlcv"
        ).fetchone()
    print(f"  adjusted_ohlcv: {syms_in_db[0]} symbols, "
          f"date range {syms_in_db[1]} → {syms_in_db[2]}")

    print("\n  Done. Run the backtest:")
    print("  python scripts/backtest_run.py --start-year 2018 --end-year 2024\n")


if __name__ == "__main__":
    main()
