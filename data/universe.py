"""
Universe screener.

Fetches Nifty 200 constituents from niftyindices.com, builds sector map
from sectoral index CSVs, applies liquidity / corporate-action / sector
filters, and returns a filtered universe with sector labels.

Run quarterly (after index rebalancing) to keep universe current.
"""
from __future__ import annotations

import io
from datetime import date, timedelta
from pathlib import Path
import pandas as pd
import requests
from loguru import logger
from config import settings

CACHE_PATH = settings.DATA_DIR / "universe_cache.parquet"
SECTOR_MAP_PATH = settings.DATA_DIR / "sector_map.parquet"

NIFTY100_SYMBOLS_PATH = settings.DATA_DIR / "nifty100_symbols.txt"

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; piedpiper/1.0)",
    "Referer": "https://www.niftyindices.com/",
}


def _fetch_index_csv(url: str) -> pd.DataFrame:
    """Download and parse a niftyindices.com constituent CSV."""
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=20)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        return df
    except Exception as exc:
        logger.error("Failed to fetch {}: {}", url, exc)
        raise


def fetch_nifty200() -> pd.DataFrame:
    """Fetch current universe constituents (controlled by UNIVERSE_CSV_URL in settings)."""
    url = getattr(settings, "UNIVERSE_CSV_URL", settings.NIFTY200_CSV_URL)
    logger.info("Fetching universe from {} ...", url.split("/")[-1])
    df = _fetch_index_csv(url)

    # niftyindices CSV has 'symbol' or 'ticker' column
    for col_candidate in ("symbol", "ticker", "nsesymbol", "nse_symbol"):
        if col_candidate in df.columns:
            df = df.rename(columns={col_candidate: "symbol"})
            break

    df["symbol"] = df["symbol"].str.strip().str.upper()
    logger.info("Nifty 200 raw: {} stocks", len(df))
    return df[["symbol"] + [c for c in df.columns if c != "symbol"]]


def fetch_nifty100_symbols() -> set[str]:
    """Fetch Nifty 100 symbols (used for slippage tier classification)."""
    df = _fetch_index_csv(settings.NIFTY100_CSV_URL)
    for col_candidate in ("symbol", "ticker", "nsesymbol"):
        if col_candidate in df.columns:
            symbols = set(df[col_candidate].str.strip().str.upper())
            NIFTY100_SYMBOLS_PATH.write_text("\n".join(sorted(symbols)))
            return symbols
    return set()


def build_sector_map() -> pd.DataFrame:
    """
    Build symbol → sector mapping from niftyindices sectoral index CSVs.
    A stock's sector is the first sectoral index it appears in.
    """
    logger.info("Building sector map from {} sectoral indices...", len(settings.SECTORAL_INDEX_URLS))
    rows = []
    for sector, url in settings.SECTORAL_INDEX_URLS.items():
        try:
            df = _fetch_index_csv(url)
            for col_candidate in ("symbol", "ticker", "nsesymbol"):
                if col_candidate in df.columns:
                    for sym in df[col_candidate].str.strip().str.upper():
                        rows.append({"symbol": sym, "sector": sector})
                    break
        except Exception as exc:
            logger.warning("Skipping sector {}: {}", sector, exc)

    sector_df = pd.DataFrame(rows)
    # Keep first occurrence (priority ordering in SECTORAL_INDEX_URLS)
    sector_df = sector_df.drop_duplicates(subset="symbol", keep="first")

    sector_df.to_parquet(SECTOR_MAP_PATH, index=False)
    logger.info("Sector map: {} stocks mapped", len(sector_df))
    return sector_df


def get_sector_map() -> pd.DataFrame:
    if SECTOR_MAP_PATH.exists():
        return pd.read_parquet(SECTOR_MAP_PATH)
    return build_sector_map()


def get_nifty100_symbols() -> set[str]:
    if NIFTY100_SYMBOLS_PATH.exists():
        return set(NIFTY100_SYMBOLS_PATH.read_text().splitlines())
    return fetch_nifty100_symbols()


def screen_universe(
    nifty200: pd.DataFrame,
    ohlcv_cache: dict[str, pd.DataFrame] | None = None,
    exclude_recent_ca_symbols: set[str] | None = None,
    min_turnover_cr: float = settings.MIN_AVG_DAILY_TURNOVER_CR,
    sector_cap_pct: float = settings.UNIVERSE_SECTOR_CAP_PCT,
) -> pd.DataFrame:
    """
    Apply filters to Nifty 200 pool:
    1. Liquidity: average daily turnover >= min_turnover_cr crores
    2. Corporate action exclusion: remove stocks with recent CA
    3. Sector cap: no more than sector_cap_pct from one sector

    ohlcv_cache: {symbol: daily_ohlcv_df} used for turnover filter.
    If None, skip turnover filter (useful for initial universe build before data is loaded).
    """
    df = nifty200.copy()

    # Join sector labels
    sector_map = get_sector_map()
    df = df.merge(sector_map[["symbol", "sector"]], on="symbol", how="left")
    df["sector"] = df["sector"].fillna("Other")

    # Corporate action exclusion
    if exclude_recent_ca_symbols:
        before = len(df)
        df = df[~df["symbol"].isin(exclude_recent_ca_symbols)]
        logger.info("CA exclusion removed {} stocks", before - len(df))

    # Liquidity filter
    if ohlcv_cache:
        turnovers = {}
        for sym, ohlcv in ohlcv_cache.items():
            if "close" in ohlcv.columns and "volume" in ohlcv.columns:
                ohlcv = ohlcv.tail(60)  # 3-month average
                avg_turnover_cr = (ohlcv["close"] * ohlcv["volume"]).mean() / 1e7  # ₹ → crores
                turnovers[sym] = avg_turnover_cr

        df["avg_turnover_cr"] = df["symbol"].map(turnovers)
        before = len(df)
        df = df[df["avg_turnover_cr"].fillna(0) >= min_turnover_cr]
        logger.info("Liquidity filter removed {} stocks (threshold: ₹{}Cr)", before - len(df), min_turnover_cr)

    # Sector cap
    total = len(df)
    max_per_sector = int(total * sector_cap_pct)
    sector_counts = df["sector"].value_counts()
    capped_sectors = sector_counts[sector_counts > max_per_sector].index.tolist()

    if capped_sectors:
        logger.info("Applying sector cap to: {} (max {} per sector)", capped_sectors, max_per_sector)
        rows_to_keep = []
        for sector in df["sector"].unique():
            sector_df = df[df["sector"] == sector]
            if sector in capped_sectors and "avg_turnover_cr" in sector_df.columns:
                sector_df = sector_df.nlargest(max_per_sector, "avg_turnover_cr")
            rows_to_keep.append(sector_df)
        df = pd.concat(rows_to_keep).drop_duplicates("symbol")

    logger.info("Final universe: {} stocks across {} sectors",
                len(df), df["sector"].nunique())
    return df.reset_index(drop=True)


def get_universe(force_refresh: bool = False) -> pd.DataFrame:
    """
    Load the filtered universe, using cache if fresh (same day).
    Run with force_refresh=True after index rebalancing.
    """
    if not force_refresh and CACHE_PATH.exists():
        cached = pd.read_parquet(CACHE_PATH)
        # Check if cache is from today
        if "cached_date" in cached.attrs and cached.attrs["cached_date"] == str(date.today()):
            return cached
        # Fall through to refresh

    nifty200 = fetch_nifty200()
    universe = screen_universe(nifty200)
    universe.attrs["cached_date"] = str(date.today())
    universe.to_parquet(CACHE_PATH, index=False)
    return universe


if __name__ == "__main__":
    u = get_universe(force_refresh=True)
    logger.info("Universe:\n{}", u[["symbol", "sector"]].to_string())
    logger.info("Sector distribution:\n{}", u["sector"].value_counts().to_string())
