"""
SmartAPI instrument master — maps trading symbols to symboltokens.

The instrument master JSON is downloaded once and cached locally.
Refresh it daily (it changes when new instruments are listed/delisted).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from datetime import date
import pandas as pd
import requests
from loguru import logger
from config import settings

MASTER_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
CACHE_PATH = settings.DATA_DIR / "instrument_master.json"
CACHE_DATE_PATH = settings.DATA_DIR / "instrument_master_date.txt"


def _cache_is_fresh() -> bool:
    if not CACHE_PATH.exists() or not CACHE_DATE_PATH.exists():
        return False
    cached_date = CACHE_DATE_PATH.read_text().strip()
    return cached_date == str(date.today())


def refresh_master(force: bool = False) -> None:
    """Download and cache the instrument master if stale or forced."""
    if not force and _cache_is_fresh():
        logger.debug("Instrument master cache is fresh (today)")
        return

    logger.info("Downloading instrument master from Angel One...")
    resp = requests.get(MASTER_URL, timeout=60)
    resp.raise_for_status()
    CACHE_PATH.write_bytes(resp.content)
    CACHE_DATE_PATH.write_text(str(date.today()))
    logger.info("Instrument master cached ({:.1f} MB)", len(resp.content) / 1e6)


def load_master() -> pd.DataFrame:
    """Load the cached instrument master as a DataFrame."""
    refresh_master()
    with open(CACHE_PATH) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    return df


def get_nse_equity_master() -> pd.DataFrame:
    """Return only NSE equity (cash segment) instruments."""
    df = load_master()
    mask = (df["exch_seg"] == "NSE") & (df["instrumenttype"] == "")
    return df[mask][["symbol", "name", "token", "lotsize", "tick_size", "exch_seg"]].copy()


def symbol_to_token(symbol: str) -> str | None:
    """Look up symboltoken for an NSE equity symbol. Returns None if not found."""
    df = get_nse_equity_master()
    match = df[df["symbol"] == symbol]
    if match.empty:
        # Try with -EQ suffix
        match = df[df["symbol"] == f"{symbol}-EQ"]
    if match.empty:
        logger.warning("Symbol not found in instrument master: {}", symbol)
        return None
    return str(match.iloc[0]["token"])


def token_to_symbol(token: str) -> str | None:
    """Reverse lookup: token → symbol."""
    df = get_nse_equity_master()
    match = df[df["token"] == str(token)]
    if match.empty:
        return None
    return str(match.iloc[0]["symbol"])


def verify_india_vix() -> dict:
    """
    Phase 0 check: confirm India VIX token in the current master.
    Returns the matching row(s) as a dict, or raises if not found.
    """
    df = load_master()
    vix_matches = df[df["symbol"].str.contains("India VIX", case=False, na=False)]
    if vix_matches.empty:
        vix_matches = df[df["token"] == settings.INDIA_VIX_TOKEN]
    if vix_matches.empty:
        raise ValueError(
            f"India VIX not found in instrument master. "
            f"Hardcoded token {settings.INDIA_VIX_TOKEN!r} may be stale — check manually."
        )
    result = vix_matches.iloc[0].to_dict()
    logger.info("India VIX confirmed: symbol={}, token={}", result.get("symbol"), result.get("token"))
    return result


if __name__ == "__main__":
    refresh_master(force=True)
    equity = get_nse_equity_master()
    logger.info("Loaded {} NSE equity instruments", len(equity))
    vix = verify_india_vix()
    logger.info("VIX: {}", vix)
    tok = symbol_to_token("RELIANCE-EQ")
    logger.info("RELIANCE token: {}", tok)
