"""
F&O instrument lookup — near-month futures for stock and index symbols.

Reads from the Angel One instrument master (cached daily).
Near-month selection: current month expiry, unless within ROLL_DAYS trading
days of expiry → roll to next month.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import NamedTuple

import pandas as pd
from loguru import logger

ROLL_DAYS = 5   # switch to next month this many trading days before expiry


class FutureInfo(NamedTuple):
    trading_symbol: str   # e.g. RELIANCE25AUG26FUT
    token:          str
    lot_size:       int
    expiry:         date


def _load_futures(instrument_type: str = "FUTSTK") -> pd.DataFrame:
    from data.instrument_master import load_master
    df = load_master()
    fno = df[(df["exch_seg"] == "NFO") & (df["instrumenttype"] == instrument_type)].copy()
    fno["expiry_dt"] = pd.to_datetime(fno["expiry"], format="%d%b%Y", errors="coerce")
    fno = fno.dropna(subset=["expiry_dt"])
    fno["lotsize"] = pd.to_numeric(fno["lotsize"], errors="coerce").fillna(0).astype(int)
    return fno


def _near_month_expiry(expiries: list[date], today: date) -> date | None:
    """
    Pick the nearest expiry that is at least ROLL_DAYS away from today.
    This causes automatic roll-over in expiry week.
    """
    cutoff = today + timedelta(days=ROLL_DAYS)
    future_expiries = sorted(e for e in expiries if e >= cutoff)
    return future_expiries[0] if future_expiries else None


def get_near_month_future(nse_symbol: str, today: date | None = None) -> FutureInfo | None:
    """
    Look up the near-month stock future for an NSE equity symbol.
    Returns None if the symbol has no F&O contract.

    Example: get_near_month_future("RELIANCE") →
        FutureInfo("RELIANCE25AUG26FUT", "58371", 500, date(2026,8,25))
    """
    today = today or date.today()
    fno   = _load_futures("FUTSTK")

    # Match by name (the base symbol without expiry suffix)
    matches = fno[fno["name"] == nse_symbol]
    if matches.empty:
        return None

    expiries = [d.date() for d in matches["expiry_dt"]]
    near     = _near_month_expiry(expiries, today)
    if near is None:
        logger.warning("No valid near-month expiry found for {}", nse_symbol)
        return None

    row = matches[matches["expiry_dt"].dt.date == near].iloc[0]
    return FutureInfo(
        trading_symbol = str(row["symbol"]),
        token          = str(row["token"]),
        lot_size       = int(row["lotsize"]),
        expiry         = near,
    )


def get_index_future(index_name: str = "NIFTY", today: date | None = None) -> FutureInfo | None:
    """
    Look up near-month index future. index_name: "NIFTY", "BANKNIFTY", etc.
    """
    today = today or date.today()
    fno   = _load_futures("FUTIDX")

    matches = fno[fno["name"] == index_name]
    if matches.empty:
        return None

    expiries = [d.date() for d in matches["expiry_dt"]]
    near     = _near_month_expiry(expiries, today)
    if near is None:
        return None

    row = matches[matches["expiry_dt"].dt.date == near].iloc[0]
    return FutureInfo(
        trading_symbol = str(row["symbol"]),
        token          = str(row["token"]),
        lot_size       = int(row["lotsize"]),
        expiry         = near,
    )


def get_fno_eligible_symbols(today: date | None = None) -> set[str]:
    """
    Return the set of NSE equity base symbols that have a near-month stock future.
    Used to filter the short ORB universe to only F&O-able stocks.
    """
    today = today or date.today()
    fno   = _load_futures("FUTSTK")
    cutoff = today + timedelta(days=ROLL_DAYS)
    valid = fno[fno["expiry_dt"].dt.date >= cutoff]
    return set(valid["name"].unique())


def build_fno_token_map(today: date | None = None) -> dict[str, FutureInfo]:
    """
    Build {nse_symbol → FutureInfo} for all F&O eligible stocks in one pass.
    More efficient than calling get_near_month_future() per symbol.
    """
    today = today or date.today()
    fno   = _load_futures("FUTSTK")
    cutoff = today + timedelta(days=ROLL_DAYS)

    # Keep only valid near-month rows
    fno = fno[fno["expiry_dt"].dt.date >= cutoff].copy()

    # For each base symbol pick the nearest expiry
    result: dict[str, FutureInfo] = {}
    for name, grp in fno.groupby("name"):
        expiries = sorted(grp["expiry_dt"].dt.date.tolist())
        near = _near_month_expiry(expiries, today)
        if near is None:
            continue
        row = grp[grp["expiry_dt"].dt.date == near].iloc[0]
        result[str(name)] = FutureInfo(
            trading_symbol = str(row["symbol"]),
            token          = str(row["token"]),
            lot_size       = int(row["lotsize"]),
            expiry         = near,
        )
    return result
