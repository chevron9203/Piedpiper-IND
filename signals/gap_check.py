"""
Opening-gap check.

At 9:15 AM IST, the actual NSE open can deviate from yesterday's close
(and from our planned entry zone). This module flags such gaps so the
daily report can highlight positions that may need to be skipped or
re-evaluated before entry.

Gap-status vocabulary
---------------------
"ok"                   — actual open is within the entry zone (± tolerance)
"gapped_up_beyond_entry" — stock opened above entry_high * (1 + tolerance_pct)
"gapped_down"          — stock opened below entry_low * (1 - tolerance_pct)
"pending"              — actual open price was not available (pre-open or API failure)
"""
from __future__ import annotations

from loguru import logger

import pandas as pd

from config.settings import OPENING_GAP_FLAG_PCT


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_opening_gap(
    signals: pd.DataFrame,
    actual_opens: dict[str, float],
    tolerance_pct: float = OPENING_GAP_FLAG_PCT,
) -> pd.DataFrame:
    """
    For each signal, classify whether the actual open price falls within the
    planned entry zone.

    Parameters
    ----------
    signals : pd.DataFrame
        Today's signals. Must contain columns: symbol, entry_low, entry_high.
    actual_opens : dict[str, float]
        Mapping of {symbol: actual_open_price} from a live quote source.
        Symbols absent from this dict receive gap_status="pending".
    tolerance_pct : float
        Allowable deviation beyond the entry zone before flagging.
        Default 0.005 (0.5%) per OPENING_GAP_FLAG_PCT.

    Returns
    -------
    pd.DataFrame
        Copy of ``signals`` with two additional columns:
          - actual_open   : float  (NaN if unavailable)
          - gap_status    : str    "ok" | "gapped_up_beyond_entry" | "gapped_down" | "pending"
    """
    if signals.empty:
        signals = signals.copy()
        signals["actual_open"] = pd.Series(dtype=float)
        signals["gap_status"] = pd.Series(dtype=str)
        return signals

    result = signals.copy()
    actual_opens_col: list[float | None] = []
    gap_status_col: list[str] = []

    for _, row in result.iterrows():
        symbol: str = row["symbol"]
        open_price = actual_opens.get(symbol)

        if open_price is None or pd.isna(open_price):
            actual_opens_col.append(float("nan"))
            gap_status_col.append("pending")
            continue

        entry_low: float = row["entry_low"]
        entry_high: float = row["entry_high"]

        upper_limit = entry_high * (1.0 + tolerance_pct)
        lower_limit = entry_low * (1.0 - tolerance_pct)

        if open_price > upper_limit:
            status = "gapped_up_beyond_entry"
        elif open_price < lower_limit:
            status = "gapped_down"
        else:
            status = "ok"

        actual_opens_col.append(float(open_price))
        gap_status_col.append(status)

        if status != "ok":
            logger.warning(
                "gap_check: {} — open=₹{:.2f} vs zone=[{:.2f}, {:.2f}] → {}",
                symbol,
                open_price,
                entry_low,
                entry_high,
                status,
            )

    result["actual_open"] = actual_opens_col
    result["gap_status"] = gap_status_col

    n_flagged = sum(1 for s in gap_status_col if s != "ok" and s != "pending")
    n_pending = sum(1 for s in gap_status_col if s == "pending")
    logger.info(
        "gap_check complete: {}/{} signals flagged, {} pending",
        n_flagged,
        len(result),
        n_pending,
    )

    return result


def fetch_opening_prices(symbols: list[str], smart_client) -> dict[str, float]:
    """
    Fetch last-traded / live price for each symbol via SmartAPI getLTP.

    Parameters
    ----------
    symbols : list[str]
        NSE ticker symbols to quote.
    smart_client : SmartConnect
        An authenticated SmartConnect instance from data.auth.get_client().

    Returns
    -------
    dict[str, float]
        {symbol: price}. Missing or failed symbols are silently omitted;
        the caller treats absent symbols as "pending".
    """
    prices: dict[str, float] = {}

    if not symbols:
        return prices

    for symbol in symbols:
        try:
            # getLTP expects exchange, tradingsymbol, symboltoken
            # We use NSE cash segment; token lookup would normally come from
            # instrument_master but we attempt a best-effort call here.
            resp = smart_client.ltpData("NSE", symbol, "")
            if resp and resp.get("status") and resp.get("data"):
                ltp = resp["data"].get("ltp")
                if ltp is not None:
                    prices[symbol] = float(ltp)
                    logger.debug("fetch_opening_prices: {}=₹{:.2f}", symbol, ltp)
            else:
                logger.debug(
                    "fetch_opening_prices: no LTP data returned for {} (resp={})",
                    symbol,
                    resp,
                )
        except Exception as exc:
            logger.warning(
                "fetch_opening_prices: failed for {} — {} (gap check will show 'pending')",
                symbol,
                exc,
            )

    logger.info(
        "fetch_opening_prices: retrieved {}/{} prices",
        len(prices),
        len(symbols),
    )
    return prices
