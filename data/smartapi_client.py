"""
SmartAPI historical candle fetcher.

Handles pagination (API has a max-days-per-request limit), rate limiting,
and retry logic. Returns clean DataFrames indexed by datetime.
"""
import time
from datetime import datetime, timedelta, timezone, date
import pandas as pd
from loguru import logger
from data.auth import get_client
from config import settings

IST = timezone(timedelta(hours=5, minutes=30))

INTERVAL_MAP = {
    "1d":   "ONE_DAY",
    "1w":   "ONE_WEEK",
    "1M":   "ONE_MONTH",
    "1m":   "ONE_MINUTE",
    "3m":   "THREE_MINUTE",
    "5m":   "FIVE_MINUTE",
    "10m":  "TEN_MINUTE",
    "15m":  "FIFTEEN_MINUTE",
    "30m":  "THIRTY_MINUTE",
    "1h":   "ONE_HOUR",
}

# Conservative max days per request for ONE_DAY interval (verify in Phase 0)
MAX_DAYS_DAILY = settings.SMARTAPI_MAX_DAYS_PER_REQUEST


def _parse_candles(raw: list) -> pd.DataFrame:
    """Convert raw API candle list to a clean DataFrame."""
    if not raw:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(raw, columns=["datetime", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def fetch_candles(
    symbol_token: str,
    exchange: str,
    interval: str,
    from_date,       # date or datetime
    to_date,         # date or datetime
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV candles with automatic pagination.

    Args:
        symbol_token: SmartAPI symboltoken string
        exchange:     "NSE" or "BSE"
        interval:     shorthand key from INTERVAL_MAP (e.g. "1d")
        from_date:    start of range
        to_date:      end of range (inclusive)
        max_retries:  per-chunk retry count
    """
    if interval not in INTERVAL_MAP:
        raise ValueError(f"Unknown interval {interval!r}. Use one of {list(INTERVAL_MAP)}")

    api_interval = INTERVAL_MAP[interval]
    # datetime is a subclass of date — check datetime first to preserve the time component
    from_dt = from_date if isinstance(from_date, datetime) else datetime.combine(from_date, datetime.min.time())
    to_dt   = to_date   if isinstance(to_date,   datetime) else datetime.combine(to_date,   datetime.max.time().replace(microsecond=0))

    chunks = _date_chunks(from_dt, to_dt, api_interval)
    all_dfs = []

    for chunk_from, chunk_to in chunks:
        df = _fetch_chunk(symbol_token, exchange, api_interval, chunk_from, chunk_to, max_retries)
        if not df.empty:
            all_dfs.append(df)
        time.sleep(settings.SMARTAPI_REQUEST_DELAY_SEC)

    if not all_dfs:
        logger.warning("No candle data returned for token={} from {} to {}", symbol_token, from_date, to_date)
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    combined = pd.concat(all_dfs)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    return combined


def _date_chunks(
    from_dt: datetime, to_dt: datetime, api_interval: str
) -> list[tuple[datetime, datetime]]:
    """Split a date range into API-compatible chunks."""
    if api_interval == "ONE_DAY":
        max_delta = timedelta(days=MAX_DAYS_DAILY)
    else:
        max_delta = timedelta(days=30)  # conservative for intraday

    chunks = []
    current = from_dt
    while current < to_dt:
        chunk_end = min(current + max_delta, to_dt)
        chunks.append((current, chunk_end))
        current = chunk_end + timedelta(days=1)
    return chunks


def _fetch_chunk(
    symbol_token: str,
    exchange: str,
    api_interval: str,
    from_dt: datetime,
    to_dt: datetime,
    max_retries: int,
) -> pd.DataFrame:
    """Fetch a single chunk with retry."""
    params = {
        "exchange": exchange,
        "symboltoken": symbol_token,
        "interval": api_interval,
        "fromdate": from_dt.strftime("%Y-%m-%d %H:%M"),
        "todate": to_dt.strftime("%Y-%m-%d %H:%M"),
    }

    for attempt in range(1, max_retries + 1):
        try:
            client = get_client()
            resp = client.getCandleData(params)
            if resp and resp.get("status"):
                return _parse_candles(resp.get("data") or [])
            logger.warning("Empty/failed response (attempt {}): {}", attempt, resp)
        except Exception as exc:
            logger.warning("Candle fetch error (attempt {}): {}", attempt, exc)
            if attempt < max_retries:
                # Rate limit needs a longer cooldown than a normal transient error
                is_rate_limit = "rate" in str(exc).lower() or "429" in str(exc)
                time.sleep(15 if is_rate_limit else 2 ** attempt)

    logger.error("Failed to fetch chunk after {} retries: token={}, {}-{}",
                 max_retries, symbol_token, from_dt.date(), to_dt.date())
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


def fetch_last_n_years(symbol_token: str, exchange: str = "NSE",
                       interval: str = "1d", years: int = 7) -> pd.DataFrame:
    """Convenience wrapper: fetch N years of daily data ending today."""
    to_date = date.today()
    from_date = date(to_date.year - years, to_date.month, to_date.day)
    return fetch_candles(symbol_token, exchange, interval, from_date, to_date)


if __name__ == "__main__":
    from data.instrument_master import symbol_to_token
    tok = symbol_to_token("RELIANCE-EQ")
    logger.info("Fetching 2 years of RELIANCE daily data...")
    df = fetch_last_n_years(tok, years=2)
    logger.info("Got {} rows\n{}", len(df), df.tail())
