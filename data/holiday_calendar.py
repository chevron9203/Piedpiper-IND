"""
NSE trading holiday calendar.

Fetches from NSE's official API, caches for the year, exposes
is_trading_day() and next_trading_day(). Refreshes automatically
mid-year (NSE has added holidays after publishing the annual list).
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
import requests
from loguru import logger
from config import settings

CACHE_PATH = settings.DATA_DIR / "nse_holidays.json"

# NSE's public holiday API (JSON response)
NSE_HOLIDAY_URL = "https://www.nseindia.com/api/holiday-master?type=trading"

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; piedpiper/1.0)",
    "Accept": "application/json",
    "Referer": "https://www.nseindia.com/",
}


def _fetch_from_nse() -> list[str]:
    """Fetch holiday list from NSE API. Returns list of ISO date strings."""
    session = requests.Session()
    # NSE requires a cookie from the main page before API calls work
    session.get("https://www.nseindia.com", headers=NSE_HEADERS, timeout=15)
    resp = session.get(NSE_HOLIDAY_URL, headers=NSE_HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    holidays = []
    # Response is {"CM": [...], "FO": [...], ...} — use CM (cash market) segment
    for segment_key in ("CM", "equities"):
        if segment_key in data:
            for entry in data[segment_key]:
                # Entry format: {"tradingDate": "01-Jan-2026", ...}
                dt_str = entry.get("tradingDate", "")
                try:
                    dt = _parse_nse_date(dt_str)
                    holidays.append(dt.isoformat())
                except ValueError:
                    logger.warning("Could not parse NSE holiday date: {}", dt_str)
            break
    return sorted(set(holidays))


def _parse_nse_date(dt_str: str) -> date:
    """Parse NSE date formats: '01-Jan-2026' or '2026-01-01'."""
    from datetime import datetime
    for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(dt_str, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {dt_str!r}")


def _load_cache() -> list[str]:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return []


def _save_cache(holidays: list[str]) -> None:
    CACHE_PATH.write_text(json.dumps(holidays, indent=2))


def refresh_holidays(force: bool = False) -> list[str]:
    """
    Load holidays from cache, refreshing from NSE if stale.
    NSE can add holidays mid-year, so refresh monthly.
    """
    cached = _load_cache()
    current_year = str(date.today().year)

    needs_refresh = (
        force
        or not cached
        or not any(h.startswith(current_year) for h in cached)
    )

    if needs_refresh:
        logger.info("Fetching NSE holiday calendar...")
        try:
            holidays = _fetch_from_nse()
            _save_cache(holidays)
            logger.info("Cached {} NSE holidays", len(holidays))
            return holidays
        except Exception as exc:
            logger.warning("Could not fetch NSE holidays ({}). Using cache/fallback.", exc)
            if cached:
                return cached
            # Hard fallback: known 2026 holidays — update annually
            return _fallback_2026_holidays()

    return cached


def get_holidays() -> set[date]:
    """Return set of holiday dates for the current and next year."""
    raw = refresh_holidays()
    return {date.fromisoformat(d) for d in raw}


def is_trading_day(d: date | None = None) -> bool:
    """Return True if d is an NSE trading day (not weekend, not holiday)."""
    if d is None:
        d = date.today()
    if d.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    return d not in get_holidays()


def next_trading_day(d: date | None = None) -> date:
    """Return the next trading day after d (exclusive)."""
    if d is None:
        d = date.today()
    candidate = d + timedelta(days=1)
    while not is_trading_day(candidate):
        candidate += timedelta(days=1)
    return candidate


def prev_trading_day(d: date | None = None) -> date:
    """Return the most recent trading day on or before d."""
    if d is None:
        d = date.today()
    candidate = d
    while not is_trading_day(candidate):
        candidate -= timedelta(days=1)
    return candidate


def trading_days_between(start: date, end: date) -> list[date]:
    """Return all trading days in [start, end] inclusive."""
    days = []
    current = start
    holidays = get_holidays()
    while current <= end:
        if current.weekday() < 5 and current not in holidays:
            days.append(current)
        current += timedelta(days=1)
    return days


def _fallback_2026_holidays() -> list[str]:
    """Known NSE holidays for 2026 — hard fallback if NSE API is unreachable."""
    return [
        "2026-01-26",  # Republic Day
        "2026-03-25",  # Holi
        "2026-04-02",  # Ram Navami (check)
        "2026-04-14",  # Dr. Ambedkar Jayanti / Baisakhi
        "2026-05-01",  # Maharashtra Day
        "2026-08-15",  # Independence Day
        "2026-10-02",  # Gandhi Jayanti
        "2026-11-04",  # Diwali Laxmi Pujan (check)
        "2026-11-05",  # Diwali Balipratipada (check)
        "2026-12-25",  # Christmas
    ]


if __name__ == "__main__":
    holidays = get_holidays()
    logger.info("Loaded {} holidays", len(holidays))
    today = date.today()
    logger.info("Today ({}) is trading day: {}", today, is_trading_day(today))
    logger.info("Next trading day: {}", next_trading_day(today))
