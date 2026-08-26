"""
Nifty Futures SHORT — bear-day intraday script.

What it does:
  1. Checks today's regime (Nifty vs EMA20, from DuckDB / yfinance fallback)
  2. On bear days (Nifty < EMA20): SHORT 1 Nifty near-month futures lot at market open
  3. Places SL-M BUY stop at entry + STOP_PCT above entry (upside risk)
  4. Angel One MIS auto-squares off the position by 3:15 PM
  5. Logs the trade to DuckDB fno_intraday_trades table
  6. Sends Telegram notification

On bull days: exits immediately — ORB LONG (intraday_run.py) handles those.

Backtested result (daily-bar proxy, 2015-2026):
  Net P&L: +₹2,44,178 on ₹5L capital | CAGR: +3.5% | Win rate: 50%
  Best year: 2023 (+₹1,00,846) | Worst year: 2025 (-₹66,242)

Run:
  python scripts/fno_nifty_short.py               # paper mode (default)
  python scripts/fno_nifty_short.py --live         # LIVE — real orders
  python scripts/fno_nifty_short.py --test         # test all helpers, no orders

Cron (9:31 AM IST = 04:01 UTC, Mon-Fri):
  1 4 * * 1-5 /path/.venv/bin/python3 /path/scripts/fno_nifty_short.py
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from data.fno_instruments import get_index_future, FutureInfo
from data.holiday_calendar import is_trading_day
from execution.angel_orders import AngelOrderExecutor, EXCHANGE_NFO
from reporting.notifier import Notifier
from storage import store

IST = timezone(timedelta(hours=5, minutes=30))

NIFTY_LOT_SIZE = 50          # Nifty 50 lot = 50 units
LOTS           = 1           # trade 1 lot (start conservative)
STOP_PCT       = 0.005       # SL at 0.5% above entry (SHORT stop = BUY above)

_LOG_FILE = settings.LOG_DIR / "fno_nifty_short" / "{time:YYYY-MM-DD}.log"
logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")
logger.add(str(_LOG_FILE), level="DEBUG", rotation="1 month", retention="6 months",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}")


# ── Regime check (same logic as intraday_run.py) ──────────────────────────────

def _is_bear_day() -> bool:
    """Return True when Nifty < 20d EMA (bear regime → SHORT signal)."""
    try:
        with store.db_conn() as conn:
            rows = conn.execute(
                "SELECT dt, close FROM adjusted_ohlcv WHERE symbol = 'Nifty 50' "
                "ORDER BY dt DESC LIMIT 40"
            ).fetchall()
        if rows:
            latest = rows[0][0]
            if (date.today() - latest).days <= 3:
                closes = pd.Series({pd.Timestamp(r[0]): float(r[1]) for r in rows}).sort_index()
                ema20  = closes.ewm(span=20, adjust=False).mean()
                bear   = closes.iloc[-1] < ema20.iloc[-1]
                logger.info("Regime (DB): Nifty {:.0f} vs EMA20 {:.0f} → {}",
                            closes.iloc[-1], ema20.iloc[-1], "BEAR" if bear else "BULL")
                return bool(bear)
    except Exception as exc:
        logger.warning("Regime DB query failed: {}", exc)

    try:
        import yfinance as yf
        nifty  = yf.download("^NSEI", period="45d", interval="1d", progress=False, auto_adjust=True)
        closes = nifty["Close"].squeeze().dropna()
        ema20  = closes.ewm(span=20, adjust=False).mean()
        bear   = float(closes.iloc[-1]) < float(ema20.iloc[-1])
        logger.info("Regime (yf): Nifty {:.0f} vs EMA20 {:.0f} → {}",
                    float(closes.iloc[-1]), float(ema20.iloc[-1]), "BEAR" if bear else "BULL")
        return bool(bear)
    except Exception as exc:
        logger.warning("Regime yfinance failed ({}), defaulting to skip SHORT", exc)
        return False   # safe default: no trade if unsure


# ── Fetch Nifty near-month futures contract ────────────────────────────────────

def _get_nifty_future(today: date) -> FutureInfo | None:
    try:
        info = get_index_future("NIFTY", today)
        if info:
            logger.info("Nifty future: {} | lot={} | expiry={}",
                        info.trading_symbol, info.lot_size, info.expiry)
        else:
            logger.error("No Nifty FUTIDX contract found in instrument master")
        return info
    except Exception as exc:
        logger.error("Could not load Nifty future info: {}", exc)
        return None


# ── DuckDB trade logging ───────────────────────────────────────────────────────

def _ensure_table(conn) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fno_intraday_trades (
            trade_id        VARCHAR PRIMARY KEY,
            trade_date      DATE,
            symbol          VARCHAR,
            direction       VARCHAR,
            lots            INTEGER,
            qty             INTEGER,
            entry_order_id  VARCHAR,
            sl_order_id     VARCHAR,
            entry_price     DOUBLE,
            sl_trigger      DOUBLE,
            paper           BOOLEAN,
            logged_at       TIMESTAMP
        )
    """)


def _log_trade(trade_id: str, today: date, info: FutureInfo, lots: int,
               entry_id: str, sl_id: str, sl_trigger: float, paper: bool) -> None:
    try:
        with store.db_conn() as conn:
            _ensure_table(conn)
            conn.execute("""
                INSERT INTO fno_intraday_trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT (trade_id) DO NOTHING
            """, [
                trade_id, today, info.trading_symbol, "SHORT",
                lots, lots * info.lot_size,
                entry_id, sl_id,
                None,          # entry_price filled in post-trade update
                sl_trigger,
                paper,
                datetime.now(IST).replace(tzinfo=None),
            ])
        logger.info("Trade logged: {}", trade_id)
    except Exception as exc:
        logger.warning("Could not log trade to DB: {}", exc)


# ── Current Nifty price for SL calculation ────────────────────────────────────

def _get_current_nifty_price() -> float:
    """
    Fetch today's Nifty spot price for SL placement.
    For SHORT: SL BUY stop must be ABOVE entry, so we need current (not stale DB) price.
    Tries Angel One LTP first, falls back to yfinance 1-day 1-min bar last close.
    """
    try:
        import yfinance as yf
        tick = yf.download("^NSEI", period="1d", interval="1m", progress=False, auto_adjust=True)
        if not tick.empty:
            price = float(tick["Close"].squeeze().dropna().iloc[-1])
            logger.info("Current Nifty (yfinance 1m): {:.0f}", price)
            return price
    except Exception as exc:
        logger.warning("yfinance 1m Nifty fetch failed: {}", exc)

    # Fallback: last close from DB (may be 1 day stale, but better than nothing)
    try:
        with store.db_conn() as conn:
            row = conn.execute(
                "SELECT close FROM adjusted_ohlcv WHERE symbol='Nifty 50' ORDER BY dt DESC LIMIT 1"
            ).fetchone()
        if row:
            logger.warning("Using stale DB Nifty close as entry estimate: {:.0f}", float(row[0]))
            return float(row[0])
    except Exception:
        pass

    logger.warning("Could not determine current Nifty price — using 24000 as fallback estimate")
    return 24000.0


# ── Telegram notification ──────────────────────────────────────────────────────

def _send_telegram(today: date, info: FutureInfo, lots: int, sl_trigger: float,
                   paper: bool, no_notify: bool) -> None:
    if no_notify:
        return
    mode  = "PAPER" if paper else "LIVE"
    qty   = lots * info.lot_size
    lines = [
        f"*Piedpiper F&O SHORT ({mode}) — {today}*\n",
        f"*SHORT Nifty Futures* ↓",
        f"  Contract: {info.trading_symbol}",
        f"  Qty: {qty} units ({lots} lot)  |  Expiry: {info.expiry}",
        f"  SL-M BUY trigger: {sl_trigger:.2f} (+0.5% above entry)",
        f"  Exit: Auto SQO at 3:15 PM IST (MIS product)",
        f"\n_Bear regime today (Nifty < EMA20). Check at 3:15 PM for exit price._",
    ]
    ok = Notifier().send_telegram("\n".join(lines))
    logger.info("Telegram: {}", "sent" if ok else "not sent (check credentials)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Nifty Futures SHORT on bear days")
    ap.add_argument("--live",      action="store_true", help="Place real orders (default: paper)")
    ap.add_argument("--no-notify", action="store_true", help="Suppress Telegram")
    ap.add_argument("--test",      action="store_true", help="Run regime+instrument checks only")
    args = ap.parse_args()

    paper = not args.live
    today = date.today()

    logger.info("=" * 55)
    logger.info("Nifty Futures SHORT — {} | {}", today, "LIVE" if not paper else "PAPER")
    logger.info("=" * 55)

    if not is_trading_day(today):
        logger.info("{} is not a trading day — exiting", today)
        return

    # Only trade on bear days
    if not _is_bear_day():
        msg = f"*Nifty Futures SHORT {today}*\nBull regime today — no SHORT trade. ORB LONG runs via intraday_run.py."
        logger.info("Bull regime — skipping SHORT")
        if not args.no_notify:
            Notifier().send_telegram(msg)
        return

    # Get Nifty futures contract
    store.init_schema()
    info = _get_nifty_future(today)
    if info is None:
        logger.error("Cannot trade — no Nifty futures contract found")
        sys.exit(1)

    qty = LOTS * info.lot_size
    logger.info("Placing SHORT: {} × {} units = {} total", LOTS, info.lot_size, qty)

    if args.test:
        logger.info("TEST mode — instrument found, regime is BEAR. No orders placed.")
        logger.info("Would SHORT {} qty={}", info.trading_symbol, qty)
        return

    executor = AngelOrderExecutor(paper=paper)
    import uuid
    trade_id = str(uuid.uuid4())[:12]

    # Entry: SELL (SHORT)
    entry_id = executor.place_fno_market_order(
        trading_symbol=info.trading_symbol,
        token=info.token,
        qty=qty,
        side="SELL",
    )

    # Estimate entry price for SL using live Nifty quote (yfinance fallback)
    # For SHORT: SL is a BUY stop ABOVE entry. Must use current price, not stale DB.
    est_entry = _get_current_nifty_price()
    sl_trigger = round(est_entry * (1 + STOP_PCT), 1)   # SHORT SL = entry + 0.5%

    # SL-M: BUY stop above entry (caps loss if Nifty spikes up)
    sl_id = executor.place_sl_market_order(
        symbol=info.trading_symbol,
        token=info.token,
        qty=qty,
        side="BUY",
        trigger_price=sl_trigger,
        exchange=EXCHANGE_NFO,
    )

    logger.info("Entry order: {}  |  SL order: {}  |  SL trigger: {:.1f}", entry_id, sl_id, sl_trigger)

    _log_trade(trade_id, today, info, LOTS, entry_id, sl_id, sl_trigger, paper)
    _send_telegram(today, info, LOTS, sl_trigger, paper, args.no_notify)

    logger.info("Nifty SHORT placed. Position will auto square-off at 3:15 PM IST (MIS).")


if __name__ == "__main__":
    main()
