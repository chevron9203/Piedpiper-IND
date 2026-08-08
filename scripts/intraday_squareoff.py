"""
Intraday square-off — run at 3:10 PM IST every trading day.

Closes all open intraday positions BEFORE Angel One's auto-SQO at 3:15 PM.
Running this explicitly gives better control over exit prices vs auto-SQO
which tends to get worse fills due to concentrated order flow.

In paper mode: updates DuckDB trade records with simulated exits.
In live mode: calls Angel One square_off_all() to close real positions.

Cron (3:10 PM IST = 09:40 UTC, Mon-Fri):
  40 9 * * 1-5 /path/to/.venv/bin/python /path/to/scripts/intraday_squareoff.py

Or run manually if you want to close early:
  python scripts/intraday_squareoff.py --live
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.holiday_calendar import is_trading_day
from execution.angel_orders import AngelOrderExecutor
from reporting.notifier import Notifier
from storage import store

IST = timezone(timedelta(hours=5, minutes=30))


def _update_paper_exits(today: date, executor: AngelOrderExecutor) -> list[dict]:
    """
    For paper mode: mark today's open trades as closed using the last available price.
    Simulates squaring off at 3:10 PM.
    """
    trades_df = store.load_intraday_trades(from_date=today, to_date=today, paper_only=True)
    trades_df = trades_df[trades_df["exit_price"].isna()]  # only open trades

    now_ist = datetime.now(IST).replace(tzinfo=None)
    closed = []
    for _, row in trades_df.iterrows():
        sym       = row["symbol"]
        qty       = int(row["qty"] or 0)
        entry_px  = float(row["entry_price"] or 0)
        direction = row.get("direction", "LONG")
        is_short  = direction == "SHORT"

        # In paper mode we don't have live prices; use entry_price as placeholder
        exit_price = entry_px

        # LONG: (exit - entry)*qty; SHORT: (entry - exit)*qty
        gross_pnl = ((entry_px - exit_price) if is_short else (exit_price - entry_px)) * qty
        # STT 0.025% on sell side: LONG sells at exit, SHORT sells at entry
        stt_base  = (entry_px if is_short else exit_price) * qty
        charges   = 80.0 + stt_base * 0.00025
        net_pnl   = gross_pnl - charges

        closed_trade = dict(row)
        closed_trade.update({
            "exit_time":   now_ist,
            "exit_price":  exit_price,
            "exit_reason": "squareoff",
            "gross_pnl":   round(gross_pnl, 2),
            "charges":     round(charges, 2),
            "net_pnl":     round(net_pnl, 2),
        })
        store.log_intraday_trade(closed_trade)
        closed.append(closed_trade)

    return closed


def _update_live_exits(today: date, executor: AngelOrderExecutor) -> list[dict]:
    """
    After live square_off_all(), fetch today's open DB trades, get LTP, and update exits.
    Uses LTP as best approximation of market-order fill price.
    """
    trades_df = store.load_intraday_trades(from_date=today, to_date=today)
    trades_df = trades_df[trades_df["exit_price"].isna()]

    if trades_df.empty:
        return []

    now_ist = datetime.now(IST).replace(tzinfo=None)
    closed = []
    for _, row in trades_df.iterrows():
        sym       = row["symbol"]
        qty       = int(row["qty"] or 0)
        entry_px  = float(row["entry_price"] or 0)
        direction = row.get("direction", "LONG")
        is_short  = direction == "SHORT"
        tok       = str(row.get("token") or "")

        ltp = executor.get_ltp("NSE", sym, tok) if tok else None
        exit_price = ltp if ltp else entry_px

        gross_pnl = ((entry_px - exit_price) if is_short else (exit_price - entry_px)) * qty
        stt_base  = (entry_px if is_short else exit_price) * qty
        charges   = 80.0 + stt_base * 0.00025
        net_pnl   = gross_pnl - charges

        closed_trade = dict(row)
        closed_trade.update({
            "exit_time":   now_ist,
            "exit_price":  exit_price,
            "exit_reason": "squareoff",
            "gross_pnl":   round(gross_pnl, 2),
            "charges":     round(charges, 2),
            "net_pnl":     round(net_pnl, 2),
        })
        store.log_intraday_trade(closed_trade)
        closed.append(closed_trade)

    return closed


def _format_eod_msg(today: date, closed_trades: list[dict], paper: bool) -> str:
    mode = "PAPER" if paper else "LIVE"
    lines = [f"*Intraday EOD ({mode}) — {today}*\n"]

    total_pnl  = sum(t.get("net_pnl") or 0 for t in closed_trades)
    total_trades = len(closed_trades)
    winners    = sum(1 for t in closed_trades if (t.get("net_pnl") or 0) > 0)

    lines.append(f"Trades: {total_trades}  |  Winners: {winners}/{total_trades}  |  Net P&L: ₹{total_pnl:+,.0f}")
    lines.append("")

    for t in closed_trades:
        reason = t.get("exit_reason", "?")
        pnl    = t.get("net_pnl") or 0
        emoji  = "✓" if pnl > 0 else "✗"
        lines.append(
            f"{emoji} {t.get('symbol')} | "
            f"Entry {t.get('entry_price',0):.2f} → Exit {t.get('exit_price',0):.2f} "
            f"({reason}) | P&L ₹{pnl:+,.0f}"
        )

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Intraday square-off at 3:10 PM")
    ap.add_argument("--live",      action="store_true",
                    help="LIVE mode — actually closes Angel One positions")
    ap.add_argument("--no-notify", action="store_true")
    args = ap.parse_args()

    store.init_schema()

    today = date.today()
    paper = not args.live

    if not is_trading_day(today):
        logger.info("{} is not a trading day — nothing to square off", today)
        return

    logger.info("Intraday square-off — {} | mode={}", today, "LIVE" if not paper else "PAPER")

    executor = AngelOrderExecutor(paper=paper)

    if paper:
        closed_trades = _update_paper_exits(today, executor)
        logger.info("Paper square-off: {} trades closed", len(closed_trades))
    else:
        n = executor.square_off_all()
        logger.info("Live square-off: {} positions closed", n)
        closed_trades = _update_live_exits(today, executor)

    if not args.no_notify:
        msg = _format_eod_msg(today, closed_trades, paper)
        Notifier().send_telegram(msg)

    logger.info("Square-off complete.")


if __name__ == "__main__":
    main()
