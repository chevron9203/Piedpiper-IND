"""
Intraday position monitor — runs every 15 min from 9:45 AM to 3:00 PM.

What it does:
  1. Fetches LTP for all open paper/live positions
  2. Auto-exits at target (places market sell in live, logs in paper)
  3. Moves stop to breakeven once trade is 50% to target (trailing stop)
  4. Shows live P&L — visible on dashboard /intraday tab
  5. Supports manual early exit via --close SYMBOL flag

Cron (every 15 min, 9:45–3:00 IST Mon-Fri):
  */15 9-14 * * 1-5 ... intraday_monitor.py
  (see setup_cron.sh — cron block added automatically)

Manual early exit:
  python scripts/intraday_monitor.py --close RELIANCE
  python scripts/intraday_monitor.py --close ALL
"""
from __future__ import annotations

import argparse
import sys
import uuid
from datetime import date, datetime, time, timezone, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.holiday_calendar import is_trading_day
from execution.angel_orders import AngelOrderExecutor
from reporting.notifier import Notifier
from storage import store

IST = timezone(timedelta(hours=5, minutes=30))


def _now_ist() -> datetime:
    return datetime.now(IST).replace(tzinfo=None)


def _get_ltp(symbol: str, token: str, exchange: str = "NSE") -> float | None:
    """Fetch LTP via Angel One API. Use exchange='NFO' for futures."""
    try:
        from data.auth import get_client
        client = get_client()
        resp = client.ltpData(exchange, symbol, token)
        if resp and resp.get("status"):
            return float(resp["data"]["ltp"])
    except Exception as exc:
        logger.debug("LTP fetch failed for {}: {}", symbol, exc)
    return None


def _intraday_charges(buy_val: float, sell_val: float) -> float:
    brok = 80.0
    stt  = sell_val * 0.00025
    exch = (buy_val + sell_val) * 0.0000335
    sebi = (buy_val + sell_val) * 0.000001
    stmp = buy_val * 0.00003
    gst  = (brok + exch + sebi) * 0.18
    return brok + stt + exch + sebi + stmp + gst


def _close_position(row: dict, exit_price: float, exit_reason: str,
                    executor: AngelOrderExecutor, token_map: dict) -> dict:
    """Close one position: place order (live) or simulate (paper), update DuckDB."""
    sym       = row["symbol"]
    qty       = int(row["qty"] or 0)
    entry_px  = float(row["entry_price"] or 0)
    direction = row.get("direction", "LONG")
    is_short  = direction == "SHORT"

    if not executor.paper:
        tok = token_map.get(sym + "-EQ") or token_map.get(sym, "")
        try:
            # LONG: SELL to close; SHORT: BUY to close (equity MIS short-sell)
            side = "BUY" if is_short else "SELL"
            executor.place_market_order(sym + "-EQ", tok, qty, side)
        except Exception as exc:
            logger.error("Close order failed for {} ({}): {}", sym, side, exc)

    # P&L: LONG = (exit - entry), SHORT = (entry - exit)
    gross_pnl = (entry_px - exit_price) * qty if is_short else (exit_price - entry_px) * qty
    # Charges: buy_val / sell_val depend on direction
    # LONG:  bought at entry, sold at exit  → buy=entry, sell=exit
    # SHORT: sold at entry, bought at exit  → buy=exit,  sell=entry
    if is_short:
        charges = _intraday_charges(exit_price * qty, entry_px * qty)
    else:
        charges = _intraday_charges(entry_px * qty, exit_price * qty)
    net_pnl = gross_pnl - charges

    updated = dict(row)
    updated.update({
        "exit_time":   _now_ist(),
        "exit_price":  round(exit_price, 2),
        "exit_reason": exit_reason,
        "gross_pnl":   round(gross_pnl, 2),
        "charges":     round(charges, 2),
        "net_pnl":     round(net_pnl, 2),
    })
    store.log_intraday_trade(updated)
    logger.info("{} closed — {} @ {:.2f} | net P&L ₹{:+,.0f}",
                sym, exit_reason, exit_price, net_pnl)
    return updated


def _update_trailing_stop(row: dict) -> None:
    """
    2-tier trailing stop — works for both LONG and SHORT.
      ≥50% to target → stop moves to entry (breakeven)
      ≥75% to target → stop locks half the gain
    """
    entry     = float(row["entry_price"] or 0)
    stop      = float(row["stop_loss"]   or 0)
    target    = float(row["target"]      or 0)
    ltp       = row.get("_ltp")
    is_short  = row.get("direction", "LONG") == "SHORT"

    if ltp is None or entry <= 0:
        return

    if is_short:
        if target >= entry:   # target must be below entry for shorts
            return
        gain_range = entry - target        # positive: how far price needs to fall
        progress   = (entry - ltp) / gain_range if gain_range > 0 else 0
        tier2_stop = entry - 0.5 * gain_range   # half the gain locked (below entry)
        if progress >= 0.75 and stop > tier2_stop:
            updated = dict(row)
            updated["stop_loss"] = round(tier2_stop, 2)
            store.log_intraday_trade(updated)
            logger.info("{} SHORT trailing stop → {:.2f} (locked 50% gain)", row["symbol"], tier2_stop)
        elif progress >= 0.50 and stop > entry:
            updated = dict(row)
            updated["stop_loss"] = round(entry, 2)
            store.log_intraday_trade(updated)
            logger.info("{} SHORT stop → breakeven @ {:.2f}", row["symbol"], entry)
    else:
        if target <= entry:
            return
        gain_range = target - entry
        progress   = (ltp - entry) / gain_range if gain_range > 0 else 0
        tier2_stop = entry + 0.5 * gain_range
        if progress >= 0.75 and stop < tier2_stop:
            updated = dict(row)
            updated["stop_loss"] = round(tier2_stop, 2)
            store.log_intraday_trade(updated)
            logger.info("{} trailing stop → {:.2f} (locked 50% gain, {:.0f}% to target)",
                        row["symbol"], tier2_stop, progress * 100)
        elif progress >= 0.50 and stop < entry:
            updated = dict(row)
            updated["stop_loss"] = round(entry, 2)
            store.log_intraday_trade(updated)
            logger.info("{} stop → breakeven @ {:.2f} ({:.0f}% to target)",
                        row["symbol"], entry, progress * 100)


def main() -> None:
    ap = argparse.ArgumentParser(description="Intraday position monitor")
    ap.add_argument("--live",  action="store_true", help="Live mode — real orders")
    ap.add_argument("--close", metavar="SYMBOL",
                    help="Manually close a symbol early (use ALL to close everything)")
    ap.add_argument("--no-notify", action="store_true")
    args = ap.parse_args()

    today = date.today()
    paper = not args.live

    if not is_trading_day(today):
        return

    now = _now_ist().time()
    if now < time(9, 31) or now > time(15, 15):
        logger.info("Outside trading hours ({}) — nothing to monitor", now)
        return

    store.init_schema()
    executor = AngelOrderExecutor(paper=paper)

    # Load token map for live exits
    token_map: dict = {}
    try:
        from data.instrument_master import get_nse_equity_master
        master = get_nse_equity_master()
        token_map = dict(zip(master["symbol"], master["token"].astype(str)))
    except Exception:
        pass

    # Load today's open positions
    trades_df = store.load_intraday_trades(from_date=today, to_date=today, paper_only=paper)
    open_trades = trades_df[trades_df["exit_price"].isna()] if not trades_df.empty else pd.DataFrame()

    if open_trades.empty:
        logger.info("No open positions to monitor")
        return

    logger.info("Monitoring {} open position(s)", len(open_trades))

    closed_now: list[dict] = []
    still_open: list[dict] = []

    for _, row in open_trades.iterrows():
        row      = row.to_dict()
        sym      = row["symbol"]
        entry    = float(row["entry_price"] or 0)
        stop     = float(row["stop_loss"]   or 0)
        target   = float(row["target"]      or 0)
        qty      = int(row["qty"] or 0)
        is_short = row.get("direction", "LONG") == "SHORT"

        # Fetch LTP — all positions are equity (NSE), SHORT is equity MIS short-sell
        eq_tok = token_map.get(sym + "-EQ") or token_map.get(sym, "")
        ltp    = _get_ltp(sym + "-EQ", eq_tok, exchange="NSE")
        row["_ltp"] = ltp

        if ltp is None:
            logger.warning("Could not get LTP for {} — skipping", sym)
            still_open.append(row)
            continue

        # Unrealised P&L direction-aware
        unreal_pnl = (entry - ltp) * qty if is_short else (ltp - entry) * qty
        logger.info("{} {} LTP={:.2f} entry={:.2f} stop={:.2f} target={:.2f} unreal ₹{:+,.0f}",
                    "SHORT" if is_short else "LONG", sym, ltp, entry, stop, target, unreal_pnl)

        # Manual early close
        if args.close and (args.close.upper() == "ALL" or args.close.upper() == sym.upper()):
            closed_now.append(_close_position(row, ltp, "manual", executor, token_map))
            continue

        # Exit checks — direction-aware
        if is_short:
            if ltp <= target:   # price fell to target (profit for short)
                closed_now.append(_close_position(row, target, "target", executor, token_map))
                continue
            if ltp >= stop:     # price rose to stop-loss (loss for short)
                closed_now.append(_close_position(row, stop, "stop", executor, token_map))
                continue
        else:
            if ltp >= target:
                closed_now.append(_close_position(row, target, "target", executor, token_map))
                continue
            if ltp <= stop:
                closed_now.append(_close_position(row, stop, "stop", executor, token_map))
                continue

        # 2-tier trailing stop update
        _update_trailing_stop(row)
        still_open.append(row)

    # Send status update
    if not args.no_notify and (closed_now or still_open):
        lines = [f"*Intraday Monitor — {_now_ist().strftime('%H:%M IST')}*\n"]
        for t in closed_now:
            pnl = t.get("net_pnl", 0) or 0
            lines.append(f"{'✓' if pnl>0 else '✗'} {t['symbol']} CLOSED {t['exit_reason'].upper()} "
                         f"@ ₹{t['exit_price']:.2f} | P&L ₹{pnl:+,.0f}")
        for r in still_open:
            ltp = r.get("_ltp")
            if ltp:
                ep       = float(r["entry_price"])
                r_short  = r.get("direction", "LONG") == "SHORT"
                unreal   = (ep - ltp) * int(r["qty"]) if r_short else (ltp - ep) * int(r["qty"])
                pct      = (ltp - ep) / ep * 100 * (-1 if r_short else 1)
                lines.append(f"↔ {r['symbol']} {'SHORT' if r_short else 'LONG'} LTP ₹{ltp:.2f} ({pct:+.1f}%) "
                             f"unreal ₹{unreal:+,.0f}")
        Notifier().send_telegram("\n".join(lines))

    total_pnl = sum(t.get("net_pnl", 0) or 0 for t in closed_now)
    logger.info("Monitor done. Closed: {} | Still open: {} | Realised P&L: ₹{:+,.0f}",
                len(closed_now), len(still_open), total_pnl)


if __name__ == "__main__":
    main()
