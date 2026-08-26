"""
Momentum Daily Guard — runs at 4:05 PM IST every trading day.

Checks two things ONLY:
  1. Regime guard: Nifty 50 < 200d EMA → exit ALL holdings next morning → LIQUIDBEES
  2. Hard floor: any position down -15% from entry → exit that position next morning

Everything else (ranking, portfolio selection, entry) stays with monthly_run.py.
Trailing stops and breakout entries were backtested and hurt performance — not included.

Cron line (4:05 PM IST, Mon–Fri):
  Mac:      5 16 * * 1-5 cd /path/to/piedpiper && python scripts/momentum_daily.py
  Cloud VM: 35 10 * * 1-5 cd /path/to/piedpiper && python scripts/momentum_daily.py

Test:
  python scripts/momentum_daily.py --paper --no-notify
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from reporting.notifier import Notifier
from storage import store

# ── Constants ────────────────────────────────────────────────────────────────
NIFTY_EMA_SPAN  = 200     # 200-day EMA for regime detection
HARD_FLOOR_PCT  = 0.15    # exit if position is down 15% from entry
VIX_EXIT        = 35.0    # India VIX above this → forced exit all

_LOG_FILE = settings.LOG_DIR / "momentum_daily" / "{time:YYYY-MM-DD}.log"
logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")
logger.add(str(_LOG_FILE), level="DEBUG", rotation="1 month", retention="6 months",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}")


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_nifty_history(lookback_days: int = 320) -> pd.Series:
    """Load Nifty 50 close history — enough for 200d EMA (needs 280+ calendar days)."""
    from_dt = date.today() - timedelta(days=lookback_days)
    with store.db_conn() as conn:
        df = conn.execute("""
            SELECT dt, close FROM adjusted_ohlcv
            WHERE symbol = ? AND dt >= ?
            ORDER BY dt
        """, [settings.NIFTY50_SYMBOL, from_dt]).df()
    if df.empty:
        return pd.Series(dtype=float)
    df["dt"] = pd.to_datetime(df["dt"])
    return df.set_index("dt")["close"].sort_index()


def _load_vix() -> float:
    """Load latest India VIX close. Returns nan if unavailable."""
    try:
        with store.db_conn() as conn:
            row = conn.execute("""
                SELECT close FROM adjusted_ohlcv
                WHERE symbol = ? ORDER BY dt DESC LIMIT 1
            """, [settings.INDIA_VIX_SYMBOL]).fetchone()
        return float(row[0]) if row else float("nan")
    except Exception:
        return float("nan")


def _load_current_holdings() -> list[dict]:
    """
    Load active momentum positions from momentum_entries.
    Returns the entries for the most-recent in-market signal_date.
    """
    try:
        with store.db_conn() as conn:
            latest = conn.execute("""
                SELECT signal_date FROM monthly_signals
                WHERE in_market = TRUE ORDER BY signal_date DESC LIMIT 1
            """).fetchone()
            if not latest:
                return []
            sig_date = latest[0]
            rows = conn.execute("""
                SELECT symbol, entry_price, weight, qty, hard_stop, signal_date
                FROM momentum_entries WHERE signal_date = ?
            """, [sig_date]).fetchall()
        return [
            {"symbol": r[0], "entry_price": r[1], "weight": r[2],
             "qty": r[3], "hard_stop": r[4], "signal_date": r[5]}
            for r in rows
        ]
    except Exception as exc:
        logger.warning("Could not load holdings: {}", exc)
        return []


def _load_current_prices(symbols: list[str]) -> dict[str, float]:
    """Fetch latest close price for each symbol from DuckDB."""
    if not symbols:
        return {}
    try:
        with store.db_conn() as conn:
            df = conn.execute("""
                SELECT symbol, close FROM adjusted_ohlcv
                WHERE symbol = ANY(?) AND source = 'eod2'
                ORDER BY symbol, dt DESC
            """, [symbols]).df()
        if df.empty:
            return {}
        return df.groupby("symbol")["close"].first().to_dict()
    except Exception as exc:
        logger.warning("Could not load current prices: {}", exc)
        return {}


def _load_already_exited_today(today: date) -> set[str]:
    """Return symbols that already have an exit decision logged today."""
    try:
        with store.db_conn() as conn:
            rows = conn.execute("""
                SELECT symbol FROM momentum_daily_decisions
                WHERE decision_date = ? AND action = 'SELL'
            """, [str(today)]).fetchall()
        return {r[0] for r in rows}
    except Exception:
        return set()


# ─────────────────────────────────────────────────────────────────────────────
# Signal checks
# ─────────────────────────────────────────────────────────────────────────────

def _check_regime(nifty_series: pd.Series) -> tuple[bool, float, float]:
    """
    Returns (is_bull, nifty_close_today, ema200_today).
    is_bull = True when Nifty > 200d EMA.
    """
    if nifty_series.empty or len(nifty_series) < 50:
        logger.warning("Not enough Nifty history for 200d EMA — assuming bull")
        return True, float("nan"), float("nan")

    ema200 = float(nifty_series.ewm(span=NIFTY_EMA_SPAN, adjust=False).mean().iloc[-1])
    nifty_now = float(nifty_series.iloc[-1])
    return nifty_now > ema200, nifty_now, ema200


# ─────────────────────────────────────────────────────────────────────────────
# DB logging
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_decisions_table() -> None:
    with store.db_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS momentum_daily_decisions (
                decision_date      DATE,
                symbol             VARCHAR,
                action             VARCHAR,
                reason             VARCHAR,
                detail             VARCHAR,
                trigger_price      DOUBLE,
                trail_stop         DOUBLE,
                entry_price        DOUBLE,
                unrealized_pnl_pct DOUBLE,
                paper              BOOLEAN,
                logged_at          TIMESTAMP,
                PRIMARY KEY (decision_date, symbol, action)
            )
        """)


def _log_decision(decision: dict, paper: bool) -> None:
    try:
        _ensure_decisions_table()
        with store.db_conn() as conn:
            conn.execute("""
                INSERT INTO momentum_daily_decisions VALUES (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT (decision_date, symbol, action) DO UPDATE SET
                    reason=excluded.reason, detail=excluded.detail,
                    trigger_price=excluded.trigger_price,
                    unrealized_pnl_pct=excluded.unrealized_pnl_pct,
                    logged_at=excluded.logged_at
            """, [
                decision["date"], decision["symbol"], decision["action"],
                decision["reason"], decision.get("detail", ""),
                decision.get("trigger_price"), None,
                decision.get("entry_price"), decision.get("unrealized_pnl_pct"),
                paper, pd.Timestamp.now(),
            ])
    except Exception as exc:
        logger.warning("Could not log decision: {}", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Telegram formatting
# ─────────────────────────────────────────────────────────────────────────────

def _format_telegram(
    sells: list[dict],
    today: date,
    is_bull: bool,
    nifty_close: float,
    ema200: float,
    vix_val: float,
    paper: bool,
) -> str:
    tag = " [PAPER]" if paper else ""
    lines = [f"*Piedpiper Daily Guard — {today.strftime('%d %b %Y')}*{tag}\n"]

    regime_str = f"{'📈 BULL' if is_bull else '📉 BEAR'}"
    nifty_str  = f"{nifty_close:,.0f}" if not np.isnan(nifty_close) else "N/A"
    ema_str    = f"{ema200:,.0f}"       if not np.isnan(ema200)      else "N/A"
    vix_str    = f"{vix_val:.1f}"       if not np.isnan(vix_val)     else "N/A"

    lines.append(f"Regime: {regime_str}  |  Nifty: {nifty_str}  |  EMA200: {ema_str}  |  VIX: {vix_str}\n")

    if not sells:
        lines.append("_No guard signals today. All positions intact._")
        return "\n".join(lines)

    lines.append(f"*⚠️ EXIT {len(sells)} position(s):*")
    for s in sells:
        pnl_str = f"{s['unrealized_pnl_pct']:+.1f}%" if s.get("unrealized_pnl_pct") is not None else "N/A"
        lines.append(f"  SELL *{s['symbol']}* — _{s['reason']}_")
        lines.append(f"    {s['detail']}")
        lines.append(f"    P&L from entry: {pnl_str}")
        lines.append(f"  → Park proceeds in *LIQUIDBEES* at tomorrow's open")

    lines.append("")
    if not is_bull:
        lines.append("_Regime exit: hold LIQUIDBEES until Nifty > 200d EMA_")
        lines.append("_monthly\\_run.py will re-enter at next month-end_")
    lines.append("_Execute at open, 9:30 AM IST tomorrow_")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_daily_check(paper: bool = True, notify: bool = True) -> list[dict]:
    """
    Run daily guard checks. Returns list of sell decisions.
    """
    today = date.today()
    logger.info("=== Momentum Daily Guard — {} ===", today)

    # ── 1. Regime: Nifty 200d EMA ────────────────────────────────────────────
    nifty_series = _load_nifty_history()
    is_bull, nifty_close, ema200 = _check_regime(nifty_series)

    logger.info("Regime: {} | Nifty {:.0f} | EMA200 {:.0f}",
                "BULL" if is_bull else "BEAR",
                nifty_close if not np.isnan(nifty_close) else 0,
                ema200      if not np.isnan(ema200)      else 0)

    # ── 2. VIX spike ─────────────────────────────────────────────────────────
    vix_val       = _load_vix()
    vix_exit_flag = (not np.isnan(vix_val)) and (vix_val > VIX_EXIT)

    if vix_exit_flag:
        logger.warning("VIX {:.1f} > {:.0f} — forced exit all positions", vix_val, VIX_EXIT)

    # ── 3. Load current holdings ──────────────────────────────────────────────
    holdings = _load_current_holdings()
    logger.info("Active holdings: {}", len(holdings))

    if not holdings:
        if is_bull:
            logger.info("In market (bull) — no open positions, nothing to do")
        else:
            logger.info("In cash (bear regime) — no positions to exit")
        return []

    already_exited = _load_already_exited_today(today)
    held_syms = [h["symbol"] for h in holdings if h["symbol"] not in already_exited]

    if not held_syms:
        logger.info("All positions already handled today")
        return []

    # ── 4. Fetch current prices ───────────────────────────────────────────────
    prices = _load_current_prices(held_syms)

    # ── 5. Generate exit signals ─────────────────────────────────────────────
    sells: list[dict] = []

    for h in holdings:
        sym         = h["symbol"]
        entry_price = h["entry_price"] or 0
        qty         = h["qty"]

        if sym in already_exited:
            continue

        today_close = prices.get(sym)
        if today_close is None:
            logger.warning("{}: no price data — cannot check", sym)
            continue

        unrealized_pct = (
            (today_close - entry_price) / entry_price * 100
            if entry_price and entry_price > 0 else None
        )

        decision = None

        # Regime exit (Nifty < 200d EMA OR VIX > 35) → exit all
        if not is_bull or vix_exit_flag:
            reason = "vix_exit" if vix_exit_flag else "regime_exit"
            if vix_exit_flag:
                detail = f"VIX {vix_val:.1f} > {VIX_EXIT:.0f} — emergency defensive exit"
            else:
                detail = (f"Nifty {nifty_close:,.0f} < EMA200 {ema200:,.0f} "
                          f"— bear regime: exit equity, park in LIQUIDBEES")
            decision = {
                "action": "SELL", "reason": reason, "detail": detail,
                "trigger_price": today_close,
            }

        # Hard floor — even in bull regime: catastrophic single-stock protection
        elif entry_price > 0 and today_close <= entry_price * (1 - HARD_FLOOR_PCT):
            floor_px = entry_price * (1 - HARD_FLOOR_PCT)
            decision = {
                "action": "SELL", "reason": "hard_floor",
                "detail": (f"close {today_close:,.0f} ≤ hard_floor {floor_px:,.0f} "
                           f"(-{HARD_FLOOR_PCT:.0%} from entry {entry_price:,.0f})"),
                "trigger_price": today_close,
            }

        if decision:
            decision.update({
                "date": today, "symbol": sym,
                "qty": qty, "entry_price": entry_price,
                "unrealized_pnl_pct": round(unrealized_pct, 2) if unrealized_pct is not None else None,
            })
            sells.append(decision)
            _log_decision(decision, paper)
            logger.info("EXIT {}: {} — {}", sym, decision["reason"], decision["detail"])
        else:
            pnl_str = f"{unrealized_pct:+.1f}%" if unrealized_pct is not None else "N/A"
            logger.info("HOLD {} — close={:,.0f} entry={:,.0f} ({}) | regime=BULL",
                        sym, today_close, entry_price, pnl_str)

    # ── 6. Telegram notification ─────────────────────────────────────────────
    if notify and sells:
        msg = _format_telegram(sells, today, is_bull, nifty_close, ema200, vix_val, paper)
        ok  = Notifier().send_telegram(msg)
        logger.info("Telegram: {}", "sent" if ok else "failed")
    elif not sells:
        logger.info("No exit signals — no notification sent")

    return sells


def main() -> None:
    ap = argparse.ArgumentParser(description="Momentum daily guard — regime + hard floor")
    ap.add_argument("--paper",     action="store_true", default=True)
    ap.add_argument("--live",      action="store_true",
                    help="Live mode (overrides --paper)")
    ap.add_argument("--no-notify", action="store_true",
                    help="Suppress Telegram notification")
    args = ap.parse_args()

    paper  = not args.live
    notify = not args.no_notify

    if not paper:
        logger.warning("LIVE MODE — orders will be placed via Angel One")

    sells = run_daily_check(paper=paper, notify=notify)

    print(f"\nDaily guard complete: {len(sells)} exit signal(s)")
    for s in sells:
        print(f"  EXIT  {s['symbol']:<14}  {s['reason']:<15}  {s.get('detail','')[:70]}")
    if not sells:
        print("  All clear — no guard signals today")


if __name__ == "__main__":
    main()
