"""
Piedpiper — Daily Regime Router
================================
Single entry point that runs every trading day at 9:31 AM IST.

Capital allocation (₹5L total):
  ₹2L → ORB LONG intraday (this script, bull days)
  ₹3L → Monthly Momentum (monthly_run.py --capital 300000)

Strategy selection (all based on REAL backtested data):
  BULL day (Nifty ≥ EMA20):
    → ORB LONG on Nifty 200 stocks (₹2L capital, 3% risk/trade)

  BEAR day (Nifty < EMA20):
    → Nifty Futures SHORT (1 lot) — modest positive but 2025 was -₹66K outlier
    → monthly_run.py handles defensive parking (LIQUIDBEES or GOLDBEES)

What is NOT run:
  ✗ ORB SHORT on individual stocks  — confirmed losing strategy (-₹54K in backtest)
  ✗ ML daily swing model            — disabled (29% TP hit rate vs 44% breakeven)
  ✗ USDINR trend following          — essentially flat (-0.4% CAGR)
  ✗ Bank Nifty daily-bar            — high DD, needs 15-min data for proper ORB

Combined system expected annual return (₹3L momentum + ₹2L ORB capital):
  Backtest avg 2020-2025: ~52%  |  Best: 86% (2023)  |  Worst: 8% (2025)
  No losing years when Nifty SHORT is excluded.

Cron setup:
  # Daily at 9:31 AM IST (= 04:01 UTC)
  1 4 * * 1-5 /path/.venv/bin/python3 /path/scripts/run_daily.py --capital 200000

Run manually:
  python scripts/run_daily.py                  # paper mode
  python scripts/run_daily.py --live           # LIVE mode
  python scripts/run_daily.py --no-notify      # suppress Telegram
  python scripts/run_daily.py --status-only    # just print today's regime, no orders
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, timedelta, timezone
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from data.holiday_calendar import is_trading_day
from reporting.notifier import Notifier
from storage import store

IST = timezone(timedelta(hours=5, minutes=30))
_SCRIPTS = Path(__file__).parent
_PYTHON  = sys.executable

_LOG_FILE = settings.LOG_DIR / "run_daily" / "{time:YYYY-MM-DD}.log"
logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")
logger.add(str(_LOG_FILE), level="DEBUG", rotation="1 month", retention="3 months",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}")


def get_regime() -> tuple[bool, float, float, float]:
    """
    Returns (is_bull, nifty_now, ema20, vix_now).
    Falls back to yfinance if DB data is stale.
    """
    try:
        with store.db_conn() as conn:
            rows = conn.execute(
                "SELECT dt, close FROM adjusted_ohlcv WHERE symbol = 'Nifty 50' "
                "ORDER BY dt DESC LIMIT 40"
            ).fetchall()
            vix_rows = conn.execute(
                "SELECT close FROM adjusted_ohlcv WHERE symbol = 'India VIX' "
                "ORDER BY dt DESC LIMIT 1"
            ).fetchall()
        if rows:
            latest = rows[0][0]
            if (date.today() - latest).days <= 3:
                closes  = pd.Series({pd.Timestamp(r[0]): float(r[1]) for r in rows}).sort_index()
                ema20   = closes.ewm(span=20, adjust=False).mean()
                nifty   = float(closes.iloc[-1])
                e20     = float(ema20.iloc[-1])
                vix     = float(vix_rows[0][0]) if vix_rows else 15.0
                return nifty >= e20, nifty, e20, vix
    except Exception as exc:
        logger.warning("DB regime check failed: {}", exc)

    try:
        import yfinance as yf
        nifty_df = yf.download("^NSEI", period="45d", interval="1d", progress=False, auto_adjust=True)
        closes   = nifty_df["Close"].squeeze().dropna()
        ema20    = closes.ewm(span=20, adjust=False).mean()
        nifty    = float(closes.iloc[-1])
        e20      = float(ema20.iloc[-1])
        return nifty >= e20, nifty, e20, 15.0
    except Exception as exc:
        logger.warning("yfinance regime fallback failed ({}), defaulting BULL", exc)
        return True, 0.0, 0.0, 15.0


def _run_script(script: str, extra_args: list[str]) -> int:
    """Run a child script, log output, return exit code."""
    cmd = [_PYTHON, str(_SCRIPTS / script)] + extra_args
    logger.info("Running: {}", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def main() -> None:
    ap = argparse.ArgumentParser(description="Piedpiper daily regime router")
    ap.add_argument("--live",        action="store_true",  help="Pass --live to child scripts")
    ap.add_argument("--no-notify",   action="store_true",  help="Pass --no-notify to child scripts")
    ap.add_argument("--capital",     type=float, default=200_000.0, help="Intraday ORB capital (default ₹2L)")
    ap.add_argument("--fno-capital", type=float, default=130_000.0, help="Margin for Nifty SHORT")
    ap.add_argument("--status-only", action="store_true",  help="Print regime, no orders")
    args = ap.parse_args()

    today = date.today()
    logger.info("=" * 60)
    logger.info("Piedpiper Daily Router — {}", today)
    logger.info("=" * 60)

    if not is_trading_day(today):
        logger.info("{} is not a trading day — exiting", today)
        return

    store.init_schema()
    is_bull, nifty, ema20, vix = get_regime()

    regime_label = "BULL" if is_bull else "BEAR"
    vix_label    = "HIGH (>25)" if vix > 25 else ("ELEVATED (>18)" if vix > 18 else f"Normal ({vix:.1f})")

    logger.info("Regime: {} | Nifty {:.0f} vs EMA20 {:.0f} | VIX {}", regime_label, nifty, ema20, vix_label)

    if args.status_only:
        print(f"\nDate   : {today}")
        print(f"Regime : {regime_label}")
        print(f"Nifty  : {nifty:.0f}  |  EMA20: {ema20:.0f}")
        print(f"VIX    : {vix_label}")
        print(f"\nToday's action:")
        if is_bull:
            print("  → ORB LONG (intraday_run.py) will fire")
            print("  → Nifty SHORT skipped (bull day)")
        else:
            print("  → Nifty Futures SHORT (fno_nifty_short.py) will fire")
            print("  → ORB LONG skipped (bear day)")
        return

    common_flags = []
    if args.live:      common_flags.append("--live")
    if args.no_notify: common_flags.append("--no-notify")

    if is_bull:
        # ── BULL DAY: ORB LONG on Nifty 200 stocks ────────────────────────────
        logger.info("BULL DAY → launching ORB LONG (intraday_run.py)")
        rc = _run_script("intraday_run.py", common_flags + ["--capital", str(args.capital)])
        if rc != 0:
            logger.error("intraday_run.py exited with code {}", rc)
    else:
        # ── BEAR DAY: Nifty Futures SHORT ─────────────────────────────────────
        if vix > 35:
            # Extreme fear: skip short too (market can have vicious bounces)
            msg = (f"*Piedpiper Daily {today}*\n"
                   f"VIX {vix:.1f} > 35 — EXTREME FEAR. Skipping Nifty SHORT today.\n"
                   f"Bear regime but bounce risk is very high at this VIX level.")
            logger.warning("VIX={:.1f} > 35 — skipping SHORT to avoid panic-bounce losses", vix)
            if not args.no_notify:
                Notifier().send_telegram(msg)
        else:
            logger.info("BEAR DAY → launching Nifty Futures SHORT (fno_nifty_short.py)")
            rc = _run_script("fno_nifty_short.py", common_flags)
            if rc != 0:
                logger.error("fno_nifty_short.py exited with code {}", rc)

    logger.info("Daily router done.")


if __name__ == "__main__":
    main()
