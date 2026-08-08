"""
Daily cron entry point — run after market close (~4:00 PM IST).

Cron line (4 PM IST = 10:30 UTC, Mon–Fri):
  30 10 * * 1-5 /path/to/.venv/bin/python /path/to/scripts/daily_run.py

Idempotency guarantee
---------------------
The script is safe to re-run if the cron job fires twice on the same day.
DuckDB signal inserts use INSERT OR REPLACE so duplicate runs overwrite
with identical data rather than erroring or duplicating rows.

Error handling
--------------
Each pipeline step is wrapped in its own try/except. A failed step is
logged at ERROR level and the script continues to completion (so that
the report and notification still go out, even if, say, EOD2 update
stalled). A step that is truly unrecoverable (e.g. auth failure) will
log ERROR and sys.exit(1) so the cron monitor can alert.

Timing
------
Each step logs start / end wall-clock time. Total elapsed time is logged
on completion. All timestamps are IST (Asia/Kolkata).
"""
from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from datetime import date, timezone, timedelta
from pathlib import Path
from typing import Any

from loguru import logger

# ---------------------------------------------------------------------------
# Logging bootstrap — must happen before any project imports so that all
# downstream loguru calls go to the same handler.
# ---------------------------------------------------------------------------
import os

# Ensure project root is on the path when run directly (e.g. by cron).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import settings  # noqa: E402  (after sys.path patch)

IST = timezone(timedelta(hours=5, minutes=30))

_LOG_FILE = settings.LOG_DIR / "daily_run" / "{time:YYYY-MM-DD}.log"

logger.remove()  # drop default stderr handler
logger.add(
    sys.stderr,
    level=settings.LOG_LEVEL,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
)
logger.add(
    str(_LOG_FILE),
    level="DEBUG",
    rotation="1 day",
    retention="30 days",
    compression="gz",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{line} — {message}",
)

# ---------------------------------------------------------------------------
# Project imports (after path patch)
# ---------------------------------------------------------------------------
from data.auth import get_client, get_session                # noqa: E402
from data.holiday_calendar import is_trading_day             # noqa: E402
from data.eod2_manager import update_eod2                    # noqa: E402
from data.universe import get_universe                       # noqa: E402
from signals.generator import SignalGenerator                # noqa: E402
from portfolio.ledger_a import LedgerA                       # noqa: E402
from reporting.daily_report import generate_report, save_report  # noqa: E402
from reporting.notifier import Notifier                      # noqa: E402
from storage.store import init_schema, load_signals          # noqa: E402


# ---------------------------------------------------------------------------
# Step timing context manager
# ---------------------------------------------------------------------------

@contextmanager
def timed_step(name: str):
    """Log start / end wall time for a pipeline step."""
    t0 = time.perf_counter()
    logger.info("── STEP START: {} ──", name)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        logger.info("── STEP END:   {} ({:.1f}s) ──", name, elapsed)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    run_start = time.perf_counter()
    today = date.today()
    logger.info("=" * 60)
    logger.info("Piedpiper daily run — {}", today.isoformat())
    logger.info("=" * 60)

    # ── Step 0: Schema init (idempotent) ──────────────────────────────────────
    with timed_step("0 — init schema"):
        try:
            init_schema()
        except Exception as exc:
            logger.error("Schema init failed: {} — aborting", exc)
            sys.exit(1)

    # ── Step 1: Re-authenticate with SmartAPI ─────────────────────────────────
    with timed_step("1 — SmartAPI authentication"):
        try:
            session = get_session()
            session.authenticate()
            smart_client = session.get_client()
            logger.info("SmartAPI session acquired")
        except Exception as exc:
            logger.error("SmartAPI authentication failed: {} — aborting", exc)
            sys.exit(1)

    # ── Step 2: Trading-day guard ─────────────────────────────────────────────
    with timed_step("2 — trading-day check"):
        try:
            if not is_trading_day(today):
                logger.info(
                    "{} is not a trading day (weekend or NSE holiday) — exiting cleanly",
                    today,
                )
                sys.exit(0)
            logger.info("{} is a trading day — proceeding", today)
        except Exception as exc:
            logger.warning(
                "Could not determine trading-day status ({}); proceeding cautiously",
                exc,
            )

    # ── Step 3: Update EOD2 data ──────────────────────────────────────────────
    with timed_step("3 — EOD2 update"):
        try:
            update_eod2()
        except Exception as exc:
            logger.error(
                "EOD2 update failed: {} — signals will use yesterday's adjusted data",
                exc,
            )
            # Non-fatal: if today's bars aren't available we still run with stale data

    # ── Step 4: Load universe ─────────────────────────────────────────────────
    universe = None
    with timed_step("4 — load universe"):
        try:
            universe = get_universe()
            logger.info("Universe loaded: {} symbols", len(universe))
        except Exception as exc:
            logger.error("Universe load failed: {} — aborting", exc)
            sys.exit(1)

    # ── Step 5: Initialise Ledger A and get current state ─────────────────────
    ledger = LedgerA()
    ledger_metrics: dict[str, Any] = {}

    with timed_step("5 — Ledger A state"):
        try:
            # Attempt to restore open positions and equity from the DB.
            # Full persistence of LedgerA state across sessions is deferred
            # to Phase 3; here we compute metrics from closed trades stored
            # in the signals table and log the current in-memory state.
            # In a live system this step would hydrate from ledger_a DB table.
            ledger_metrics = ledger.get_metrics()
            logger.info(
                "Ledger A — equity=₹{:,.0f} open_positions={}",
                ledger_metrics.get("current_equity", settings.STARTING_VIRTUAL_CAPITAL),
                len(ledger.open_positions),
            )
        except Exception as exc:
            logger.warning("Ledger A metrics failed: {} — using defaults", exc)
            ledger_metrics = {}

    # ── Step 6: Run signal generator ──────────────────────────────────────────
    signals = None
    with timed_step("6 — signal generation"):
        try:
            generator = SignalGenerator()
            signals = generator.run(
                universe=universe,
                date=today,
                n_open_positions=len(ledger.open_positions),
                total_capital=ledger_metrics.get(
                    "current_equity", settings.STARTING_VIRTUAL_CAPITAL
                ),
            )
            logger.info(
                "Signal generation complete: {} qualifying signal(s)",
                len(signals) if signals is not None else 0,
            )
        except Exception as exc:
            logger.error("Signal generation failed: {}", exc)
            signals = None  # report will note failure

    # ── Step 7: Update Ledger A with new signals ──────────────────────────────
    with timed_step("7 — Ledger A update"):
        if signals is not None and not signals.empty:
            _update_ledger_a(ledger, signals, today)
            try:
                ledger_metrics = ledger.get_metrics()
            except Exception as exc:
                logger.warning("Ledger A post-signal metrics failed: {}", exc)
        else:
            logger.info("No signals — Ledger A unchanged")

    # ── Step 8: Generate and save report ──────────────────────────────────────
    report_path: Path | None = None
    with timed_step("8 — generate report"):
        try:
            report_md = generate_report(
                signals=signals if signals is not None else _empty_signals(),
                date=today,
                ledger_a_metrics=ledger_metrics,
                universe_size=len(universe),
            )
            report_path = save_report(report_md, today)
            logger.info("Report saved to {}", report_path)
        except Exception as exc:
            logger.error("Report generation failed: {}", exc)

    # ── Step 9: Push notification ─────────────────────────────────────────────
    with timed_step("9 — push notification"):
        try:
            notifier = Notifier()
            notifier.send_daily_alert(
                signals=signals if signals is not None else _empty_signals(),
                date=today,
                universe_size=len(universe),
            )
        except Exception as exc:
            logger.error("Notification dispatch failed: {}", exc)

    # ── Done ──────────────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - run_start
    logger.info("=" * 60)
    logger.info(
        "Piedpiper daily run COMPLETE — {} in {:.1f}s",
        today.isoformat(),
        total_elapsed,
    )
    if report_path:
        logger.info("Report: {}", report_path)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _update_ledger_a(ledger: LedgerA, signals: "pd.DataFrame", today: date) -> None:  # type: ignore[name-defined]
    """
    Open virtual positions in Ledger A for today's signals.

    Uses the signal's entry_high as a conservative fill price (next-bar open
    could be anywhere in the entry zone; using entry_high is slightly
    pessimistic, consistent with the plan's conservative fill assumption).
    """
    import pandas as pd

    for _, row in signals.iterrows():
        symbol: str = row["symbol"]
        entry_price = float(row.get("entry_high", row.get("entry_low", 0)))
        pos_size = float(row.get("position_size", 0))
        sl = float(row["stop_loss"])
        tp = float(row["target"])
        conf = float(row["confidence"])
        sector = str(row.get("sector", "Other"))
        symbol_tier = str(row.get("symbol_tier", "midcap100"))

        if entry_price <= 0 or pos_size <= 0:
            logger.debug("Skipping Ledger A open for {} — invalid price/size", symbol)
            continue

        quantity = max(1, int(pos_size // entry_price))

        try:
            ledger.open_position(
                symbol=symbol,
                entry_date=today,
                entry_price=entry_price,
                quantity=quantity,
                stop_loss=sl,
                target=tp,
                confidence=conf,
                sector=sector,
                signal_date=today,
                symbol_tier=symbol_tier,
            )
        except Exception as exc:
            logger.warning("LedgerA.open_position failed for {}: {}", symbol, exc)


def _empty_signals() -> "pd.DataFrame":  # type: ignore[name-defined]
    """Return an empty DataFrame with the expected signal schema columns."""
    import pandas as pd

    return pd.DataFrame(
        columns=[
            "dt", "symbol", "direction", "confidence",
            "entry_low", "entry_high", "stop_loss", "target",
            "max_hold_days", "position_size", "model_version",
            "sector", "symbol_tier", "prev_close", "gap_status",
        ]
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
