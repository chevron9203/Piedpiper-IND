"""
Ledger B — Feron's actual trade log.

This is the single source of truth for what was actually executed vs. what
signals were seen and declined. All reads/writes go to the DuckDB `ledger_b`
table via storage.store.

Actions stored in the `action` column:
  "trade"    — a position was opened
  "declined" — signal was seen but not acted on
  "exit"     — a trade was closed (updates existing row)
"""
from __future__ import annotations

import uuid
from datetime import date
from typing import Optional

import pandas as pd
from loguru import logger

from storage import store


class LedgerB:
    """
    Logs Feron's actual trades and declined signals.
    Data source of truth is the DB — reads/writes to storage.store (DuckDB ledger_b table).
    """

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def log_trade(
        self,
        symbol: str,
        signal_date,
        entry_date,
        entry_price: float,
        quantity: int,
        gtt_stop: float,
        gtt_target: float,
        notes: str = "",
    ) -> str:
        """
        Log a real trade entry.

        Parameters
        ----------
        symbol      : NSE ticker
        signal_date : the bar that generated the signal
        entry_date  : actual date of order execution
        entry_price : actual fill price
        quantity    : number of shares bought
        gtt_stop    : stop-loss price set as a GTT order
        gtt_target  : take-profit price set as a GTT order
        notes       : free-text field (e.g. "Entered 5 min after open, gap up")

        Returns
        -------
        trade_id : UUID string for later reference
        """
        trade_id = str(uuid.uuid4())

        row = {
            "trade_id":    trade_id,
            "symbol":      symbol,
            "signal_date": _to_date(signal_date),
            "action":      "trade",
            "entry_date":  _to_date(entry_date),
            "entry_price": float(entry_price),
            "quantity":    int(quantity),
            "gtt_stop":    float(gtt_stop),
            "gtt_target":  float(gtt_target),
            "exit_date":   None,
            "exit_price":  None,
            "exit_reason": None,
            "gross_pnl":   None,
            "net_pnl":     None,
            "notes":       notes,
        }
        self._insert_row(row)

        logger.info(
            "LedgerB LOG_TRADE  {} | {} x {} @ ₹{:.2f} | SL=₹{:.2f} TP=₹{:.2f} | id={}",
            symbol, quantity, _to_date(entry_date), entry_price,
            gtt_stop, gtt_target, trade_id
        )
        return trade_id

    def log_declined(
        self,
        symbol: str,
        signal_date,
        reason: str = "",
    ) -> str:
        """
        Log a signal that was seen but deliberately not acted on.

        Returns
        -------
        trade_id : a UUID for the declined-signal record
        """
        trade_id = str(uuid.uuid4())

        row = {
            "trade_id":    trade_id,
            "symbol":      symbol,
            "signal_date": _to_date(signal_date),
            "action":      "declined",
            "entry_date":  None,
            "entry_price": None,
            "quantity":    None,
            "gtt_stop":    None,
            "gtt_target":  None,
            "exit_date":   None,
            "exit_price":  None,
            "exit_reason": None,
            "gross_pnl":   None,
            "net_pnl":     None,
            "notes":       reason,
        }
        self._insert_row(row)

        logger.info(
            "LedgerB DECLINED  {} | signal_date={} | reason={}",
            symbol, _to_date(signal_date), reason or "(none)"
        )
        return trade_id

    def log_exit(
        self,
        trade_id: str,
        exit_date,
        exit_price: float,
        exit_reason: str,
    ) -> None:
        """
        Record the exit of a real trade (GTT triggered, or manual close).

        Parameters
        ----------
        trade_id    : the UUID returned by log_trade()
        exit_date   : date of exit
        exit_price  : actual fill price at exit
        exit_reason : e.g. "gtt_stop", "gtt_target", "manual", "corporate_action"
        """
        with store.db_conn() as conn:
            # Fetch entry data first to compute P&L
            row = conn.execute(
                "SELECT entry_price, quantity FROM ledger_b WHERE trade_id = ?",
                [trade_id]
            ).fetchone()

            if row is None:
                logger.error("LedgerB.log_exit: trade_id {} not found", trade_id)
                return

            entry_price, quantity = row
            if entry_price is None or quantity is None:
                logger.error(
                    "LedgerB.log_exit: trade {} has no entry_price/quantity (was it a declined row?)",
                    trade_id
                )
                return

            ep = float(entry_price)
            qty = int(quantity)
            gross_pnl = qty * (float(exit_price) - ep)

            # Import cost model locally to avoid circular at module level
            from config import cost_model as cm
            costs = cm.round_trip_cost(qty, ep, float(exit_price))
            net_pnl = gross_pnl - costs["total"]

            conn.execute("""
                UPDATE ledger_b SET
                    exit_date   = ?,
                    exit_price  = ?,
                    exit_reason = ?,
                    gross_pnl   = ?,
                    net_pnl     = ?
                WHERE trade_id = ?
            """, [
                _to_date(exit_date),
                float(exit_price),
                exit_reason,
                gross_pnl,
                net_pnl,
                trade_id,
            ])

        logger.info(
            "LedgerB LOG_EXIT  {} on {} @ ₹{:.2f} | reason={} | net P&L=₹{:.0f}",
            trade_id[:8], _to_date(exit_date), exit_price, exit_reason, net_pnl
        )

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_open_trades(self) -> pd.DataFrame:
        """Return all trade rows where exit_date is NULL and action='trade'."""
        with store.db_conn() as conn:
            df = conn.execute("""
                SELECT * FROM ledger_b
                WHERE action = 'trade' AND exit_date IS NULL
                ORDER BY entry_date DESC
            """).df()
        return _parse_dates(df, ["signal_date", "entry_date", "exit_date"])

    def get_closed_trades(self) -> pd.DataFrame:
        """Return all completed trade rows (exit_date IS NOT NULL, action='trade')."""
        with store.db_conn() as conn:
            df = conn.execute("""
                SELECT * FROM ledger_b
                WHERE action = 'trade' AND exit_date IS NOT NULL
                ORDER BY exit_date DESC
            """).df()
        return _parse_dates(df, ["signal_date", "entry_date", "exit_date"])

    def get_declined_signals(self) -> pd.DataFrame:
        """Return all rows where action='declined'."""
        with store.db_conn() as conn:
            df = conn.execute("""
                SELECT * FROM ledger_b
                WHERE action = 'declined'
                ORDER BY signal_date DESC
            """).df()
        return _parse_dates(df, ["signal_date"])

    def get_all_trades(self) -> pd.DataFrame:
        """Return every row in ledger_b (trades + declined)."""
        with store.db_conn() as conn:
            df = conn.execute(
                "SELECT * FROM ledger_b ORDER BY logged_at DESC"
            ).df()
        return _parse_dates(df, ["signal_date", "entry_date", "exit_date"])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _insert_row(self, row: dict) -> None:
        """Insert a single row into ledger_b."""
        df = pd.DataFrame([row])
        with store.db_conn() as conn:
            conn.register("_lb_insert", df)
            conn.execute("""
                INSERT INTO ledger_b (
                    trade_id, symbol, signal_date, action,
                    entry_date, entry_price, quantity,
                    gtt_stop, gtt_target,
                    exit_date, exit_price, exit_reason,
                    gross_pnl, net_pnl, notes
                )
                SELECT
                    trade_id, symbol, signal_date, action,
                    entry_date, entry_price, quantity,
                    gtt_stop, gtt_target,
                    exit_date, exit_price, exit_reason,
                    gross_pnl, net_pnl, notes
                FROM _lb_insert
            """)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _to_date(d) -> Optional[date]:
    if d is None:
        return None
    if isinstance(d, date):
        return d
    return pd.Timestamp(d).date()


def _parse_dates(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df
