"""
Ledger A — autonomous virtual portfolio.

Mirrors what the ML system would have done if every signal were acted on
without hesitation. Used as the performance benchmark against Ledger B
(Feron's actual trades).

Tracking model:
  - Positions opened at NEXT-BAR OPEN of the signal bar.
  - GTT-style stop-loss / target checked daily against bar's low/high.
  - Stop-loss wins when both SL and TP are breached intraday (conservative).
  - Portfolio constraints enforced at open_position() time.
  - Equity history updated daily via mark_to_market().
"""
from __future__ import annotations

import uuid
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import (
    STARTING_VIRTUAL_CAPITAL,
    MAX_CONCURRENT_POSITIONS,
    MAX_CAPITAL_DEPLOYED_PCT,
    MAX_POSITIONS_PER_SECTOR,
)
from config import cost_model
from backtest.metrics import (
    sharpe_ratio,
    max_drawdown,
    hit_rate,
    avg_win_loss_ratio,
    cagr,
    sortino_ratio,
)


class LedgerA:
    """
    Autonomous virtual portfolio. Tracks positions and P&L independently
    of what Feron actually trades. Uses GTT-style stop/target tracking.
    """

    def __init__(self, start_capital: float = STARTING_VIRTUAL_CAPITAL):
        self.start_capital = start_capital
        self.capital = start_capital            # available cash
        self.open_positions: dict[str, dict] = {}    # symbol → position dict
        self.closed_trades: list[dict] = []
        self.equity_history: list[dict] = []   # [{date, equity, cash, positions_value}]

    # ------------------------------------------------------------------
    # Constraint checking
    # ------------------------------------------------------------------

    def can_open_position(self, symbol: str, sector: str) -> tuple[bool, str]:
        """
        Check all portfolio-level constraints before opening a new position.

        Returns
        -------
        (True, "")                    — all constraints pass
        (False, "<reason_string>")    — blocked, with human-readable reason
        """
        if symbol in self.open_positions:
            return False, f"Already holding {symbol}"

        if len(self.open_positions) >= MAX_CONCURRENT_POSITIONS:
            return False, (
                f"Max positions ({MAX_CONCURRENT_POSITIONS}) already open"
            )

        sector_count = sum(
            1 for pos in self.open_positions.values()
            if pos.get("sector") == sector
        )
        if sector_count >= MAX_POSITIONS_PER_SECTOR:
            return False, (
                f"Sector cap: already {sector_count} positions in '{sector}'"
            )

        # Capital constraint: don't open if less than one slot worth of capital
        total_equity = self._current_equity_estimate()
        max_deployable = total_equity * MAX_CAPITAL_DEPLOYED_PCT
        already_deployed = sum(
            p["entry_price"] * p["quantity"] for p in self.open_positions.values()
        )
        if already_deployed >= max_deployable:
            return False, (
                f"Capital deployment cap ({MAX_CAPITAL_DEPLOYED_PCT:.0%}) reached"
            )

        return True, ""

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def open_position(
        self,
        symbol: str,
        entry_date,
        entry_price: float,
        quantity: int,
        stop_loss: float,
        target: float,
        confidence: float,
        sector: str,
        signal_date,
        symbol_tier: str = "midcap100",
    ) -> None:
        """
        Record opening of a new virtual position.

        Parameters
        ----------
        symbol      : NSE ticker
        entry_date  : date of execution (next bar after signal)
        entry_price : fill price (next-bar open)
        quantity    : number of shares
        stop_loss   : stop-loss price
        target      : take-profit price
        confidence  : model confidence score at signal time
        sector      : sector string (for constraint checking)
        signal_date : bar that generated the signal (for reconciliation)
        symbol_tier : "nifty100" or "midcap100" (drives cost/slippage)
        """
        ok, reason = self.can_open_position(symbol, sector)
        if not ok:
            logger.warning("LedgerA.open_position blocked for {}: {}", symbol, reason)
            return

        entry_value = entry_price * quantity
        if entry_value > self.capital:
            logger.warning(
                "LedgerA: insufficient cash for {} (need ₹{:.0f}, have ₹{:.0f})",
                symbol, entry_value, self.capital
            )
            return

        trade_id = str(uuid.uuid4())
        self.open_positions[symbol] = {
            "trade_id":    trade_id,
            "symbol":      symbol,
            "sector":      sector,
            "signal_date": signal_date,
            "entry_date":  entry_date,
            "entry_price": entry_price,
            "quantity":    quantity,
            "stop_loss":   stop_loss,
            "target":      target,
            "confidence":  confidence,
            "symbol_tier": symbol_tier,
        }
        self.capital -= entry_value

        logger.info(
            "LedgerA OPEN  {} | {} x {} @ ₹{:.2f} | SL=₹{:.2f} TP=₹{:.2f} | conf={:.2f}",
            symbol, quantity, entry_date, entry_price, stop_loss, target, confidence
        )

    def update_positions(
        self, date, prices: dict[str, dict]
    ) -> list[dict]:
        """
        Check each open position against today's high/low for GTT-style exits.

        Parameters
        ----------
        date   : current trading date
        prices : {symbol: {open, high, low, close}}

        Returns
        -------
        List of closed trade dicts (may be empty).
        """
        newly_closed = []

        for symbol in list(self.open_positions.keys()):
            pos = self.open_positions[symbol]
            bar = prices.get(symbol)
            if bar is None:
                continue

            # Do not check exits on the entry bar
            if pd.Timestamp(pos["entry_date"]) == pd.Timestamp(date):
                continue

            sl = pos["stop_loss"]
            tp = pos["target"]
            exit_price = None
            exit_reason = None

            # Stop-loss first (conservative: worst case wins)
            if bar.get("low", float("inf")) <= sl:
                exit_price = sl
                exit_reason = "stop_loss"
            elif bar.get("high", 0) >= tp:
                exit_price = tp
                exit_reason = "target"

            if exit_reason:
                trade = self._close_position(pos, date, exit_price, exit_reason)
                newly_closed.append(trade)
                self.closed_trades.append(trade)
                self.capital += exit_price * pos["quantity"]
                del self.open_positions[symbol]

                logger.info(
                    "LedgerA CLOSE {} ({}) on {} @ ₹{:.2f} | net P&L=₹{:.0f}",
                    symbol, exit_reason, date, exit_price, trade["net_pnl"]
                )

        return newly_closed

    def mark_to_market(self, date, prices: dict[str, dict]) -> float:
        """
        Current portfolio value = cash + mark-to-market of all open positions.

        Parameters
        ----------
        date   : trading date (stored in equity history)
        prices : {symbol: {open, high, low, close}}

        Returns
        -------
        Total portfolio value in ₹.
        """
        positions_value = 0.0
        for symbol, pos in self.open_positions.items():
            bar = prices.get(symbol)
            price = bar["close"] if bar and "close" in bar else pos["entry_price"]
            positions_value += price * pos["quantity"]

        total = self.capital + positions_value
        self.equity_history.append({
            "date":             pd.Timestamp(date),
            "equity":           total,
            "cash":             self.capital,
            "positions_value":  positions_value,
            "n_open":           len(self.open_positions),
        })
        return total

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict:
        """
        Compute performance metrics from closed trades and equity history.

        Returns
        -------
        dict with: sharpe, sortino, max_drawdown, cagr, hit_rate,
                   win_loss_ratio, net_pnl, gross_pnl, n_trades
        """
        trades_df = self.to_dataframe()
        eq_history = pd.DataFrame(self.equity_history)

        if eq_history.empty:
            return {"error": "no equity history recorded"}

        eq_series = eq_history.set_index("date")["equity"]
        eq_series.index = pd.to_datetime(eq_series.index)
        eq_series = eq_series.sort_index()

        daily_returns = eq_series.pct_change().dropna()

        metrics = {
            "sharpe":         sharpe_ratio(daily_returns),
            "sortino":        sortino_ratio(daily_returns),
            "max_drawdown":   max_drawdown(eq_series),
            "cagr":           cagr(eq_series),
            "hit_rate":       hit_rate(trades_df),
            "win_loss_ratio": avg_win_loss_ratio(trades_df),
            "n_trades":       len(trades_df),
            "net_pnl":        float(trades_df["net_pnl"].sum()) if not trades_df.empty else 0.0,
            "gross_pnl":      float(trades_df["gross_pnl"].sum()) if not trades_df.empty else 0.0,
            "current_equity": float(eq_series.iloc[-1]) if not eq_series.empty else self.start_capital,
            "total_return":   float(
                (eq_series.iloc[-1] / self.start_capital) - 1
            ) if not eq_series.empty else 0.0,
        }

        logger.info(
            "LedgerA metrics | Sharpe={:.2f} | MaxDD={:.1%} | HitRate={:.1%} "
            "| Net P&L=₹{:,.0f} | Trades={}",
            metrics["sharpe"], metrics["max_drawdown"],
            metrics["hit_rate"], metrics["net_pnl"], metrics["n_trades"]
        )
        return metrics

    def to_dataframe(self) -> pd.DataFrame:
        """Return all closed trades as a DataFrame."""
        if not self.closed_trades:
            return pd.DataFrame()
        return pd.DataFrame(self.closed_trades)

    def equity_curve(self) -> pd.Series:
        """Return daily equity curve as a pd.Series (DatetimeIndex)."""
        if not self.equity_history:
            return pd.Series(dtype=float)
        df = pd.DataFrame(self.equity_history)
        s = df.set_index("date")["equity"]
        s.index = pd.to_datetime(s.index)
        return s.sort_index()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _close_position(
        self,
        pos: dict,
        exit_date,
        exit_price: float,
        exit_reason: str,
    ) -> dict:
        """Compute costs and P&L, return completed trade dict."""
        qty = pos["quantity"]
        entry_price = pos["entry_price"]
        tier = pos.get("symbol_tier", "midcap100")

        gross_pnl = qty * (exit_price - entry_price)
        costs = cost_model.round_trip_cost(qty, entry_price, exit_price, tier)
        net_pnl = gross_pnl - costs["total"]

        return {
            "trade_id":       pos["trade_id"],
            "symbol":         pos["symbol"],
            "sector":         pos.get("sector", "Unknown"),
            "signal_date":    pos.get("signal_date"),
            "entry_date":     pos["entry_date"],
            "entry_price":    entry_price,
            "quantity":       qty,
            "stop_loss":      pos["stop_loss"],
            "target":         pos["target"],
            "exit_date":      exit_date,
            "exit_price":     exit_price,
            "exit_reason":    exit_reason,
            "gross_pnl":      gross_pnl,
            "net_pnl":        net_pnl,
            "cost_breakdown": costs,
            "confidence":     pos.get("confidence", 0.0),
            "symbol_tier":    tier,
        }

    def _current_equity_estimate(self) -> float:
        """Rough equity estimate using entry prices (no live price needed)."""
        positions_value = sum(
            p["entry_price"] * p["quantity"]
            for p in self.open_positions.values()
        )
        return self.capital + positions_value
