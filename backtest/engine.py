"""
Vectorised backtesting engine for the NSE trading signal system.

Execution model (LOCKED):
  - Signal on bar t  → entry at bar (t+1) OPEN.  No same-bar-close fills ever.
  - Stop-loss check : if bar's LOW < stop_loss price → fill at stop_loss price.
  - Target check    : if bar's HIGH > target price  → fill at target price.
  - When both SL and TP are breached on the same bar, stop-loss wins
    (conservative assumption — worst case for the strategy).
  - Signal turning 0 exits at next-bar open (same shift logic as entry).
  - Max hold: closed at open of bar (entry_date + max_hold_days + 1) if still open.

Position sizing:
  - Equal-weight across MAX_CONCURRENT_POSITIONS slots.
  - At most MAX_CAPITAL_DEPLOYED_PCT of current cash+mtm is deployed.
  - At most MAX_POSITIONS_PER_SECTOR symbols from the same sector simultaneously.
  - Positions are ranked by confidence score when slots are contested.
  - Quantity is floored to whole shares; leftover cash stays in reserve.

Cost model:
  - Full Indian cost model via config.cost_model.round_trip_cost().
  - Symbol tier (nifty100 / midcap100) drives slippage rate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import (
    STARTING_VIRTUAL_CAPITAL,
    MAX_CONCURRENT_POSITIONS,
    MAX_CAPITAL_DEPLOYED_PCT,
    MAX_POSITIONS_PER_SECTOR,
    BARRIER_MAX_HOLDING_DAYS,
    MIN_HOLD_TRADING_DAYS,
    VIX_ENTRY_BLOCK_THRESHOLD,
)
from config import cost_model
from backtest.metrics import compute_all_metrics


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

_TRADE_COLUMNS = [
    "trade_id",
    "symbol",
    "sector",
    "signal_date",
    "entry_date",
    "entry_price",
    "quantity",
    "stop_loss",
    "target",
    "exit_date",
    "exit_price",
    "exit_reason",
    "gross_pnl",
    "net_pnl",
    "cost_breakdown",
    "confidence",
    "direction",
    "symbol_tier",
]


class BacktestEngine:
    """
    Event-driven (but vectorised over symbols) backtester.

    Walk-forward folds are handled externally; this class processes one fold
    at a time, or the full out-of-sample period.
    """

    def __init__(
        self,
        start_capital: float = STARTING_VIRTUAL_CAPITAL,
        max_positions: int = MAX_CONCURRENT_POSITIONS,
        max_deployed_pct: float = MAX_CAPITAL_DEPLOYED_PCT,
        max_per_sector: int = MAX_POSITIONS_PER_SECTOR,
        max_hold_days: int = BARRIER_MAX_HOLDING_DAYS,
        min_hold_days: int = MIN_HOLD_TRADING_DAYS,
    ):
        self.start_capital = start_capital
        self.max_positions = max_positions
        self.max_deployed_pct = max_deployed_pct
        self.max_per_sector = max_per_sector
        self.max_hold_days = max_hold_days
        self.min_hold_days = min_hold_days

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        signal_df: pd.DataFrame,
        ohlcv_dict: dict[str, pd.DataFrame],
        sector_map: dict[str, str],
        tp_pct: float = 0.06,
        sl_pct: float = 0.03,
        symbol_tiers: dict[str, str] | None = None,
        vix_df: pd.DataFrame | None = None,
        vix_threshold: float = VIX_ENTRY_BLOCK_THRESHOLD,
    ) -> dict:
        """
        Run the backtest.

        Parameters
        ----------
        signal_df    : date × symbol DataFrame; values are confidence scores.
                       >0 = long signal (enter long at next bar open),
                       <0 = short signal (magnitude = short confidence),
                       0  = no signal.  Index must be DatetimeIndex sorted ascending.
        ohlcv_dict   : {symbol: OHLCV DataFrame} — each with DatetimeIndex and
                       columns [open, high, low, close, volume] (case-insensitive).
        sector_map   : {symbol: sector_string}
        tp_pct       : gross take-profit as fraction of entry price (default 6%)
        sl_pct       : gross stop-loss as fraction of entry price (default 3%)
        symbol_tiers : {symbol: "nifty100"|"midcap100"} for cost/slippage tier.
                       Symbols absent from this dict default to "midcap100".
        vix_df       : India VIX OHLCV DataFrame (DatetimeIndex, needs 'close' col).
                       When VIX close > vix_threshold on a signal date, no new
                       positions are opened that day.  Pass None to disable.
        vix_threshold: India VIX level above which new entries are blocked.

        Returns
        -------
        dict with keys:
          equity_curve  : pd.Series (DatetimeIndex, daily ₹ portfolio value)
          trades        : pd.DataFrame of completed trades
          metrics_gross : dict of metrics computed on gross P&L equity curve
          metrics_net   : dict of metrics computed on net P&L equity curve
        """
        if symbol_tiers is None:
            symbol_tiers = {}

        # Build VIX block set: dates where VIX close > threshold → no new entries
        vix_blocked_dates: set = set()
        if vix_df is not None and not vix_df.empty:
            vcol = next((c for c in vix_df.columns if c.lower() == "close"), None)
            if vcol:
                vix_close = vix_df[vcol].dropna()
                vix_close.index = pd.to_datetime(vix_close.index).normalize()
                vix_blocked_dates = set(vix_close[vix_close > vix_threshold].index)
                logger.info(
                    "VIX regime filter: {} dates blocked (VIX > {:.0f})",
                    len(vix_blocked_dates), vix_threshold,
                )

        signal_df = self._prepare_signals(signal_df)
        ohlcv = self._prepare_ohlcv(ohlcv_dict)  # {sym: normalised df}

        all_dates = signal_df.index
        if all_dates.empty:
            logger.warning("BacktestEngine.run: signal_df is empty, aborting")
            return self._empty_result()

        logger.info(
            "BacktestEngine: {} symbols | {} trading days | capital=₹{:,.0f}",
            len(signal_df.columns),
            len(all_dates),
            self.start_capital,
        )

        cash = self.start_capital
        open_positions: dict[str, dict] = {}  # symbol → position dict
        completed_trades: list[dict] = []
        equity_curve: list[tuple] = []  # (date, equity)

        for i, date in enumerate(all_dates):
            # ── 1. Update open positions: check stop/target on TODAY's bar ──────
            #    Entry was at (t-1) signal → today's open, so we have today's
            #    full bar available for checking SL/TP.
            newly_closed, cash = self._check_exits(
                date, open_positions, ohlcv, cash, symbol_tiers
            )
            completed_trades.extend(newly_closed)

            # ── 2. Check signal-exit (signal turned 0 since last bar) ───────────
            #    Symbols where signal is 0 today and we hold → exit at today's open.
            #    (The signal from yesterday turning 0 means we exit at today's open.)
            if i > 0:
                prev_date = all_dates[i - 1]
                cash = self._signal_exits(
                    date, prev_date, open_positions, signal_df, ohlcv,
                    completed_trades, cash, symbol_tiers, all_dates
                )

            # ── 3. Check max-hold exits ──────────────────────────────────────────
            cash = self._max_hold_exits(
                date, open_positions, ohlcv, completed_trades, cash, symbol_tiers, all_dates
            )

            # ── 4. Open new positions from yesterday's signals ──────────────────
            #    signal_df[date] = signals generated AT CLOSE of (i-1) bar.
            #    We execute them NOW at today's (bar i) OPEN.
            if i > 0:
                prev_date = all_dates[i - 1]
                cash = self._open_entries(
                    signal_df, prev_date, date, ohlcv,
                    open_positions, sector_map, cash, symbol_tiers,
                    tp_pct, sl_pct, vix_blocked_dates
                )

            # ── 5. Mark to market ────────────────────────────────────────────────
            mtm = self._mark_to_market(date, open_positions, ohlcv)
            equity_curve.append((date, cash + mtm))

        # Close any remaining open positions at last bar's close
        last_date = all_dates[-1]
        for symbol in list(open_positions.keys()):
            pos = open_positions[symbol]
            bar = self._get_bar(ohlcv, symbol, last_date)
            if bar is None:
                exit_price = pos["entry_price"]
            else:
                exit_price = bar["close"]
            trade = self._close_position(
                pos, last_date, exit_price, "end_of_backtest", symbol_tiers
            )
            completed_trades.append(trade)
            if pos.get("direction", "long") == "short":
                cash += (2 * pos["entry_price"] - exit_price) * pos["quantity"]
            else:
                cash += exit_price * pos["quantity"]
            del open_positions[symbol]

        eq_series = pd.Series(
            [e for _, e in equity_curve],
            index=pd.DatetimeIndex([d for d, _ in equity_curve]),
            name="equity",
        )

        trades_df = (
            pd.DataFrame(completed_trades, columns=_TRADE_COLUMNS)
            if completed_trades
            else pd.DataFrame(columns=_TRADE_COLUMNS)
        )

        # Compute separate gross/net equity curves from trade P&L
        gross_eq = self._pnl_to_equity(eq_series, trades_df, use_gross=True)
        net_eq = self._pnl_to_equity(eq_series, trades_df, use_gross=False)

        metrics_gross = compute_all_metrics(gross_eq, trades_df, label="gross")
        metrics_net = compute_all_metrics(net_eq, trades_df, label="net")

        logger.info(
            "BacktestEngine finished: {} trades | net P&L=₹{:,.0f} | final equity=₹{:,.0f}",
            len(trades_df),
            trades_df["net_pnl"].sum() if not trades_df.empty else 0,
            eq_series.iloc[-1] if not eq_series.empty else self.start_capital,
        )

        return {
            "equity_curve": eq_series,
            "trades": trades_df,
            "metrics_gross": metrics_gross,
            "metrics_net": metrics_net,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _prepare_signals(self, signal_df: pd.DataFrame) -> pd.DataFrame:
        """Normalise signal DataFrame: DatetimeIndex, sorted, fill NaN with 0."""
        df = signal_df.copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.fillna(0.0)
        return df

    def _prepare_ohlcv(
        self, ohlcv_dict: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """Normalise each symbol's OHLCV: DatetimeIndex, lowercase columns, sorted."""
        result = {}
        for symbol, df in ohlcv_dict.items():
            _df = df.copy()
            _df.index = pd.to_datetime(_df.index)
            _df.columns = [c.lower() for c in _df.columns]
            _df = _df.sort_index()
            result[symbol] = _df
        return result

    def _get_bar(
        self, ohlcv: dict[str, pd.DataFrame], symbol: str, date: pd.Timestamp
    ) -> dict | None:
        """Return {open, high, low, close} for symbol on date, or None."""
        df = ohlcv.get(symbol)
        if df is None or date not in df.index:
            return None
        row = df.loc[date]
        return {
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
        }

    def _portfolio_value(
        self,
        cash: float,
        open_positions: dict,
        ohlcv: dict,
        date: pd.Timestamp,
    ) -> float:
        mtm = self._mark_to_market(date, open_positions, ohlcv)
        return cash + mtm

    def _mark_to_market(
        self,
        date: pd.Timestamp,
        open_positions: dict,
        ohlcv: dict,
    ) -> float:
        """Sum of (close × quantity) for all open positions on date."""
        total = 0.0
        for symbol, pos in open_positions.items():
            bar = self._get_bar(ohlcv, symbol, date)
            price = bar["close"] if bar else pos["entry_price"]
            if pos.get("direction", "long") == "short":
                # Short MTM: margin locked + unrealized gain/loss
                # = entry_price*qty + (entry_price - price)*qty = (2*entry - price)*qty
                total += (2 * pos["entry_price"] - price) * pos["quantity"]
            else:
                total += price * pos["quantity"]
        return total

    def _position_slot_available(
        self,
        symbol: str,
        sector: str,
        open_positions: dict,
        sector_map: dict,
    ) -> tuple[bool, str]:
        """Check position-level constraints. Returns (ok, reason)."""
        if symbol in open_positions:
            return False, "already_in_position"
        if len(open_positions) >= self.max_positions:
            return False, "max_positions_reached"
        sector_count = sum(
            1 for s in open_positions
            if sector_map.get(s, "Unknown") == sector
        )
        if sector_count >= self.max_per_sector:
            return False, f"sector_cap_reached_{sector}"
        return True, ""

    def _calc_quantity(
        self,
        entry_price: float,
        cash: float,
        open_positions: dict,
        ohlcv: dict,
        date: pd.Timestamp,
    ) -> int:
        """
        Position size = equal allocation across max_positions slots,
        capped at MAX_CAPITAL_DEPLOYED_PCT of total portfolio value.
        Returns whole number of shares (floor). Returns 0 if not enough cash.
        """
        total_value = cash + self._mark_to_market(date, open_positions, ohlcv)
        max_deployable = total_value * self.max_deployed_pct
        per_slot = max_deployable / self.max_positions
        # Limit to available cash (can't use MTM of open positions as fresh cash)
        available = min(cash, per_slot)
        if available < entry_price:
            return 0
        return int(available // entry_price)

    def _open_entries(
        self,
        signal_df: pd.DataFrame,
        signal_date: pd.Timestamp,
        exec_date: pd.Timestamp,
        ohlcv: dict,
        open_positions: dict,
        sector_map: dict,
        cash: float,
        symbol_tiers: dict,
        tp_pct: float,
        sl_pct: float,
        vix_blocked_dates: set | None = None,
    ) -> float:
        """
        Open new positions for all signals from signal_date, executed at exec_date's OPEN.
        Positive signal values → long entry; negative values → short entry.
        Long signals processed first (higher priority when slots are contested).
        Returns updated cash.
        """
        if signal_date not in signal_df.index:
            return cash

        # VIX regime filter: if VIX was high on the signal date, skip all entries
        if vix_blocked_dates and signal_date.normalize() in vix_blocked_dates:
            logger.debug("VIX blocked: no new entries on {}", signal_date.date())
            return cash

        day_signals = signal_df.loc[signal_date]

        # Long signals: positive confidence, highest first
        long_cands = day_signals[day_signals > 0].sort_values(ascending=False)
        for symbol, confidence in long_cands.items():
            cash = self._try_open_position(
                symbol, float(confidence), "long",
                signal_date, exec_date, ohlcv, open_positions,
                sector_map, cash, symbol_tiers, tp_pct, sl_pct,
            )

        # Short signals: negative confidence, most-negative (highest short conf) first
        short_cands = day_signals[day_signals < 0].sort_values(ascending=True)
        for symbol, neg_confidence in short_cands.items():
            cash = self._try_open_position(
                symbol, abs(float(neg_confidence)), "short",
                signal_date, exec_date, ohlcv, open_positions,
                sector_map, cash, symbol_tiers, tp_pct, sl_pct,
            )

        return cash

    def _try_open_position(
        self,
        symbol: str,
        confidence: float,
        direction: str,
        signal_date: pd.Timestamp,
        exec_date: pd.Timestamp,
        ohlcv: dict,
        open_positions: dict,
        sector_map: dict,
        cash: float,
        symbol_tiers: dict,
        tp_pct: float,
        sl_pct: float,
    ) -> float:
        """Attempt to open one position (long or short). Returns updated cash."""
        sector = sector_map.get(symbol, "Unknown")
        ok, _ = self._position_slot_available(symbol, sector, open_positions, sector_map)
        if not ok:
            return cash

        bar = self._get_bar(ohlcv, symbol, exec_date)
        if bar is None:
            logger.debug("No OHLCV for {} on {} (exec date), skipping", symbol, exec_date.date())
            return cash

        entry_price = bar["open"]  # NEXT-BAR OPEN — locked constraint
        if entry_price <= 0:
            return cash

        quantity = self._calc_quantity(entry_price, cash, open_positions, ohlcv, exec_date)
        if quantity <= 0:
            logger.debug("Insufficient cash for {} on {}: need ₹{:.0f}, have ₹{:.0f}",
                         symbol, exec_date.date(), entry_price, cash)
            return cash

        entry_value = quantity * entry_price
        if entry_value > cash:
            return cash

        tier = symbol_tiers.get(symbol, "midcap100")
        if direction == "short":
            stop_loss = entry_price * (1 + sl_pct)   # short SL: price rises
            target    = entry_price * (1 - tp_pct)   # short TP: price falls
        else:
            stop_loss = entry_price * (1 - sl_pct)
            target    = entry_price * (1 + tp_pct)

        open_positions[symbol] = {
            "symbol":      symbol,
            "sector":      sector,
            "signal_date": signal_date,
            "entry_date":  exec_date,
            "entry_price": entry_price,
            "quantity":    quantity,
            "stop_loss":   stop_loss,
            "target":      target,
            "confidence":  confidence,
            "symbol_tier": tier,
            "direction":   direction,
        }
        cash -= entry_value

        logger.debug("ENTER {} ({}) on {} @ ₹{:.2f} x{} | SL=₹{:.2f} TP=₹{:.2f}",
                     symbol, direction, exec_date.date(), entry_price, quantity, stop_loss, target)
        return cash

    def _check_exits(
        self,
        date: pd.Timestamp,
        open_positions: dict,
        ohlcv: dict,
        cash: float,
        symbol_tiers: dict,
    ) -> tuple[list[dict], float]:
        """
        For every open position, check if today's bar hits stop or target.
        Stop-loss wins when both are hit on the same bar (conservative).

        Returns (list_of_closed_trades, updated_cash).
        """
        closed = []
        for symbol in list(open_positions.keys()):
            pos = open_positions[symbol]
            # Don't check exits on the entry bar itself
            if pos["entry_date"] == date:
                continue

            bar = self._get_bar(ohlcv, symbol, date)
            if bar is None:
                continue

            sl = pos["stop_loss"]
            tp = pos["target"]
            exit_price = None
            exit_reason = None
            direction = pos.get("direction", "long")

            if direction == "short":
                # Short SL: high reaches up to stop → worst-case fills at stop (SL priority)
                if bar["high"] >= sl:
                    exit_price = sl
                    exit_reason = "stop_loss"
                # Short TP: low falls to target
                if bar["low"] <= tp and exit_reason is None:
                    exit_price = tp
                    exit_reason = "target"
            else:
                # Long SL: low breaches SL → fill at SL (worst-case)
                if bar["low"] <= sl:
                    exit_price = sl
                    exit_reason = "stop_loss"
                # Long TP: high breaches TP → fill at TP (SL wins if both hit)
                if bar["high"] >= tp and exit_reason is None:
                    exit_price = tp
                    exit_reason = "target"

            if exit_reason:
                trade = self._close_position(
                    pos, date, exit_price, exit_reason, symbol_tiers
                )
                closed.append(trade)
                if direction == "short":
                    cash += (2 * pos["entry_price"] - exit_price) * pos["quantity"]
                else:
                    cash += exit_price * pos["quantity"]
                del open_positions[symbol]
                logger.debug(
                    "EXIT {} ({}) on {} @ ₹{:.2f} | net P&L=₹{:.0f}",
                    symbol, exit_reason, date.date(), exit_price, trade["net_pnl"]
                )

        return closed, cash

    def _signal_exits(
        self,
        date: pd.Timestamp,
        prev_date: pd.Timestamp,
        open_positions: dict,
        signal_df: pd.DataFrame,
        ohlcv: dict,
        completed_trades: list,
        cash: float,
        symbol_tiers: dict,
        all_dates: pd.DatetimeIndex,
    ) -> float:
        """
        If a held symbol's signal turned 0 on prev_date, exit at today's open.
        (Signal from bar t=0 → execute exit at bar t+1 open — same shift as entry.)
        Minimum hold of self.min_hold_days trading bars is enforced to avoid
        churning through round-trip costs on 1-day holds.
        """
        if prev_date not in signal_df.index:
            return cash

        prev_signals = signal_df.loc[prev_date]
        current_bar_idx = all_dates.get_loc(date)

        for symbol in list(open_positions.keys()):
            # Skip if stop/target already closed this bar (already removed from dict)
            if symbol not in open_positions:
                continue
            pos = open_positions[symbol]
            # Don't exit on the same bar as entry
            if pos["entry_date"] == date:
                continue

            sig = prev_signals.get(symbol, 0)
            direction = pos.get("direction", "long")
            if direction == "long" and sig > 0:
                continue  # long signal still active
            if direction == "short" and sig < 0:
                continue  # short signal still active

            # Enforce minimum hold period (trading bars, not calendar days)
            try:
                entry_bar_idx = all_dates.get_loc(pd.Timestamp(pos["entry_date"]))
                hold_bars = current_bar_idx - entry_bar_idx
            except KeyError:
                hold_bars = self.min_hold_days  # unknown — allow exit
            if hold_bars < self.min_hold_days:
                continue  # too early; hold until min_hold_days even without signal

            bar = self._get_bar(ohlcv, symbol, date)
            if bar is None:
                # No price data — close at entry price (safe fallback)
                exit_price = pos["entry_price"]
            else:
                exit_price = bar["open"]  # Next-bar open exit

            trade = self._close_position(
                pos, date, exit_price, "signal_exit", symbol_tiers
            )
            completed_trades.append(trade)
            if pos.get("direction", "long") == "short":
                cash += (2 * pos["entry_price"] - exit_price) * pos["quantity"]
            else:
                cash += exit_price * pos["quantity"]
            del open_positions[symbol]
            logger.debug(
                "EXIT {} ({}, signal_exit) on {} @ ₹{:.2f} | hold={} bars | net P&L=₹{:.0f}",
                symbol, pos.get("direction", "long"), date.date(), exit_price, hold_bars, trade["net_pnl"]
            )

        return cash

    def _max_hold_exits(
        self,
        date: pd.Timestamp,
        open_positions: dict,
        ohlcv: dict,
        completed_trades: list,
        cash: float,
        symbol_tiers: dict,
        all_dates: pd.DatetimeIndex,
    ) -> float:
        """
        Exit positions that have been held for max_hold_days trading bars.
        Exit at today's open. Uses actual trading-day count from the signal index.
        """
        current_bar_idx = all_dates.get_loc(date)

        for symbol in list(open_positions.keys()):
            pos = open_positions[symbol]
            try:
                entry_bar_idx = all_dates.get_loc(pd.Timestamp(pos["entry_date"]))
                hold_bars = current_bar_idx - entry_bar_idx
            except KeyError:
                # Entry date not in all_dates — fall back to calendar-day approximation
                hold_bars = (date - pd.Timestamp(pos["entry_date"])).days
            if hold_bars < self.max_hold_days:
                continue

            bar = self._get_bar(ohlcv, symbol, date)
            exit_price = bar["open"] if bar else pos["entry_price"]

            trade = self._close_position(
                pos, date, exit_price, "max_hold", symbol_tiers
            )
            completed_trades.append(trade)
            if pos.get("direction", "long") == "short":
                cash += (2 * pos["entry_price"] - exit_price) * pos["quantity"]
            else:
                cash += exit_price * pos["quantity"]
            del open_positions[symbol]
            logger.debug(
                "EXIT {} ({}, max_hold) on {} @ ₹{:.2f} after {} bars",
                symbol, pos.get("direction", "long"), date.date(), exit_price, hold_bars
            )

        return cash

    def _close_position(
        self,
        pos: dict,
        exit_date: pd.Timestamp,
        exit_price: float,
        exit_reason: str,
        symbol_tiers: dict,
    ) -> dict:
        """Compute gross/net P&L and return a completed trade dict."""
        quantity = pos["quantity"]
        entry_price = pos["entry_price"]
        tier = symbol_tiers.get(pos["symbol"], pos.get("symbol_tier", "midcap100"))

        direction = pos.get("direction", "long")
        if direction == "short":
            gross_pnl = quantity * (entry_price - exit_price)
        else:
            gross_pnl = quantity * (exit_price - entry_price)
        costs = cost_model.round_trip_cost(quantity, entry_price, exit_price, tier)
        net_pnl = gross_pnl - costs["total"]

        return {
            "trade_id":       _make_trade_id(pos["symbol"], pos["entry_date"]),
            "symbol":         pos["symbol"],
            "sector":         pos.get("sector", "Unknown"),
            "signal_date":    pos.get("signal_date"),
            "entry_date":     pos["entry_date"],
            "entry_price":    entry_price,
            "quantity":       quantity,
            "stop_loss":      pos["stop_loss"],
            "target":         pos["target"],
            "exit_date":      exit_date,
            "exit_price":     exit_price,
            "exit_reason":    exit_reason,
            "gross_pnl":      gross_pnl,
            "net_pnl":        net_pnl,
            "cost_breakdown": costs,
            "confidence":     pos.get("confidence", 0.0),
            "direction":      direction,
            "symbol_tier":    tier,
        }

    def _pnl_to_equity(
        self,
        base_eq: pd.Series,
        trades_df: pd.DataFrame,
        use_gross: bool,
    ) -> pd.Series:
        """
        base_eq is the GROSS equity curve (cash accounting never deducts costs on exit).
        Gross  → return base_eq as-is.
        Net    → subtract cumulative costs on each exit date so they propagate forward.
        """
        if trades_df.empty or use_gross:
            return base_eq.copy()

        trades_copy = trades_df.copy()
        trades_copy["exit_date"] = pd.to_datetime(trades_copy["exit_date"])
        trades_copy["costs"] = (
            trades_copy["gross_pnl"].fillna(0) - trades_copy["net_pnl"].fillna(0)
        )
        daily_costs = trades_copy.groupby("exit_date")["costs"].sum()

        result = base_eq.copy()
        for exit_dt, cost in daily_costs.items():
            dt = pd.Timestamp(exit_dt)
            if dt in result.index:
                result.loc[dt:] -= cost
        return result

    def _empty_result(self) -> dict:
        return {
            "equity_curve": pd.Series(dtype=float),
            "trades": pd.DataFrame(columns=_TRADE_COLUMNS),
            "metrics_gross": {},
            "metrics_net": {},
        }


# ---------------------------------------------------------------------------
# Walk-forward split utility
# ---------------------------------------------------------------------------

def walk_forward_splits(
    dates: pd.DatetimeIndex,
    train_years: int = 3,
    test_days: int = 63,
    embargo_days: int = 10,
) -> list[dict]:
    """
    Generate purged + embargoed walk-forward splits.

    Parameters
    ----------
    dates        : full sorted DatetimeIndex of available trading days
    train_years  : minimum training window in years (default 3)
    test_days    : test fold size in trading days (default 63 = quarterly)
    embargo_days : embargo gap between train end and test start

    Returns
    -------
    list of dicts: [{train_start, train_end, embargo_end, test_start, test_end}, ...]
    """
    min_train_days = int(train_years * 252)
    splits = []
    n = len(dates)

    fold_start = min_train_days
    while fold_start + test_days <= n:
        train_end_idx = fold_start - 1
        embargo_end_idx = min(fold_start + embargo_days - 1, n - 1)
        test_start_idx = embargo_end_idx + 1
        test_end_idx = min(test_start_idx + test_days - 1, n - 1)

        if test_start_idx >= n:
            break

        splits.append({
            "train_start":  dates[0],
            "train_end":    dates[train_end_idx],
            "embargo_end":  dates[embargo_end_idx],
            "test_start":   dates[test_start_idx],
            "test_end":     dates[test_end_idx],
        })

        fold_start += test_days  # advance by one test fold

    logger.info(
        "walk_forward_splits: {} folds | train_years={} | test_days={} | embargo_days={}",
        len(splits), train_years, test_days, embargo_days
    )
    return splits


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_trade_counter: dict[str, int] = {}


def _make_trade_id(symbol: str, entry_date) -> str:
    """Generate a unique trade ID: SYMBOL_YYYYMMDD_N."""
    key = f"{symbol}_{entry_date}"
    _trade_counter[key] = _trade_counter.get(key, 0) + 1
    date_str = pd.Timestamp(entry_date).strftime("%Y%m%d")
    return f"{symbol}_{date_str}_{_trade_counter[key]:03d}"
