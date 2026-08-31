"""
Piedpiper — Live ORB Trading Engine + Momentum Portfolio Observer
==================================================================
Runs continuously 9:15 AM–3:30 PM IST every trading day.

Two simultaneous modes on the same WebSocket:

1. ORB LONG (₹2L intraday):
   State machine: WARMUP → BUILDING (9:15) → SCANNING (9:30–10:00)
                  → MANAGING → SQUAREOFF (3:15) → DONE (3:30)
   Entry: LTP breaks ORB high + 3× vol surge + stock EMA20 filter
   Trailing stop: breakeven at +0.5×ORB, lock +0.5× at +1.0×ORB, target at +1.5×ORB

2. Momentum Portfolio Observer (₹3L CNC, passive watch):
   Loads current month's top-10 momentum picks from monthly_signals DB.
   Tracks intraday LTP for each holding throughout the day.
   Alerts via Telegram if any holding drops >8% from yesterday's close.
   Also flags if a momentum stock generates an ORB breakout signal (cross-signal).
   EOD summary includes momentum portfolio P&L for the day.

Capital: ₹2L ORB | Risk: 3% per trade | Max ORB trades/day: 5

Run:
  python scripts/live_orb.py              # paper mode (safe)
  python scripts/live_orb.py --live       # REAL orders
  python scripts/live_orb.py --test       # pre-market check (no WebSocket)
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from data.holiday_calendar import is_trading_day
from data.instrument_master import get_nse_equity_master
from data.live_feed import LiveFeed
from data.universe import fetch_nifty200
from execution.angel_orders import AngelOrderExecutor
from reporting.notifier import Notifier
from storage import store

IST       = timezone(timedelta(hours=5, minutes=30))
CAPITAL   = 200_000.0
RISK_PCT  = 0.03       # 3% of capital per trade → ₹6,000 on ₹2L
STOP_FRAC = 0.40       # stop at 40% of ORB range below entry
TARGET_MULT = 1.5      # hard target at 1.5× ORB range
TRAIL1_FRAC = 0.5      # when +0.5×: move stop to entry (breakeven)
TRAIL2_FRAC = 1.0      # when +1.0×: lock in 0.5× gain
VOL_SURGE  = 3.0       # volume at 9:30 must be 3× 9:15-bar avg
MAX_TRADES = 0         # 0 = unlimited; risk is capped per-trade at RISK_PCT anyway

_LOG = settings.LOG_DIR / "live_orb" / "{time:YYYY-MM-DD}.log"
logger.remove()
logger.add(sys.stderr,  level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")
logger.add(str(_LOG),   level="DEBUG", rotation="1 month", retention="3 months",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}")


# ── State machine ─────────────────────────────────────────────────────────────

class Phase(Enum):
    WARMUP    = auto()   # before 9:15
    BUILDING  = auto()   # 9:15–9:29
    SCANNING  = auto()   # 9:30–10:00
    MANAGING  = auto()   # any time — positions are open
    SQUAREOFF = auto()   # 3:15
    DONE      = auto()   # after 3:30


@dataclass
class ORBRange:
    symbol:   str
    high:     float = 0.0
    low:      float = float("inf")
    vol_9_15: float = 0.0    # volume captured in the 9:15 bar


@dataclass
class Position:
    symbol:    str
    token:     str
    entry:     float
    qty:       int
    orb_range: float          # ORB high - ORB low
    stop:      float
    target:    float
    trail_lv:  int   = 0      # 0=initial, 1=breakeven, 2=locked
    peak:      float = 0.0
    order_id:  str   = ""
    paper:     bool  = True


MOM_ALERT_DROP   = 0.08   # informational alert threshold (drop from prev close)
MOM_CAPITAL      = 300_000.0
MOM_HARD_STOP    = 0.15   # -15% from entry: exit no matter what
MOM_UNDERPERF    = 0.05   # stock underperforms market by >5% → Signal 2
MOM_VOL_SURGE    = 1.5    # Signal 2 volume: >1.5× pro-rated daily avg
MOM_MARKET_CRASH = 0.02   # Nifty proxy down >2% → skip exit (market-wide event)


@dataclass
class MomentumWatch:
    """Smart observer + auto-exit for a momentum CNC holding."""
    symbol:             str
    token:              str
    weight:             float        # portfolio weight (0–1)
    entry_price:        float        # price at which we entered (from momentum_entries)
    prev_close:         float        # yesterday's close (for alert threshold)
    ema20:              float        # 20-day EMA as of yesterday
    vol_avg20:          float        # 20-day avg daily volume
    prev_day_below_ema: bool         # was yesterday's close < EMA20? (Signal 3 history)
    approx_qty:         int
    hard_stop:          float        # entry × (1 - MOM_HARD_STOP)
    # intraday state
    ltp:          float = 0.0
    open_price:   float = 0.0       # today's opening price (from WebSocket open field)
    day_high:     float = 0.0
    day_low:      float = float("inf")
    alert_fired:  bool  = False     # 8% info-alert sent
    exit_fired:   bool  = False     # exit order placed — prevent double-exit
    cross_signal: bool  = False     # ORB breakout also fired on this stock


def load_momentum_portfolio(
    sym_to_tok: dict[str, str],
    premarket:  dict[str, dict],
) -> list[MomentumWatch]:
    """
    Reads the latest month's momentum holdings from monthly_signals.
    Merges with momentum_entries (entry prices + hard stops) and premarket
    data (EMA20, vol_avg, prev_day_below_ema).
    Returns [] if currently in defensive/cash mode.
    """
    try:
        with store.db_conn() as conn:
            row = conn.execute("""
                SELECT signal_date, in_market, target, weights
                FROM monthly_signals
                WHERE in_market = TRUE
                ORDER BY signal_date DESC
                LIMIT 1
            """).fetchone()
    except Exception as exc:
        logger.warning("Could not load momentum portfolio: {}", exc)
        return []

    if row is None:
        logger.info("Momentum: currently in DEFENSIVE/CASH mode — no holdings to watch")
        return []

    sig_date, in_market, target_json, weights_json = row
    try:
        symbols = json.loads(target_json) if isinstance(target_json, str) else target_json
        weights = json.loads(weights_json) if isinstance(weights_json, str) else weights_json
    except Exception:
        logger.warning("Momentum: could not parse target/weights JSON")
        return []

    if not symbols:
        return []

    # Load entry prices from momentum_entries (set by monthly_run.py)
    entry_map: dict[str, dict] = {}
    try:
        with store.db_conn() as conn:
            rows = conn.execute("""
                SELECT symbol, entry_price, hard_stop, qty
                FROM momentum_entries
                WHERE signal_date = ?
            """, [str(sig_date)]).fetchall()
            entry_map = {r[0]: {"entry_price": r[1], "hard_stop": r[2], "qty": r[3]}
                         for r in rows}
    except Exception as exc:
        logger.warning("Momentum: could not load entry prices ({}), using prev_close fallback", exc)

    # Load yesterday's close as fallback for entry price
    prev_close_map: dict[str, float] = {}
    try:
        with store.db_conn() as conn:
            rows = conn.execute("""
                SELECT a.symbol, a.close
                FROM adjusted_ohlcv a
                INNER JOIN (
                    SELECT symbol, MAX(dt) AS max_dt
                    FROM adjusted_ohlcv
                    WHERE symbol = ANY(?) AND dt < CURRENT_DATE
                    GROUP BY symbol
                ) b ON a.symbol = b.symbol AND a.dt = b.max_dt
            """, [symbols]).fetchall()
            prev_close_map = {r[0]: float(r[1]) for r in rows}
    except Exception as exc:
        logger.warning("Momentum: could not load prev closes: {}", exc)

    watches = []
    for sym in symbols:
        tok = sym_to_tok.get(sym, "")
        if not tok:
            logger.debug("Momentum: no token for {} — skipping", sym)
            continue

        w        = float(weights.get(sym, 1.0 / len(symbols)))
        prev_c   = prev_close_map.get(sym, 0.0)
        pm       = premarket.get(sym, {})
        ema20    = pm.get("ema20", 0.0)
        vol_avg  = pm.get("vol_avg20", 0.0)
        prev_below_ema = pm.get("prev_day_below_ema", False)

        # Entry price: use momentum_entries table; fall back to yesterday's close
        e = entry_map.get(sym, {})
        entry_px  = e.get("entry_price") or prev_c
        hard_stop = e.get("hard_stop") or (round(entry_px * (1 - MOM_HARD_STOP), 2) if entry_px > 0 else 0.0)
        qty       = e.get("qty") or (max(1, int((MOM_CAPITAL * w) / entry_px)) if entry_px > 0 else 0)

        watches.append(MomentumWatch(
            symbol=sym, token=tok, weight=w,
            entry_price=entry_px, prev_close=prev_c,
            ema20=ema20, vol_avg20=vol_avg,
            prev_day_below_ema=prev_below_ema,
            approx_qty=qty, hard_stop=hard_stop,
        ))

    entry_src = "momentum_entries" if entry_map else "prev_close fallback"
    logger.info("Momentum: {} holdings | signal={} | entry prices from {}",
                len(watches), sig_date, entry_src)
    for w in watches:
        logger.info("  {} | entry=₹{:.2f} | hard_stop=₹{:.2f} | ema20=₹{:.2f} | prev_below_ema={}",
                    w.symbol, w.entry_price, w.hard_stop, w.ema20, w.prev_day_below_ema)
    return watches


# ── Pre-market data loader ────────────────────────────────────────────────────

def load_premarket_data(symbols: list[str]) -> dict[str, dict]:
    """
    Load yesterday's EMA20, 20d-avg-volume, and EMA-break history for each symbol.
    prev_day_below_ema: was yesterday's close below its EMA20? (used for Signal 3)
    """
    logger.info("Loading pre-market data from DB for {} symbols ...", len(symbols))
    with store.db_conn() as conn:
        df = conn.execute("""
            SELECT symbol, dt, close, volume
            FROM adjusted_ohlcv
            WHERE symbol = ANY(?) AND dt >= CURRENT_DATE - INTERVAL 60 DAY
            ORDER BY symbol, dt
        """, [symbols]).df()

    df["dt"] = pd.to_datetime(df["dt"])
    result = {}
    for sym, g in df.groupby("symbol"):
        g         = g.sort_values("dt")
        ema_series = g["close"].ewm(span=20, adjust=False).mean()
        ema20      = float(ema_series.iloc[-1])
        vol_avg    = float(g["volume"].rolling(20, min_periods=5).mean().iloc[-1])
        last_close = float(g["close"].iloc[-1])
        # Was the PREVIOUS day's close below its own EMA20? (Signal 3 look-back)
        prev_day_below_ema = False
        if len(g) >= 2:
            prev_close_val = float(g["close"].iloc[-2])
            prev_ema_val   = float(ema_series.iloc[-2])
            prev_day_below_ema = prev_close_val < prev_ema_val
        result[sym] = {
            "ema20":             ema20,
            "vol_avg20":         vol_avg,
            "last_close":        last_close,
            "prev_day_below_ema": prev_day_below_ema,
        }

    logger.info("Pre-market data loaded for {} symbols", len(result))
    return result


def nifty_is_bull() -> bool:
    """Check Nifty regime from DB (same logic as run_daily.py)."""
    try:
        with store.db_conn() as conn:
            rows = conn.execute(
                "SELECT dt, close FROM adjusted_ohlcv WHERE symbol='Nifty 50' "
                "ORDER BY dt DESC LIMIT 40"
            ).fetchall()
        if rows:
            closes = pd.Series({pd.Timestamp(r[0]): float(r[1]) for r in rows}).sort_index()
            ema20  = closes.ewm(span=20, adjust=False).mean()
            bull   = closes.iloc[-1] >= ema20.iloc[-1]
            logger.info("Regime: Nifty {:.0f} vs EMA20 {:.0f} → {}",
                        closes.iloc[-1], ema20.iloc[-1], "BULL" if bull else "BEAR")
            return bool(bull)
    except Exception as exc:
        logger.warning("Regime check failed: {} — defaulting BULL", exc)
    return True


def build_token_map(symbols: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    """
    Returns (sym→token, token→sym) for all symbols found in instrument master.
    Tries SYMBOL-EQ form if plain SYMBOL not found.
    """
    master   = get_nse_equity_master()
    sym_tok  = {}
    tok_sym  = {}
    sym_set  = set(master["symbol"].values)
    for sym in symbols:
        candidate = sym if sym in sym_set else f"{sym}-EQ"
        row = master[master["symbol"] == candidate]
        if not row.empty:
            tok = str(row.iloc[0]["token"])
            sym_tok[sym] = tok
            tok_sym[tok] = sym
    logger.info("Token map: {}/{} symbols resolved", len(sym_tok), len(symbols))
    return sym_tok, tok_sym


# ── Main engine ───────────────────────────────────────────────────────────────

class LiveORBEngine:

    def __init__(self, paper: bool, no_notify: bool):
        self.paper      = paper
        self.no_notify  = no_notify
        self.phase      = Phase.WARMUP
        self._lock      = threading.Lock()

        # Per-symbol state
        self.orb_ranges: dict[str, ORBRange]  = {}
        self.positions:  dict[str, Position]  = {}
        self.fired_syms: set[str]             = set()   # already traded today

        # Pre-loaded reference data
        self.premarket: dict[str, dict] = {}
        self.sym_to_token: dict[str, str] = {}
        self.tok_to_sym:   dict[str, str] = {}

        # Momentum portfolio observer
        self.mom_watches: dict[str, MomentumWatch] = {}   # token → watch
        self.mom_by_sym:  dict[str, MomentumWatch] = {}   # symbol → watch

        # Market proxy: track day-open vs LTP for all subscribed stocks
        # Used to detect market-wide crashes (don't exit momentum on systemic drops)
        self.mkt_opens: dict[str, float] = {}   # sym → day open price
        self.mkt_ltps:  dict[str, float] = {}   # sym → last LTP

        self.executor  = AngelOrderExecutor(paper=paper)
        self.notifier  = Notifier()

    # ── Phase transitions (called from timer thread) ──────────────────────────

    def on_market_open(self) -> None:
        """9:15 — start building opening range."""
        logger.info("=== 9:15: BUILDING RANGE phase ===")
        with self._lock:
            self.phase = Phase.BUILDING
            self.orb_ranges = {sym: ORBRange(sym) for sym in self.premarket}

    def on_scan_start(self) -> None:
        """9:30 — start scanning for breakouts."""
        logger.info("=== 9:30: SCANNING phase ===")
        with self._lock:
            self.phase = Phase.SCANNING
        # Log ORB widths
        for sym, orb in sorted(self.orb_ranges.items()):
            if orb.high > 0 and orb.low < float("inf"):
                width = orb.high - orb.low
                logger.debug("ORB {}: {:.2f}–{:.2f} (width {:.2f})", sym, orb.low, orb.high, width)

    def on_scan_end(self) -> None:
        """10:00 — stop looking for new entries; just manage existing."""
        logger.info("=== 10:00: MANAGING only (no new entries) ===")
        with self._lock:
            self.phase = Phase.MANAGING

    def on_squareoff(self) -> None:
        """3:15 — force-exit all positions."""
        logger.info("=== 3:15: SQUAREOFF ===")
        with self._lock:
            self.phase = Phase.SQUAREOFF
        self._force_exit_all()

    def on_done(self) -> None:
        """3:30 — shutdown."""
        logger.info("=== 3:30: DONE — shutting down ===")
        self.phase = Phase.DONE
        self._send_eod_summary()

    # ── Tick handler (called from WebSocket thread) ───────────────────────────

    def on_tick(self, tick: dict) -> None:
        sym      = tick["symbol"]
        ltp      = tick["ltp"]
        vol      = tick["volume"]
        day_open = tick.get("open", 0.0)
        if ltp <= 0:
            return

        # Market proxy: capture opening price on first tick, track LTP always
        if day_open > 0 and sym not in self.mkt_opens:
            self.mkt_opens[sym] = day_open
        self.mkt_ltps[sym] = ltp

        with self._lock:
            phase = self.phase

        # Momentum smart observer runs continuously regardless of ORB phase
        self._update_momentum_watch(sym, ltp, vol, day_open)

        if phase == Phase.BUILDING:
            self._update_range(sym, ltp, vol)

        elif phase in (Phase.SCANNING, Phase.MANAGING):
            self._manage_position(sym, ltp)
            if phase == Phase.SCANNING:
                self._check_entry(sym, ltp, vol)

    # ── Opening range builder ─────────────────────────────────────────────────

    def _update_range(self, sym: str, ltp: float, vol: float) -> None:
        orb = self.orb_ranges.get(sym)
        if not orb:
            return
        if ltp > orb.high:
            orb.high = ltp
        if ltp < orb.low:
            orb.low = ltp
        # capture end-of-9:15-bar volume (last tick before 9:30 is the best proxy)
        orb.vol_9_15 = vol

    # ── Momentum smart observer ────────────────────────────────────────────────

    def _market_intraday_change(self) -> float:
        """
        Average % change of all subscribed stocks from their day open.
        Used as a Nifty proxy to distinguish market-wide crash from stock-specific drop.
        Only uses stocks that have received their opening price tick.
        """
        changes = []
        for sym, ltp in self.mkt_ltps.items():
            op = self.mkt_opens.get(sym, 0.0)
            if op > 0:
                changes.append((ltp - op) / op)
        return sum(changes) / len(changes) if changes else 0.0

    def _volume_surge_ratio(self, vol: float, vol_avg20: float) -> float:
        """
        Compare cumulative intraday volume against pro-rated daily average.
        At 10 AM (45 min elapsed of 390 min trading day): ratio > 1.5 means
        the stock is being sold 1.5× faster than its typical pace.
        """
        if vol_avg20 <= 0:
            return 0.0
        now     = datetime.now(IST)
        elapsed = (now.hour * 60 + now.minute) - (9 * 60 + 15)   # minutes since open
        if elapsed <= 0:
            return 0.0
        prorated = vol_avg20 * (elapsed / 390.0)   # 390 = 6.5h trading day in minutes
        return vol / prorated if prorated > 0 else 0.0

    def _update_momentum_watch(self, sym: str, ltp: float, vol: float, day_open: float) -> None:
        watch = self.mom_by_sym.get(sym)
        if watch is None or watch.exit_fired:
            return

        # Capture opening price
        if watch.open_price <= 0 and day_open > 0:
            watch.open_price = day_open

        watch.ltp = ltp
        if ltp > watch.day_high:
            watch.day_high = ltp
        if ltp < watch.day_low:
            watch.day_low = ltp

        # Informational alert: simple -8% drop from prev close (for human awareness)
        if not watch.alert_fired and watch.prev_close > 0:
            drop_from_prev = (watch.prev_close - ltp) / watch.prev_close
            if drop_from_prev >= MOM_ALERT_DROP:
                watch.alert_fired = True
                logger.warning("MOM ALERT: {} -{:.1f}% from prev close → ₹{:.2f}",
                               sym, drop_from_prev * 100, ltp)
                self._notify_mom_alert(watch, drop_from_prev)

        # Smart exit logic — only after market has opened properly (9:20 AM+)
        now = datetime.now(IST)
        if now.hour == 9 and now.minute < 20:
            return
        self._check_smart_exit(watch, ltp, vol)

    def _check_smart_exit(self, watch: MomentumWatch, ltp: float, vol: float) -> None:
        """
        Three-signal smart exit:
          Hard floor : LTP < entry × 0.85 → exit immediately, no analysis
          Signal 2   : stock underperforms market by >5% AND elevated volume (1.5× pace)
          Signal 3   : LTP < EMA20 today AND prev day also closed below EMA20
          Market guard: if Nifty proxy is down >2%, skip Signal-2+3 exit (systemic crash)
          EXIT when: hard_floor  OR  (Signal2 AND Signal3 AND NOT market_crash)
        """
        if watch.exit_fired:
            return

        # ── Hard floor: -15% from entry, unconditional ────────────────────────
        if watch.hard_stop > 0 and ltp <= watch.hard_stop:
            logger.warning("MOM HARD STOP: {} @ ₹{:.2f} (entry ₹{:.2f}, stop ₹{:.2f})",
                           watch.symbol, ltp, watch.entry_price, watch.hard_stop)
            self._execute_momentum_exit(watch, ltp, "hard_stop_-15pct")
            return

        # ── Market crash guard ────────────────────────────────────────────────
        mkt_chg = self._market_intraday_change()
        if mkt_chg <= -MOM_MARKET_CRASH:
            logger.debug("MOM: {} market-wide drop {:.1f}% — skipping smart exit",
                         watch.symbol, mkt_chg * 100)
            return

        # ── Signal 2: stock underperforming market by >5% with volume surge ──
        ref_px = watch.open_price if watch.open_price > 0 else watch.prev_close
        if ref_px <= 0:
            return
        stock_chg  = (ltp - ref_px) / ref_px
        underperf  = stock_chg - mkt_chg           # negative = lagging market
        vol_ratio  = self._volume_surge_ratio(vol, watch.vol_avg20)
        signal2    = (underperf <= -MOM_UNDERPERF) and (vol_ratio >= MOM_VOL_SURGE)

        # ── Signal 3: LTP below EMA20 today AND was below yesterday too ──────
        signal3 = (watch.ema20 > 0 and ltp < watch.ema20 and watch.prev_day_below_ema)

        if signal2 and signal3:
            logger.warning(
                "MOM SMART EXIT: {} | underperf {:.1f}% vs mkt | vol {:.1f}× | "
                "below EMA20 ₹{:.2f} (2 days) | ltp ₹{:.2f}",
                watch.symbol, underperf * 100, vol_ratio, watch.ema20, ltp,
            )
            self._execute_momentum_exit(watch, ltp,
                f"smart_exit|underperf={underperf*100:.1f}%|vol={vol_ratio:.1f}x|ema_break")

    def _execute_momentum_exit(self, watch: MomentumWatch, price: float, reason: str) -> None:
        """Place CNC SELL, log to DB, notify Telegram."""
        with self._lock:
            if watch.exit_fired:   # double-check under lock
                return
            watch.exit_fired = True

        pnl = (price - watch.entry_price) * watch.approx_qty
        logger.warning("MOM EXIT [{}]: {} qty={} @ ₹{:.2f} → pnl ₹{:+,.0f}  ({})",
                       "PAPER" if self.paper else "LIVE",
                       watch.symbol, watch.approx_qty, price, pnl, reason)

        # Place SELL order
        tok = watch.token
        self.executor.place_market_order(watch.symbol + "-EQ", tok, watch.approx_qty, "SELL")

        # Log to DB
        try:
            with store.db_conn() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS momentum_exits (
                        exit_date    DATE,
                        symbol       VARCHAR,
                        entry_price  DOUBLE,
                        exit_price   DOUBLE,
                        qty          INTEGER,
                        pnl          DOUBLE,
                        reason       VARCHAR,
                        paper        BOOLEAN,
                        logged_at    TIMESTAMP
                    )
                """)
                conn.execute("""
                    INSERT INTO momentum_exits VALUES (?,?,?,?,?,?,?,?,?)
                """, [date.today(), watch.symbol, watch.entry_price, price,
                      watch.approx_qty, pnl, reason, self.paper,
                      datetime.now(IST).replace(tzinfo=None)])
        except Exception as exc:
            logger.warning("DB log for momentum exit failed: {}", exc)

        # Notify
        self._notify_momentum_exit(watch, price, pnl, reason)

    def _notify_mom_alert(self, watch: MomentumWatch, drop: float) -> None:
        if self.no_notify:
            return
        approx_loss = watch.approx_qty * watch.prev_close * drop
        msg = (
            f"⚠️ *Momentum Watch — {watch.symbol}*\n"
            f"Down *{drop*100:.1f}%* from prev close → ₹{watch.ltp:.2f}\n"
            f"Weight: {watch.weight*100:.1f}%  |  Est. impact: ₹{approx_loss:,.0f}\n"
            f"Monitoring for smart exit signal..."
        )
        self.notifier.send_telegram(msg)

    def _notify_momentum_exit(self, watch: MomentumWatch, price: float, pnl: float, reason: str) -> None:
        if self.no_notify:
            return
        mode = "PAPER" if self.paper else "LIVE"
        sign = "✅" if pnl >= 0 else "🛑"
        reason_clean = reason.split("|")[0].replace("_", " ")
        msg = (
            f"{sign} *Momentum EXIT ({mode}) — {watch.symbol}*\n"
            f"Sold {watch.approx_qty} qty @ ₹{price:.2f}  |  Entry ₹{watch.entry_price:.2f}\n"
            f"P&L: ₹{pnl:+,.0f}  ({(price/watch.entry_price-1)*100:+.1f}%)\n"
            f"Reason: {reason_clean}\n"
            f"_Hold remaining {len([w for w in self.mom_by_sym.values() if not w.exit_fired])-1} stocks_"
        )
        self.notifier.send_telegram(msg)

    def _momentum_eod_summary(self) -> str:
        """Returns a formatted string of today's momentum portfolio performance."""
        if not self.mom_by_sym:
            return ""
        lines = ["\n*Momentum Portfolio (CNC) — Today's Performance:*"]
        total_pnl = 0.0
        exits_today = 0
        for sym, w in sorted(self.mom_by_sym.items()):
            ref_px = w.entry_price if w.entry_price > 0 else w.prev_close
            cur_px = w.ltp if w.ltp > 0 else ref_px
            if ref_px <= 0:
                continue
            chg        = (cur_px - ref_px) / ref_px
            approx_pnl = w.approx_qty * (cur_px - ref_px)
            total_pnl += approx_pnl
            icon  = "🟢" if chg >= 0 else "🔴"
            tags  = []
            if w.exit_fired: tags.append("EXITED"); exits_today += 1
            if w.cross_signal: tags.append("ORB✓")
            tag_str = f"  [{' '.join(tags)}]" if tags else ""
            lines.append(f"  {icon} {sym}: {chg*100:+.1f}%  ₹{approx_pnl:+,.0f}{tag_str}")
        lines.append(f"  ─────────────────────────")
        lines.append(f"  Portfolio est. P&L today: ₹{total_pnl:+,.0f}")
        if exits_today:
            lines.append(f"  Smart exits triggered: {exits_today}")
        return "\n".join(lines)

    # ── Entry logic ───────────────────────────────────────────────────────────

    def _check_entry(self, sym: str, ltp: float, vol: float) -> None:
        if sym in self.fired_syms:
            return
        if MAX_TRADES > 0 and len(self.positions) >= MAX_TRADES:
            return

        orb = self.orb_ranges.get(sym)
        if not orb or orb.high <= 0 or orb.low >= float("inf"):
            return

        orb_range = orb.high - orb.low
        if orb_range <= 0:
            return

        # Breakout: LTP must clear ORB high
        if ltp <= orb.high:
            return

        # Volume surge: current bar volume vs yesterday's 20-day avg
        ref = self.premarket.get(sym, {})
        vol_avg = ref.get("vol_avg20", 0)
        if vol_avg > 0 and vol < VOL_SURGE * vol_avg / 6.5:
            # Rough: daily avg ÷ 6.5 trading hours → per-hour avg
            # The 9:15 bar captures ~15min of the day, so we scale accordingly
            return

        # Quality: yesterday's close must be above EMA20
        ema20 = ref.get("ema20", 0)
        last_close = ref.get("last_close", 0)
        if ema20 > 0 and last_close < ema20:
            return

        # Size the position
        stop_dist = STOP_FRAC * orb_range
        if stop_dist <= 0:
            return
        risk_amount = CAPITAL * RISK_PCT
        qty = max(1, int(risk_amount / stop_dist))

        stop   = round(ltp - stop_dist, 2)
        target = round(ltp + TARGET_MULT * orb_range, 2)

        is_mom = sym in self.mom_by_sym
        if is_mom:
            self.mom_by_sym[sym].cross_signal = True

        logger.info("ENTRY SIGNAL{}: {} @ {:.2f} | ORB {:.2f}–{:.2f} | stop {:.2f} | target {:.2f} | qty {}",
                    " [MOM✓]" if is_mom else "", sym, ltp, orb.low, orb.high, stop, target, qty)

        token    = self.sym_to_token.get(sym, "")
        order_id = self.executor.place_market_order(sym + "-EQ", token, qty, "BUY")

        pos = Position(
            symbol=sym, token=token, entry=ltp, qty=qty,
            orb_range=orb_range, stop=stop, target=target,
            trail_lv=0, peak=ltp, order_id=order_id, paper=self.paper,
        )
        with self._lock:
            self.positions[sym] = pos
            self.fired_syms.add(sym)

        self._notify_entry(pos)

    # ── Trailing stop management ──────────────────────────────────────────────

    def _manage_position(self, sym: str, ltp: float) -> None:
        with self._lock:
            pos = self.positions.get(sym)
            if pos is None:
                return

        # Update peak
        if ltp > pos.peak:
            pos.peak = ltp

        gain = ltp - pos.entry
        orb  = pos.orb_range

        # Hard target
        if ltp >= pos.target:
            logger.info("TARGET HIT: {} @ {:.2f} (entry {:.2f}, +{:.1f}×)", sym, ltp, pos.entry, TARGET_MULT)
            self._exit_position(sym, ltp, "target")
            return

        # Trail level 2: when +1.0×, lock in 0.5× gain
        if gain >= orb * TRAIL2_FRAC and pos.trail_lv < 2:
            new_stop = round(pos.entry + 0.5 * orb, 2)
            if new_stop > pos.stop:
                pos.stop    = new_stop
                pos.trail_lv = 2
                logger.info("TRAIL LV2: {} stop → {:.2f} (locked +{:.1f}×)", sym, new_stop, 0.5)

        # Trail level 1: when +0.5×, move to breakeven
        elif gain >= orb * TRAIL1_FRAC and pos.trail_lv < 1:
            pos.stop    = round(pos.entry, 2)
            pos.trail_lv = 1
            logger.info("TRAIL LV1: {} stop → breakeven {:.2f}", sym, pos.stop)

        # Check stop
        if ltp <= pos.stop:
            reason = "trail_stop" if pos.trail_lv > 0 else "stop"
            logger.info("STOP HIT ({}): {} @ {:.2f} (entry {:.2f}, stop was {:.2f})",
                        reason, sym, ltp, pos.entry, pos.stop)
            self._exit_position(sym, ltp, reason)

    def _exit_position(self, sym: str, price: float, reason: str) -> None:
        with self._lock:
            pos = self.positions.pop(sym, None)
        if pos is None:
            return
        pnl = (price - pos.entry) * pos.qty
        logger.info("EXIT {}: {} qty={} @ {:.2f} → pnl ₹{:+,.0f} ({})",
                    "PAPER" if pos.paper else "LIVE", sym, pos.qty, price, pnl, reason)
        self.executor.place_market_order(sym + "-EQ", pos.token, pos.qty, "SELL")
        self._log_trade(pos, price, pnl, reason)
        self._notify_exit(pos, price, pnl, reason)

    def _force_exit_all(self) -> None:
        with self._lock:
            syms = list(self.positions.keys())
        for sym in syms:
            with self._lock:
                pos = self.positions.get(sym)
            if pos:
                logger.info("SQUAREOFF: {} qty={}", sym, pos.qty)
                self._exit_position(sym, 0.0, "squareoff_3:15")

    # ── DB logging ────────────────────────────────────────────────────────────

    def _log_trade(self, pos: Position, exit_price: float, pnl: float, reason: str) -> None:
        try:
            with store.db_conn() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS live_orb_trades (
                        trade_id    VARCHAR,
                        trade_date  DATE,
                        symbol      VARCHAR,
                        entry       DOUBLE,
                        exit_price  DOUBLE,
                        qty         INTEGER,
                        pnl         DOUBLE,
                        reason      VARCHAR,
                        paper       BOOLEAN,
                        logged_at   TIMESTAMP
                    )
                """)
                conn.execute("""
                    INSERT INTO live_orb_trades VALUES (?,?,?,?,?,?,?,?,?,?)
                """, [pos.order_id, date.today(), pos.symbol, pos.entry, exit_price,
                      pos.qty, pnl, reason, pos.paper,
                      datetime.now(IST).replace(tzinfo=None)])
        except Exception as exc:
            logger.warning("DB log failed: {}", exc)

    # ── Notifications ─────────────────────────────────────────────────────────

    def _notify_entry(self, pos: Position) -> None:
        if self.no_notify:
            return
        mode = "PAPER" if pos.paper else "LIVE"
        msg  = (f"*Live ORB ENTRY ({mode}) — {date.today()}*\n"
                f"BUY {pos.symbol} qty={pos.qty} @ ₹{pos.entry:.2f}\n"
                f"Stop: ₹{pos.stop:.2f}  |  Target: ₹{pos.target:.2f}\n"
                f"ORB range: ₹{pos.orb_range:.2f}  |  Risk: ₹{(pos.entry-pos.stop)*pos.qty:,.0f}")
        self.notifier.send_telegram(msg)

    def _notify_exit(self, pos: Position, price: float, pnl: float, reason: str) -> None:
        if self.no_notify:
            return
        sign = "✅" if pnl >= 0 else "❌"
        msg  = (f"{sign} *Live ORB EXIT — {pos.symbol}*\n"
                f"Exit @ ₹{price:.2f}  |  Reason: {reason}\n"
                f"P&L: ₹{pnl:+,.0f}  |  Entry was ₹{pos.entry:.2f}")
        self.notifier.send_telegram(msg)

    def _send_eod_summary(self) -> None:
        if self.no_notify:
            return
        try:
            with store.db_conn() as conn:
                rows = conn.execute("""
                    SELECT symbol, pnl, reason FROM live_orb_trades
                    WHERE trade_date = CURRENT_DATE
                    ORDER BY pnl DESC
                """).fetchall()

            lines = [f"*Piedpiper EOD Summary — {date.today()}*\n"]

            if rows:
                total = sum(r[2] for r in rows)
                lines.append("*ORB Intraday Trades:*")
                for sym, pnl, reason in rows:
                    icon = "✅" if pnl >= 0 else "❌"
                    lines.append(f"  {icon} {sym}: ₹{pnl:+,.0f} ({reason})")
                lines.append(f"  *ORB Total: ₹{total:+,.0f}*")
            else:
                lines.append("ORB: No trades today")

            # Append momentum section
            mom_text = self._momentum_eod_summary()
            if mom_text:
                lines.append(mom_text)

            self.notifier.send_telegram("\n".join(lines))
        except Exception as exc:
            logger.warning("EOD summary failed: {}", exc)


# ── Timer scheduler ───────────────────────────────────────────────────────────

def _schedule_at(target_time_str: str, callback, tz=IST) -> threading.Timer:
    """Schedule callback at a specific HH:MM IST time today. Returns timer."""
    now = datetime.now(tz)
    h, m = map(int, target_time_str.split(":"))
    target = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if target <= now:
        delay = 0.0
    else:
        delay = (target - now).total_seconds()
    t = threading.Timer(delay, callback)
    t.daemon = True
    t.start()
    logger.info("Scheduled {} at {} IST (in {:.0f}s)", callback.__name__, target_time_str, delay)
    return t


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Live ORB Trading Engine")
    ap.add_argument("--live",      action="store_true", help="Place real orders (default: paper)")
    ap.add_argument("--no-notify", action="store_true", help="Suppress Telegram")
    ap.add_argument("--test",      action="store_true", help="Pre-market check only — no WebSocket")
    args = ap.parse_args()

    paper = not args.live
    today = date.today()

    logger.info("=" * 60)
    logger.info("Piedpiper Live ORB — {} | {}", today, "PAPER" if paper else "LIVE")
    logger.info("Capital ₹{:,.0f} | Risk {}% | Max {} positions", CAPITAL, RISK_PCT*100, MAX_TRADES)
    logger.info("=" * 60)

    if not is_trading_day(today):
        logger.info("{} is not a trading day — exiting", today)
        return

    if not nifty_is_bull():
        logger.info("BEAR regime — no ORB LONG today. Exit.")
        if not args.no_notify:
            Notifier().send_telegram(
                f"*Live ORB {today}*\nBear regime — no trades today. Nifty < EMA20."
            )
        return

    store.init_schema()

    # Load universe + tokens
    universe_df = fetch_nifty200()
    symbols     = universe_df["symbol"].tolist() if not universe_df.empty else []
    if not symbols:
        logger.error("Empty universe — check ind_nifty200list.csv")
        sys.exit(1)

    sym_to_tok, tok_to_sym = build_token_map(symbols)
    premarket = load_premarket_data(list(sym_to_tok.keys()))

    # Load momentum portfolio (may be empty if currently in defensive mode)
    mom_watches = load_momentum_portfolio(sym_to_tok, premarket)

    if args.test:
        logger.info("TEST mode — pre-market check passed. {} ORB symbols ready.", len(premarket))
        logger.info("Sample ORB: {}", {k: v for k, v in list(premarket.items())[:3]})
        if mom_watches:
            logger.info("Momentum watching: {}", [w.symbol for w in mom_watches])
        else:
            logger.info("Momentum: currently defensive/cash — no holdings to watch")
        return

    # Build engine
    engine = LiveORBEngine(paper=paper, no_notify=args.no_notify)
    engine.premarket    = premarket
    engine.sym_to_token = sym_to_tok
    engine.orb_ranges   = {sym: ORBRange(sym) for sym in premarket}

    # Wire in momentum watches
    for w in mom_watches:
        engine.mom_watches[w.token] = w
        engine.mom_by_sym[w.symbol] = w

    if mom_watches:
        logger.info("Momentum observer: {} holdings | alert threshold >{:.0f}% drop",
                    len(mom_watches), MOM_ALERT_DROP * 100)
        if not args.no_notify:
            syms_str = ", ".join(w.symbol for w in mom_watches)
            Notifier().send_telegram(
                f"*Piedpiper started — {date.today()}*\n"
                f"Mode: {'PAPER' if paper else 'LIVE'}\n"
                f"ORB: Nifty 200 | ₹{CAPITAL:,.0f} | {MAX_TRADES} max trades\n"
                f"Momentum watching ({len(mom_watches)}): {syms_str}"
            )
    else:
        logger.info("Momentum observer: defensive/cash mode — no watch active")
        if not args.no_notify:
            Notifier().send_telegram(
                f"*Piedpiper started — {date.today()}*\n"
                f"Mode: {'PAPER' if paper else 'LIVE'} | Momentum: DEFENSIVE (cash)\n"
                f"ORB scanning Nifty 200 on BULL days."
            )

    # Schedule phase transitions
    _schedule_at("09:15", engine.on_market_open)
    _schedule_at("09:30", engine.on_scan_start)
    _schedule_at("10:00", engine.on_scan_end)
    _schedule_at("15:15", engine.on_squareoff)
    _schedule_at("15:30", engine.on_done)

    # Merge token maps: ORB universe + any momentum-only tokens
    # (momentum stocks are usually a subset of Nifty 200, but ensure they're all subscribed)
    merged_tok_to_sym = {**tok_to_sym}
    extra_mom_tokens  = []
    for w in mom_watches:
        if w.token not in merged_tok_to_sym:
            merged_tok_to_sym[w.token] = w.symbol
            extra_mom_tokens.append(w.token)
    if extra_mom_tokens:
        logger.info("Subscribing {} extra momentum tokens not in Nifty 200", len(extra_mom_tokens))

    all_tokens = list(merged_tok_to_sym.keys())

    # Build feed
    feed = LiveFeed(
        on_tick    = engine.on_tick,
        on_connect = lambda: logger.info(
            "Feed connected — streaming {} tokens ({} ORB + {} momentum-only)",
            len(all_tokens), len(sym_to_tok), len(extra_mom_tokens)),
    )
    feed.set_symbol_map(merged_tok_to_sym)
    feed.subscribe(all_tokens)

    # Block here until 3:30 PM
    logger.info("Starting live feed. Will run until 3:30 PM IST ...")
    while engine.phase != Phase.DONE:
        feed.start()          # blocks until disconnect/error
        if engine.phase == Phase.DONE:
            break
        logger.info("Feed disconnected — reconnecting ...")
        time.sleep(3)

    logger.info("Live ORB engine shut down.")


if __name__ == "__main__":
    main()
