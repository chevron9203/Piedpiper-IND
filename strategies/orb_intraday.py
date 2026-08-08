"""
Opening Range Breakout (ORB) — intraday strategy for NSE equities.

Rules:
  Opening range = the 9:15-9:30 candle (first 15-min bar of the day).

  LONG signal (bull regime: Nifty ≥ EMA20):
    - ORB volume surge (opening bar vol > 3x 20-day avg of 9:15 bars)
    - Stock must be above its own 20-day EMA (trend alignment)
    - Range 0.3–2.5% but NOT 1–1.5% (worst bucket, ~20% win rate)
    - Min stock price Rs 100 (no penny stocks)
    - Entry cutoff 10:00 AM
    - Entry: range_high + 0.05  |  Stop: entry - 0.4×range  |  Target: entry + 1.5×range
    - 2-tier trailing: at 50% to target → stop to entry; at 75% → lock half gain

  SHORT signal (bear regime: Nifty < EMA20):
    - Same vol surge and range filters
    - Stock must be BELOW its own 20-day EMA
    - Entry: range_low - 0.05  |  Stop: entry + 0.4×range  |  Target: entry - 1.5×range
    - Same 2-tier trailing (inverted)
    - Executed as equity MIS short-sell (SELL to open, BUY to close same day)

  Position sizing: 3% risk of intraday capital per trade. No cap on trades/day — filters do the work.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
import numpy as np
import pandas as pd
from loguru import logger

# Intraday session boundaries (IST)
SESSION_OPEN  = time(9, 15)
ORB_END       = time(9, 30)    # Opening range = first 15-min bar
SQUAREOFF     = time(15, 15)   # All positions closed by 3:15 PM

# Strategy parameters
ORB_MIN_CANDLES    = 1
MAX_RANGE_PCT      = 0.025   # skip if opening range > 2.5%
MIN_RANGE_PCT      = 0.003   # skip if range < 0.3%
SKIP_MID_RANGE_LO  = 0.010   # skip 1–1.5% range: ~20% win rate bucket
SKIP_MID_RANGE_HI  = 0.015
VOL_SURGE_MULT     = 3.0     # vs 20-day avg of 9:15 bars only (not all-bar avg)
STOP_FRAC          = 0.4
RR_MIN             = 1.5
TARGET_MULT        = 1.5
RISK_PCT           = 0.03
MIN_STOCK_PRICE    = 100.0   # skip penny stocks
ENTRY_CUTOFF       = time(10, 0)  # entries after 10:00 have negative avg P&L


@dataclass
class OrbSignal:
    symbol:       str
    token:        str
    trade_date:   date
    direction:    str          # "LONG" or "SHORT"
    entry_price:  float
    stop_loss:    float
    target:       float
    range_width:  float
    range_pct:    float
    vol_ratio:    float
    nifty_trend:  str          # "UP" | "DOWN" | "FLAT"

    def position_size(self, capital: float) -> int:
        """Shares to trade given 1% risk rule."""
        stop_dist = abs(self.entry_price - self.stop_loss)
        if stop_dist <= 0:
            return 0
        return max(1, int((capital * RISK_PCT) / stop_dist))

    def to_dict(self) -> dict:
        return {
            "symbol":      self.symbol,
            "token":       self.token,
            "trade_date":  str(self.trade_date),
            "direction":   self.direction,
            "entry_price": round(self.entry_price, 2),
            "stop_loss":   round(self.stop_loss, 2),
            "target":      round(self.target, 2),
            "range_width": round(self.range_width, 2),
            "range_pct":   round(self.range_pct * 100, 2),
            "vol_ratio":   round(self.vol_ratio, 2),
            "nifty_trend": self.nifty_trend,
        }


def compute_orb_signals(
    candles_15m:        dict[str, pd.DataFrame],    # symbol → DataFrame of 15-min bars today
    token_map:          dict[str, str],              # symbol → Angel One token
    nifty_candles_15m:  pd.DataFrame | None,         # Nifty 15-min bars today for trend filter
    avg_volumes:        dict[str, float],            # symbol → avg 15-min opening volume (20-day)
    trade_date:         date | None = None,
    direction:          str = "LONG",               # "LONG" (bull) or "SHORT" (bear)
    stock_ema20:        dict[str, float] | None = None,  # {clean_symbol: ema20_value} from prev day
    prev_closes:        dict[str, float] | None = None,  # {clean_symbol: yesterday's close}
    gap_filter_pct:     float = 0.0,                # >0 to require gap-up (LONG) or gap-down (SHORT)
    fno_map:            dict | None = None,          # unused — kept for caller compat
) -> list[OrbSignal]:
    """
    Compute ORB signals from today's 15-min bars.

    Args:
        candles_15m:   {symbol: df} with OHLCV columns, index=datetime
        token_map:     {symbol: token} for order placement
        nifty_candles_15m: today's Nifty 15-min bars
        avg_volumes:   20-day avg opening-bar volume per symbol
        trade_date:    today's date
        direction:     "LONG" or "SHORT"
        stock_ema20:   {clean_symbol: ema20_val} — stock's own 20d EMA from prev close.
                       Only LONG stocks above their EMA; only SHORT stocks below it.
                       Pass None to skip this filter.

    Returns:
        List of OrbSignal sorted by vol_ratio descending.
    """
    if trade_date is None:
        from datetime import date as _date
        trade_date = _date.today()

    # Nifty trend: UP if Nifty opening bar closed above open
    nifty_trend = "FLAT"
    if nifty_candles_15m is not None and not nifty_candles_15m.empty:
        first = nifty_candles_15m.iloc[0]
        if first["close"] > first["open"] * 1.001:
            nifty_trend = "UP"
        elif first["close"] < first["open"] * 0.999:
            nifty_trend = "DOWN"

    signals: list[OrbSignal] = []

    for sym, df in candles_15m.items():
        if df.empty or len(df) < 1:
            continue

        orb        = df.iloc[0]
        range_high = float(orb["high"])
        range_low  = float(orb["low"])
        range_open = float(orb["open"])
        range_vol  = float(orb["volume"]) if "volume" in df.columns else 0

        range_width = range_high - range_low
        range_pct   = range_width / range_open if range_open > 0 else 0

        if range_open < MIN_STOCK_PRICE:
            continue
        if range_pct > MAX_RANGE_PCT or range_pct < MIN_RANGE_PCT:
            continue
        if SKIP_MID_RANGE_LO <= range_pct < SKIP_MID_RANGE_HI:
            continue

        avg_vol   = avg_volumes.get(sym) or avg_volumes.get(sym.replace("-EQ", ""), 0)
        if avg_vol <= 0:
            continue
        vol_ratio = range_vol / avg_vol
        if vol_ratio < VOL_SURGE_MULT:
            continue

        tok = token_map.get(sym) or token_map.get(sym.replace("-EQ", ""))
        if not tok:
            continue

        clean = sym.replace("-EQ", "")

        # Stock EMA20 filter — trend alignment
        # Use yesterday's close (not today's open) to match backtest logic and avoid
        # false positives on gap-up/down days where open crosses EMA intraday.
        if stock_ema20 is not None:
            ema_val  = stock_ema20.get(clean, 0.0)
            ref_px   = (prev_closes.get(clean, 0.0) if prev_closes else 0.0) or range_open
            if ema_val > 0 and ref_px > 0:
                if direction == "LONG" and ref_px < ema_val:
                    continue   # stock below own EMA, skip LONG
                if direction == "SHORT" and ref_px > ema_val:
                    continue   # stock above own EMA, skip SHORT

        # Gap filter — event-driven quality filter
        # LONG: today must have gapped up ≥ gap_filter_pct vs yesterday's close
        if gap_filter_pct > 0 and prev_closes is not None:
            prev_c = prev_closes.get(clean, 0.0)
            if prev_c > 0:
                if direction == "LONG" and range_open < prev_c * (1 + gap_filter_pct):
                    continue
                if direction == "SHORT" and range_open > prev_c * (1 - gap_filter_pct):
                    continue

        if direction == "LONG":
            if nifty_trend == "DOWN":
                continue
            entry  = range_high + 0.05
            stop   = entry - STOP_FRAC * range_width
            target = entry + TARGET_MULT * range_width
            risk   = entry - stop
            if risk <= 0 or (target - entry) / risk < RR_MIN:
                continue
            signals.append(OrbSignal(
                symbol=sym, token=tok, trade_date=trade_date,
                direction="LONG", entry_price=entry, stop_loss=stop,
                target=target, range_width=range_width, range_pct=range_pct,
                vol_ratio=vol_ratio, nifty_trend=nifty_trend,
            ))

        else:  # SHORT — bear regime breakdown below opening range low
            if nifty_trend == "UP":
                continue
            entry  = range_low - 0.05
            stop   = entry + STOP_FRAC * range_width
            target = entry - TARGET_MULT * range_width
            risk   = stop - entry
            if risk <= 0 or (entry - target) / risk < RR_MIN:
                continue
            signals.append(OrbSignal(
                symbol=sym, token=tok, trade_date=trade_date,
                direction="SHORT", entry_price=entry, stop_loss=stop,
                target=target, range_width=range_width, range_pct=range_pct,
                vol_ratio=vol_ratio, nifty_trend=nifty_trend,
            ))

    signals.sort(key=lambda s: s.vol_ratio, reverse=True)
    logger.info("ORB signals for {}: {} (dir={} nifty={})", trade_date, len(signals), direction, nifty_trend)
    return signals


def backtest_orb_on_candles(
    symbol:           str,
    candles_15m:      pd.DataFrame,
    avg_vol_window:   int   = 20,
    risk_pct:         float = RISK_PCT,
    capital:          float = 100_000.0,
    nifty_close:      pd.Series | None = None,
    nifty_ema20:      pd.Series | None = None,
    gap_filter_pct:   float = 0.0,     # set >0 to require gap-up (LONG) or gap-down (SHORT)
    trail2:           bool  = True,
    direction:        str   = "LONG",  # "LONG" or "SHORT"
    stock_ema_filter: bool  = True,    # only trade in direction of stock's own EMA20
) -> pd.DataFrame:
    """
    Simulate ORB on historical 15-min bars for one symbol.

    LONG  — entry above range high; only on bull regime days (Nifty >= EMA20).
    SHORT — entry below range low;  only on bear regime days (Nifty < EMA20).
             Executed as equity MIS short (same charges as LONG).

    stock_ema_filter:
      LONG:  skip if prev-day close < stock EMA20
      SHORT: skip if prev-day close > stock EMA20

    gap_filter_pct (optional):
      LONG:  skip if today_open < prev_close × (1 + gap_filter_pct)
      SHORT: skip if today_open > prev_close × (1 - gap_filter_pct)
    """
    if candles_15m.empty:
        return pd.DataFrame()

    is_short = direction == "SHORT"

    df = candles_15m.copy()
    df.index = pd.to_datetime(df.index)
    df["trade_date"] = df.index.date
    days = sorted(df["trade_date"].unique())
    # Pre-build day→index lookup to avoid O(n²) list.index() inside the loop
    day_to_idx: dict = {d: i for i, d in enumerate(days)}

    # Vol avg: ONLY 9:15 opening bars — matches live _batch_avg_volumes
    # .shift(1) ensures today's bar is excluded from its own average (no look-ahead)
    orb_mask    = pd.to_datetime(df.index).time == pd.Timestamp("09:15").time()
    orb_vol_ser = (df.loc[orb_mask, "volume"]
                   .rolling(avg_vol_window, min_periods=avg_vol_window // 2)
                   .mean()
                   .shift(1))
    df["vol_avg"] = np.nan
    df.loc[orb_mask, "vol_avg"] = orb_vol_ser.values
    df["vol_avg"] = df["vol_avg"].ffill()

    # Daily closes (last bar of each day) — for stock EMA20 and gap filter
    daily_close_by_day: dict = {}
    for day in days:
        day_df = df[df["trade_date"] == day]
        if not day_df.empty:
            daily_close_by_day[day] = float(day_df.iloc[-1]["close"])

    # Stock EMA20 on daily closes (running, so future data never leaks)
    stock_daily_ema20: dict = {}
    if stock_ema_filter and len(daily_close_by_day) >= 2:
        close_ser = pd.Series(daily_close_by_day).sort_index()
        ema_ser   = close_ser.ewm(span=20, adjust=False).mean()
        stock_daily_ema20 = ema_ser.to_dict()

    # Prev-close lookup for gap filter and EMA reference
    prev_close_by_day: dict = {}
    for i, day in enumerate(days):
        if i == 0:
            continue
        prev = daily_close_by_day.get(days[i - 1])
        if prev is not None:
            prev_close_by_day[day] = prev

    trades: list[dict] = []

    for day in days:
        day_df = df[df["trade_date"] == day].copy()
        if len(day_df) < 2:
            continue

        orb = day_df.iloc[0]
        range_high  = float(orb["high"])
        range_low   = float(orb["low"])
        range_open  = float(orb["open"])
        range_width = range_high - range_low
        range_pct   = range_width / range_open if range_open > 0 else 0
        orb_vol     = float(orb["volume"])
        avg_vol     = float(orb.get("vol_avg", 0)) or orb_vol
        vol_ratio   = orb_vol / avg_vol if avg_vol > 0 else 0

        if range_open < MIN_STOCK_PRICE:
            continue
        if range_pct > MAX_RANGE_PCT or range_pct < MIN_RANGE_PCT:
            continue
        if SKIP_MID_RANGE_LO <= range_pct < SKIP_MID_RANGE_HI:
            continue
        if vol_ratio < VOL_SURGE_MULT:
            continue

        # Stock EMA20 filter — use PREVIOUS day's close vs EMA20 (no look-ahead)
        day_idx = day_to_idx[day]
        if stock_ema_filter and stock_daily_ema20 and day_idx > 0:
            prev_day   = days[day_idx - 1]
            prev_close = daily_close_by_day.get(prev_day, float("nan"))
            prev_ema   = stock_daily_ema20.get(prev_day, float("nan"))
            if not (pd.isna(prev_close) or pd.isna(prev_ema) or prev_ema == 0):
                if is_short and prev_close > prev_ema:
                    continue   # stock above EMA → don't short
                if not is_short and prev_close < prev_ema:
                    continue   # stock below EMA → don't long

        # Optional gap filter
        if gap_filter_pct > 0 and day in prev_close_by_day:
            prev_c = prev_close_by_day[day]
            if prev_c > 0:
                if is_short and range_open > prev_c * (1 - gap_filter_pct):
                    continue
                elif not is_short and range_open < prev_c * (1 + gap_filter_pct):
                    continue

        # Nifty regime + trend filters (both use previous-day data — no look-ahead)
        # Live: DB has only yesterday's close at 9:31 AM, so backtest must match.
        nifty_trend = "FLAT"
        if nifty_close is not None:
            day_ts     = pd.Timestamp(day)
            prev_dates = nifty_close.index[nifty_close.index < day_ts]

            # Regime: LONG on bull days (Nifty ≥ EMA20), SHORT on bear days.
            # Use YESTERDAY's close vs EMA20 to match live _check_regime() which
            # reads the most-recent EOD row (= yesterday when run at 9:31 AM).
            if nifty_ema20 is not None and len(prev_dates) >= 1:
                prev_day = prev_dates[-1]
                ema_val  = nifty_ema20.get(prev_day, float("nan"))
                nifty_d  = nifty_close.get(prev_day, float("nan"))
                if not pd.isna(ema_val) and not pd.isna(nifty_d):
                    if is_short and nifty_d >= ema_val:
                        continue   # bull regime — no shorts
                    if not is_short and nifty_d < ema_val:
                        continue   # bear regime — no longs

            # Directional trend: use yesterday vs day-before return.
            if len(prev_dates) >= 2:
                chg = (float(nifty_close.loc[prev_dates[-1]]) -
                       float(nifty_close.loc[prev_dates[-2]])) / float(nifty_close.loc[prev_dates[-2]])
                if not pd.isna(chg):
                    nifty_trend = "UP" if chg > 0.001 else ("DOWN" if chg < -0.001 else "FLAT")

        if is_short and nifty_trend == "UP":
            continue
        if not is_short and nifty_trend == "DOWN":
            continue

        # Entry / stop / target
        if is_short:
            entry  = range_low - 0.05
            stop   = entry + STOP_FRAC * range_width
            target = entry - TARGET_MULT * range_width
            risk   = stop - entry
        else:
            entry  = range_high + 0.05
            stop   = entry - STOP_FRAC * range_width
            target = entry + TARGET_MULT * range_width
            risk   = entry - stop

        if risk <= 0 or abs(target - entry) / risk < RR_MIN:
            continue

        qty        = max(1, int((capital * risk_pct) / risk))
        trail_stop = stop

        filled      = False
        entry_price = np.nan
        entry_time  = None
        exit_price  = np.nan
        exit_time   = None
        exit_reason = "squareoff"

        for bar_ts, bar in day_df.iloc[1:].iterrows():
            bar_time = bar_ts.time()

            if bar_time >= SQUAREOFF:
                if filled:
                    exit_price, exit_time, exit_reason = float(bar["open"]), bar_ts, "squareoff"
                break

            bar_low  = float(bar["low"])
            bar_high = float(bar["high"])

            if not filled:
                if bar_time >= ENTRY_CUTOFF:
                    break
                if (not is_short and bar_high >= entry) or (is_short and bar_low <= entry):
                    filled, entry_price, entry_time = True, entry, bar_ts
                    trail_stop = stop
            else:
                if is_short:
                    if bar_high >= trail_stop:
                        exit_price, exit_time, exit_reason = trail_stop, bar_ts, "stop"; break
                    if bar_low <= target:
                        exit_price, exit_time, exit_reason = target, bar_ts, "target"; break
                    if trail2:
                        gain_range = entry_price - target
                        if gain_range > 0:
                            progress   = (entry_price - bar_low) / gain_range
                            tier2_stop = entry_price - 0.5 * gain_range
                            if progress >= 0.75 and trail_stop > tier2_stop:
                                trail_stop = tier2_stop
                            elif progress >= 0.50 and trail_stop > entry_price:
                                trail_stop = entry_price
                else:
                    if bar_low <= trail_stop:
                        exit_price, exit_time, exit_reason = trail_stop, bar_ts, "stop"; break
                    if bar_high >= target:
                        exit_price, exit_time, exit_reason = target, bar_ts, "target"; break
                    if trail2:
                        gain_range = target - entry_price
                        if gain_range > 0:
                            progress   = (bar_high - entry_price) / gain_range
                            tier2_stop = entry_price + 0.5 * gain_range
                            if progress >= 0.75 and trail_stop < tier2_stop:
                                trail_stop = tier2_stop
                            elif progress >= 0.50 and trail_stop < entry_price:
                                trail_stop = entry_price

        if not filled or np.isnan(exit_price):
            continue

        # P&L — SHORT: profit when price falls (entry > exit).
        # Both LONG and SHORT use equity MIS charges.
        # SHORT: sell to open at entry, buy to close at exit.
        #   → buy_val = exit_price*qty, sell_val = entry_price*qty (STT on entry)
        if is_short:
            gross_pnl = (entry_price - exit_price) * qty
            charges   = _intraday_charges(exit_price * qty, entry_price * qty)
        else:
            gross_pnl = (exit_price - entry_price) * qty
            charges   = _intraday_charges(entry_price * qty, exit_price * qty)
        net_pnl = gross_pnl - charges

        trades.append({
            "symbol":      symbol,
            "direction":   direction,
            "trade_date":  day,
            "nifty_trend": nifty_trend,
            "entry_time":  entry_time,
            "entry_price": round(entry_price, 2),
            "exit_time":   exit_time,
            "exit_price":  round(exit_price, 2),
            "qty":         qty,
            "stop_loss":   round(stop, 2),
            "target":      round(target, 2),
            "exit_reason": exit_reason,
            "gross_pnl":   round(gross_pnl, 2),
            "charges":     round(charges, 2),
            "net_pnl":     round(net_pnl, 2),
            "range_pct":   round(range_pct * 100, 2),
            "vol_ratio":   round(vol_ratio, 2),
        })

    return pd.DataFrame(trades)


def _intraday_charges(buy_val: float, sell_val: float) -> float:
    """
    NSE equity MIS intraday charges.
      buy_val  = value of the BUY leg (entry for LONG, exit/cover for SHORT)
      sell_val = value of the SELL leg (exit for LONG, entry/open for SHORT)
    STT 0.025% on sell side, NSE exchange txn charge 0.00335% both sides.
    """
    brokerage = 40.0 * 2
    stt       = sell_val * 0.00025
    exchange  = (buy_val + sell_val) * 0.0000335
    sebi      = (buy_val + sell_val) * 0.000001
    stamp     = buy_val * 0.00003
    gst       = (brokerage + exchange + sebi) * 0.18
    return brokerage + stt + exchange + sebi + stamp + gst
