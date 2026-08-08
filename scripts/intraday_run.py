"""
Intraday ORB runner — schedule at 9:31 AM IST every trading day.

What it does:
  1. Authenticates with Angel One SmartAPI
  2. Fetches today's 15-min opening range bars for all Nifty 200 stocks
  3. Computes ORB signals (max 3 per day)
  4. Places entry MARKET orders + SL-M stop-loss orders
  5. Logs all orders to DuckDB (paper_trade flag controls real vs paper)
  6. Sends Telegram notification with the day's signals

LONG  (bull regime: Nifty ≥ EMA20): equity MIS BUY  + BUY-SL stop
SHORT (bear regime: Nifty < EMA20):  equity MIS SELL + SELL-SL stop (intraday short-sell)

Run:
  python scripts/intraday_run.py                  # paper mode (default)
  python scripts/intraday_run.py --live            # LIVE mode — real orders!
  python scripts/intraday_run.py --capital 50000

Cron (9:31 AM IST = 04:01 UTC, Mon-Fri):
  31 4 * * 1-5 /path/.venv/bin/python3 /path/scripts/intraday_run.py --capital 50000
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

from config import settings
from data.holiday_calendar import is_trading_day
from data.universe import fetch_nifty200
from execution.angel_orders import AngelOrderExecutor
from reporting.notifier import Notifier
from storage import store
from execution.angel_orders import VARIETY_SL
from strategies.orb_intraday import compute_orb_signals

IST = timezone(timedelta(hours=5, minutes=30))

DEFAULT_INTRADAY_CAPITAL = 50_000.0


def _get_intraday_universe() -> list[str]:
    """Fetch Nifty 200 as Angel One '-EQ' symbols; fall back to Nifty 50."""
    try:
        df = fetch_nifty200()
        syms = [s + "-EQ" for s in df["symbol"].tolist()]
        logger.info("Universe: {} Nifty 200 symbols loaded", len(syms))
        return syms
    except Exception as exc:
        logger.warning("Could not fetch Nifty 200 ({}), falling back to Nifty 50", exc)
        return [
            "RELIANCE-EQ", "TCS-EQ", "HDFCBANK-EQ", "INFY-EQ", "ICICIBANK-EQ",
            "HINDUNILVR-EQ", "SBIN-EQ", "BHARTIARTL-EQ", "ITC-EQ", "KOTAKBANK-EQ",
            "LT-EQ", "AXISBANK-EQ", "ASIANPAINT-EQ", "MARUTI-EQ", "SUNPHARMA-EQ",
            "TITAN-EQ", "BAJFINANCE-EQ", "WIPRO-EQ", "HCLTECH-EQ", "ULTRACEMCO-EQ",
            "NTPC-EQ", "POWERGRID-EQ", "TECHM-EQ", "NESTLEIND-EQ", "M&M-EQ",
            "JSWSTEEL-EQ", "TATASTEEL-EQ", "INDUSINDBK-EQ", "BAJAJ-AUTO-EQ",
            "HDFCLIFE-EQ", "BRITANNIA-EQ", "GRASIM-EQ", "DIVISLAB-EQ", "CIPLA-EQ",
            "DRREDDY-EQ", "ONGC-EQ", "COALINDIA-EQ", "ADANIPORTS-EQ", "BPCL-EQ",
            "EICHERMOT-EQ", "APOLLOHOSP-EQ", "HINDALCO-EQ", "BAJAJFINSV-EQ",
            "HEROMOTOCO-EQ", "SHRIRAMFIN-EQ", "TATACONSUM-EQ", "SBILIFE-EQ", "VEDL-EQ",
        ]


def _check_regime() -> bool:
    """
    True = bull (Nifty ≥ EMA20) → LONG ORB.
    False = bear (Nifty < EMA20) → SHORT ORB.
    DB first, yfinance fallback. Returns True (allow LONG) on any failure.
    """
    try:
        import duckdb
        con = duckdb.connect(str(settings.DB_PATH), read_only=True)
        rows = con.execute(
            "SELECT dt, close FROM adjusted_ohlcv WHERE symbol = 'Nifty 50' "
            "ORDER BY dt DESC LIMIT 40"
        ).fetchall()
        con.close()
        if rows:
            latest_date = rows[0][0]
            if (date.today() - latest_date).days <= 3:
                closes = pd.Series(
                    {pd.Timestamp(r[0]): float(r[1]) for r in rows}
                ).sort_index()
                ema20   = closes.ewm(span=20, adjust=False).mean()
                bull    = closes.iloc[-1] >= ema20.iloc[-1]
                logger.info("Regime (DB): Nifty {:.0f} vs EMA20 {:.0f} → {}",
                            closes.iloc[-1], ema20.iloc[-1], "BULL" if bull else "BEAR")
                return bool(bull)
            logger.info("DB Nifty data stale ({}) — falling back to yfinance", latest_date)
    except Exception as exc:
        logger.warning("Regime DB query failed: {}", exc)

    try:
        import yfinance as yf
        nifty   = yf.download("^NSEI", period="45d", interval="1d",
                              progress=False, auto_adjust=True)
        closes  = nifty["Close"].squeeze().dropna()
        ema20   = closes.ewm(span=20, adjust=False).mean()
        bull    = float(closes.iloc[-1]) >= float(ema20.iloc[-1])
        logger.info("Regime (yf): Nifty {:.0f} vs EMA20 {:.0f} → {}",
                    closes.iloc[-1], ema20.iloc[-1], "BULL" if bull else "BEAR")
        return bull
    except Exception as exc:
        logger.warning("Regime yfinance fallback failed ({}) — defaulting LONG", exc)
        return True


def _get_prev_closes(symbols: list[str]) -> dict[str, float]:
    """
    Return yesterday's closing price per symbol from adjusted_ohlcv (source=eod2).
    Used for the LONG gap filter: today's open must be ≥ yesterday's close × 1.003.
    """
    syms_clean = [s.replace("-EQ", "") for s in symbols]
    today = date.today()
    try:
        with store.db_conn() as conn:
            last_dt = conn.execute(
                "SELECT MAX(dt) FROM adjusted_ohlcv WHERE dt < ? AND source = 'eod2'",
                [today],
            ).fetchone()[0]
            if last_dt is None:
                return {}
            placeholders = ",".join(["?" for _ in syms_clean])
            rows = conn.execute(
                f"SELECT symbol, close FROM adjusted_ohlcv "
                f"WHERE dt = ? AND source = 'eod2' AND symbol IN ({placeholders})",
                [last_dt] + syms_clean,
            ).fetchall()
        result = {r[0]: float(r[1]) for r in rows}
        logger.info("Prev closes ({}): {} symbols", last_dt, len(result))
        return result
    except Exception as exc:
        logger.warning("Could not fetch prev closes: {}", exc)
        return {}


def _get_stock_ema20(symbols: list[str], n_lookback: int = 60) -> dict[str, float]:
    """
    Compute each stock's 20-day EMA from DB daily closes (source=eod2).
    Returns {clean_symbol: ema20_value} based on yesterday's close.
    Falls back to empty dict if DB unavailable.
    """
    syms_clean = [s.replace("-EQ", "") for s in symbols]
    today = date.today()
    try:
        with store.db_conn() as conn:
            last_dt = conn.execute(
                "SELECT MAX(dt) FROM adjusted_ohlcv WHERE dt < ? AND source = 'eod2'",
                [today],
            ).fetchone()[0]
            if last_dt is None:
                return {}
            from_dt = last_dt - timedelta(days=n_lookback)
            placeholders = ",".join(["?" for _ in syms_clean])
            df = conn.execute(
                f"SELECT symbol, dt, close FROM adjusted_ohlcv "
                f"WHERE dt >= ? AND dt <= ? AND source = 'eod2' AND symbol IN ({placeholders}) "
                f"ORDER BY symbol, dt",
                [from_dt, last_dt] + syms_clean,
            ).df()
        result: dict[str, float] = {}
        for sym, grp in df.groupby("symbol"):
            closes = grp.set_index("dt")["close"].sort_index()
            if len(closes) >= 10:
                ema = closes.ewm(span=20, adjust=False).mean()
                result[sym] = float(ema.iloc[-1])
        logger.info("Stock EMA20: computed for {}/{} symbols", len(result), len(syms_clean))
        return result
    except Exception as exc:
        logger.warning("Could not compute stock EMA20: {}", exc)
        return {}


def _fetch_today_15m_bars(symbol: str, token: str, executor: AngelOrderExecutor) -> pd.DataFrame:
    """Fetch today's 15-min bars from Angel One API, fall back to yfinance."""
    today   = date.today()
    from_dt = datetime.combine(today, time(9, 15)).replace(tzinfo=IST)
    to_dt   = datetime.combine(today, time(15, 30)).replace(tzinfo=IST)

    if token:
        try:
            from data.smartapi_client import fetch_candles
            df = fetch_candles(token, "NSE", "15m", from_dt, to_dt)
            if not df.empty:
                df.index = pd.to_datetime(df.index)
                if df.index.tzinfo is not None:
                    df.index = df.index.tz_localize(None)
                return df
        except Exception as exc:
            logger.warning("Angel One candle fetch failed for {} — falling back: {}", symbol, exc)

    try:
        import yfinance as yf
        sym_yf = symbol.replace("-EQ", "") + ".NS"
        df = yf.download(sym_yf, period="1d", interval="15m", progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        df.index = pd.to_datetime(df.index)
        if df.index.tzinfo is not None:
            df.index = df.index.tz_convert("Asia/Kolkata").tz_localize(None)
        return df.between_time("09:15", "15:30")
    except Exception as exc:
        logger.warning("yfinance fallback failed for {}: {}", symbol, exc)
        return pd.DataFrame()


def _batch_avg_volumes(symbols: list[str], n_days: int = 20) -> dict[str, float]:
    """Batch-fetch 20-day avg opening-bar volumes via yfinance (one call for all symbols)."""
    try:
        import yfinance as yf
        yf_syms = [s.replace("-EQ", "") + ".NS" for s in symbols]
        df = yf.download(
            yf_syms, period="60d", interval="15m",
            progress=False, auto_adjust=True, group_by="ticker",
        )
        if df.empty:
            return {}
        result: dict[str, float] = {}
        for sym, sym_yf in zip(symbols, yf_syms):
            clean = sym.replace("-EQ", "")
            try:
                sub = df[sym_yf] if sym_yf in df.columns.get_level_values(0) else pd.DataFrame()
                if sub.empty:
                    continue
                sub.columns = [c.lower() for c in sub.columns]
                sub.index = pd.to_datetime(sub.index)
                if sub.index.tzinfo is not None:
                    sub.index = sub.index.tz_convert("Asia/Kolkata").tz_localize(None)
                first_bars = sub.between_time("09:15", "09:29").resample("D").first()
                vol_avg = first_bars["volume"].tail(n_days).mean()
                result[clean] = float(vol_avg) if not pd.isna(vol_avg) else 0.0
            except Exception:
                result[clean] = 0.0
        logger.info("Batch avg volumes: got {} / {} symbols", len(result), len(symbols))
        return result
    except Exception as exc:
        logger.warning("Batch avg volume fetch failed: {}", exc)
        return {}


def _avg_volume_from_db(symbol: str, n_days: int = 20) -> float:
    """Per-symbol fallback: 20-day avg opening-bar volume from DuckDB or yfinance."""
    try:
        df = store.load_intraday_ohlcv(symbol.replace("-EQ", ""), "15m")
        if not df.empty:
            first_bars = df[df.index.time == time(9, 15)].tail(n_days)
            if not first_bars.empty:
                return float(first_bars["volume"].mean())
    except Exception:
        pass
    try:
        import yfinance as yf
        sym_yf = symbol.replace("-EQ", "") + ".NS"
        hist = yf.download(sym_yf, period="60d", interval="15m", progress=False, auto_adjust=True)
        if hist.empty:
            return 0.0
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = [c[0].lower() for c in hist.columns]
        else:
            hist.columns = [c.lower() for c in hist.columns]
        hist.index = pd.to_datetime(hist.index)
        if hist.index.tzinfo is not None:
            hist.index = hist.index.tz_convert("Asia/Kolkata").tz_localize(None)
        first_bars = hist.between_time("09:15", "09:29").resample("D").first()
        return float(first_bars["volume"].tail(n_days).mean()) if not first_bars.empty else 0.0
    except Exception:
        return 0.0


def _fetch_nifty_15m_today() -> pd.DataFrame:
    """
    Fetch today's Nifty 50 index 15-min bars — used for intraday direction filter.
    Angel One token 99926000 / exchange NSE. Falls back to yfinance (^NSEI).
    """
    today   = date.today()
    from_dt = datetime.combine(today, time(9, 15)).replace(tzinfo=IST)
    to_dt   = datetime.combine(today, time(15, 30)).replace(tzinfo=IST)
    try:
        from data.smartapi_client import fetch_candles
        df = fetch_candles("99926000", "NSE", "15m", from_dt, to_dt)
        if not df.empty:
            df.index = pd.to_datetime(df.index)
            if df.index.tzinfo is not None:
                df.index = df.index.tz_localize(None)
            logger.info("Nifty 15m (Angel One): {} bars", len(df))
            return df
    except Exception as exc:
        logger.warning("Nifty 15m Angel One failed: {}", exc)
    try:
        import yfinance as yf
        df = yf.download("^NSEI", period="1d", interval="15m", progress=False, auto_adjust=True)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() for c in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]
            df.index = pd.to_datetime(df.index)
            if df.index.tzinfo is not None:
                df.index = df.index.tz_convert("Asia/Kolkata").tz_localize(None)
            logger.info("Nifty 15m (yfinance): {} bars", len(df))
            return df.between_time("09:15", "15:30")
    except Exception as exc:
        logger.warning("Nifty 15m yfinance fallback failed: {}", exc)
    return pd.DataFrame()


def _get_token_map() -> dict[str, str]:
    """Load symbol → token from Angel One instrument master."""
    try:
        from data.instrument_master import get_nse_equity_master
        master = get_nse_equity_master()
        return dict(zip(master["symbol"], master["token"].astype(str)))
    except Exception as exc:
        logger.warning("Could not load instrument master: {}", exc)
        return {}


def _format_telegram_msg(signals: list, capital: float, paper: bool) -> str:
    mode = "PAPER" if paper else "LIVE"
    lines = [f"*Piedpiper Intraday ({mode}) — {date.today()}*\n"]
    if not signals:
        lines.append("No ORB signals today.")
        return "\n".join(lines)

    direction = signals[0].direction if signals else "LONG"
    lines.append(f"*{len(signals)} {direction} ORB signal(s)*\n")
    for sig in signals:
        qty      = sig.position_size(capital)
        label    = sig.symbol.replace("-EQ", "")
        risk_amt = abs(sig.entry_price - sig.stop_loss) * qty
        arrow    = "↓ SHORT" if sig.direction == "SHORT" else "↑ LONG"
        lines.append(
            f"*{label}* {arrow}\n"
            f"  Entry ₹{sig.entry_price:.2f}  SL ₹{sig.stop_loss:.2f}  Target ₹{sig.target:.2f}\n"
            f"  Qty: {qty}  Risk: ₹{risk_amt:,.0f}  Vol surge: {sig.vol_ratio:.1f}x"
        )
    lines.append(f"\n_Capital ₹{capital:,.0f} | Auto SQO 3:15 PM IST_")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Intraday ORB signal runner")
    ap.add_argument("--live",      action="store_true")
    ap.add_argument("--capital",   type=float, default=DEFAULT_INTRADAY_CAPITAL)
    ap.add_argument("--no-notify", action="store_true")
    args = ap.parse_args()

    today = date.today()
    paper = not args.live

    logger.info("=" * 60)
    logger.info("Intraday ORB — {} | {}", today, "LIVE" if not paper else "PAPER")
    logger.info("=" * 60)

    if not is_trading_day(today):
        logger.info("{} is not a trading day — exiting", today)
        return

    # Regime: BULL (Nifty ≥ EMA20) → LONG. BEAR → skip (shorts unprofitable at Nifty 200 scale).
    bull_regime = _check_regime()
    if not bull_regime:
        msg = f"*Intraday ORB {today}*\nBear regime (Nifty < EMA20) — skipping today. Shorts unprofitable at full universe scale."
        logger.info("Bear regime — no trades today")
        if not args.no_notify:
            Notifier().send_telegram(msg)
        return
    orb_direction = "LONG"
    logger.info("Bull regime → LONG ORB")

    store.init_schema()
    executor  = AngelOrderExecutor(paper=paper)
    token_map = _get_token_map()
    if not token_map and not paper:
        logger.error("No token map — cannot place live orders without instrument master")
        sys.exit(1)

    universe = _get_intraday_universe()

    # Batch avg volumes
    logger.info("Fetching 20-day avg opening volumes for {} symbols ...", len(universe))
    avg_volumes = _batch_avg_volumes(universe)
    for sym in universe:
        clean = sym.replace("-EQ", "")
        if clean not in avg_volumes or avg_volumes[clean] == 0.0:
            avg_volumes[clean] = _avg_volume_from_db(sym)
    avg_volumes.update({sym: avg_volumes.get(sym.replace("-EQ", ""), 0.0) for sym in universe})

    # Stock EMA20 + yesterday's closes (needed for both gap filter and EMA consistency)
    stock_ema20 = _get_stock_ema20(universe)
    prev_closes = _get_prev_closes(universe)  # always fetched; used for gap+EMA filters

    # Today's 15-min bars
    logger.info("Fetching 15-min bars for {} symbols ...", len(universe))
    candles_15m: dict[str, pd.DataFrame] = {}
    for sym in universe:
        token = token_map.get(sym, "")
        df = _fetch_today_15m_bars(sym, token, executor)
        if not df.empty:
            candles_15m[sym] = df

    logger.info("Got opening range data for {} symbols", len(candles_15m))

    # Nifty 50 index 15-min bars for intraday direction filter (UP/DOWN/FLAT)
    nifty_15m_today = _fetch_nifty_15m_today()

    if not candles_15m:
        msg = f"*Intraday ORB {today}*\nNo opening range data — check data feed."
        logger.warning(msg)
        if not args.no_notify:
            Notifier().send_telegram(msg)
        return

    # Compute ORB signals — LONG only, stock EMA20 filter, no gap filter
    signals = compute_orb_signals(
        candles_15m=candles_15m,
        token_map=token_map,
        nifty_candles_15m=nifty_15m_today if not nifty_15m_today.empty else None,
        avg_volumes=avg_volumes,
        trade_date=today,
        direction=orb_direction,
        stock_ema20=stock_ema20,
        prev_closes=prev_closes,
        gap_filter_pct=0.0,
    )

    if not signals:
        reason = "No stocks above EMA20 with sufficient vol surge"
        msg = f"*Intraday ORB {today}* (LONG)\nNo signals. {reason}."
        logger.info(msg)
        if not args.no_notify:
            Notifier().send_telegram(msg)
        return

    # Place orders and log trades
    # Strip timezone — DuckDB TIMESTAMP is tz-naive; storing tz-aware causes offset errors
    now = datetime.now(IST).replace(tzinfo=None)
    for sig in signals:
        qty = sig.position_size(args.capital)
        if qty == 0:
            continue

        entry_order_id = None
        sl_order_id    = None
        is_short       = sig.direction == "SHORT"

        if not paper:
            # Place entry first, then SL-M. If SL fails, cancel entry immediately.
            try:
                entry_side = "SELL" if is_short else "BUY"
                entry_order_id = executor.place_market_order(
                    sig.symbol, sig.token, qty, entry_side
                )
            except Exception as exc:
                logger.error("Entry order failed for {}: {}", sig.symbol, exc)
                continue

            try:
                sl_side = "BUY" if is_short else "SELL"
                sl_order_id = executor.place_sl_market_order(
                    sig.symbol, sig.token, qty, sl_side, sig.stop_loss
                )
            except Exception as exc:
                logger.error("SL-M failed for {} — cancelling entry order {}: {}",
                             sig.symbol, entry_order_id, exc)
                executor.cancel_order(entry_order_id)
                Notifier().send_telegram(
                    f"⚠️ *SL order failed* for {sig.symbol} — entry cancelled. Check Angel One!"
                )
                continue
        else:
            action = "SELL (short)" if is_short else "BUY"
            logger.info("PAPER: Would {} {} x {} @ {:.2f}, SL {:.2f}, Target {:.2f}",
                        action, sig.symbol, qty, sig.entry_price, sig.stop_loss, sig.target)

        clean_sym = sig.symbol.replace("-EQ", "")
        store.log_intraday_trade({
            "trade_id":       str(uuid.uuid4())[:12],
            "strategy":       "orb_short" if is_short else "orb",
            "symbol":         clean_sym,
            "trade_date":     today,
            "direction":      sig.direction,
            "entry_time":     now,
            "entry_price":    sig.entry_price,
            "qty":            qty,
            "stop_loss":      sig.stop_loss,
            "target":         sig.target,
            "exit_time":      None,
            "exit_price":     None,
            "exit_reason":    None,
            "gross_pnl":      None,
            "charges":        None,
            "net_pnl":        None,
            "angel_entry_id": entry_order_id,
            "angel_exit_id":  sl_order_id,
            "paper_trade":    paper,
        })
        logger.info("Trade logged: {} {} x {} @ {:.2f} | SL {:.2f} | Target {:.2f}",
                    sig.direction, clean_sym, qty, sig.entry_price, sig.stop_loss, sig.target)

    if not args.no_notify:
        Notifier().send_telegram(_format_telegram_msg(signals, args.capital, paper))

    logger.info("Intraday run complete. {} signal(s) processed.", len(signals))


if __name__ == "__main__":
    main()
