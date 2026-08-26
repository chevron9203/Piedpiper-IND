"""
Monthly momentum run — schedule on last trading day of each month.

What it does:
  1. Computes the monthly momentum signal (same logic as forward_test_momentum_monthly)
  2. Sends result to Telegram (and email if configured)
  3. Logs the signal to logs/monthly_run/

Cron line (run at 6:00 PM IST; self-guards against non-last-trading-day runs):
  Mac (system timezone = IST):  0 18 * * 1-5 .../python3 .../monthly_run.py --capital 300000
  Cloud VM (UTC timezone):      30 12 * * 1-5 .../python3 .../monthly_run.py --capital 300000

Test run (skip last-trading-day guard, skip notifications):
  python scripts/monthly_run.py --force --no-notify --capital 100000

Or run manually at any month-end:
  python scripts/monthly_run.py --capital 100000 --holdings INFY,TCS
  python scripts/monthly_run.py --capital 100000 --force   # skip last-trading-day guard
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from data.universe import get_sector_map
from reporting.notifier import Notifier
from storage import store

_LOG_FILE = settings.LOG_DIR / "monthly_run" / "{time:YYYY-MM-DD}.log"
logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")
logger.add(str(_LOG_FILE), level="DEBUG", rotation="1 month", retention="12 months",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}")


def _is_last_trading_day_of_month(today: date, trading_dates: pd.DatetimeIndex) -> bool:
    """Return True if today is the last trading day in its calendar month."""
    this_month_dates = trading_dates[
        (trading_dates.month == today.month) & (trading_dates.year == today.year)
    ]
    if this_month_dates.empty:
        return False
    return pd.Timestamp(today) >= this_month_dates[-1]


def _format_telegram_msg(sig: dict, current_holdings: list[str] | None, capital: float) -> str:
    as_of    = sig["as_of"]
    n6m      = sig["nifty_6m_ret"]
    vix      = sig["vix_status"]
    n200     = sig.get("nifty_below_200ema", False)
    ema200   = sig.get("nifty_ema200", float("nan"))
    nifty_px = sig.get("nifty_now", float("nan"))
    status   = "IN MARKET" if sig["in_market"] else "DEFENSIVE"
    def_mode = sig.get("defensive_mode", "cash")

    gold_6m  = sig.get("gold_6m_ret", float("nan"))
    gold_str = f"  |  Gold 6m: {gold_6m*100:+.1f}%" if gold_6m == gold_6m else ""

    lines = [
        f"*Piedpiper Monthly — {as_of.strftime('%d %b %Y')}*\n",
        f"Regime: *{status}*",
        f"Nifty: {nifty_px:,.0f}  |  200 EMA: {ema200:,.0f}  |  {'⚠ BELOW 200 EMA' if n200 else '✓ above 200 EMA'}",
        f"Nifty 6m: {n6m*100:+.1f}%{gold_str}  |  VIX: {vix}",
    ]

    if not sig["in_market"]:
        if def_mode == "gold":
            goldbees_units = int(capital / 100)  # approximate at ~₹100/unit
            lines.append(f"\n*ACTION: Move to GOLDBEES*")
            lines.append(f"BUY ~{goldbees_units} units of GOLDBEES (NSE ETF) at market open")
            lines.append(f"Amount: ₹{capital:,.0f}  |  Re-check next month-end.")
        elif def_mode == "split":
            goldbees_units = int(capital * 0.5 / 100)
            liquidbees_units = int(capital * 0.5 / 1000)  # LIQUIDBEES ~₹1000/unit
            lines.append(f"\n*ACTION: Split defensive*")
            lines.append(f"BUY ~{goldbees_units} units GOLDBEES  (₹{capital*0.5:,.0f})")
            lines.append(f"BUY ~{liquidbees_units} units LIQUIDBEES  (₹{capital*0.5:,.0f})")
            lines.append(f"Re-check next month-end.")
        else:
            # Both equity AND gold bearish — park entirely in LIQUIDBEES
            liquidbees_units = int(capital / 1000)  # LIQUIDBEES ~₹1000/unit
            lines.append(f"\n*ACTION: Move to LIQUIDBEES (liquid ETF)*")
            lines.append(f"BUY ~{liquidbees_units} units of LIQUIDBEES (NSE: LIQUIDBEES) at market open")
            lines.append(f"Amount: ₹{capital:,.0f}  |  Earns ~6.5% p.a. (overnight rate)")
            lines.append(f"Re-check next month-end. Both equity and gold are bearish.")
        return "\n".join(lines)

    target  = sig["target"]
    weights = sig["weights"]
    meta    = sig["meta"]

    lines.append(f"\n*Portfolio ({len(target)} stocks):*")
    for sym in target:
        m     = meta.get(sym, {})
        w     = weights.get(sym, 1 / max(len(target), 1))
        alloc = capital * w
        lines.append(f"  {sym}: {w*100:.0f}%  ₹{alloc:,.0f}  ({m.get('score_12m',0)*100:+.0f}% 12m)")

    if current_holdings is not None:
        # LIQUIDBEES/GOLDBEES held defensively must be sold when re-entering market
        defensive_etfs = {"LIQUIDBEES", "GOLDBEES"}
        to_sell = [s for s in current_holdings if s not in target]
        to_buy  = [s for s in target if s not in current_holdings]
        hold    = [s for s in target if s in current_holdings]
        etf_exits = [s for s in to_sell if s in defensive_etfs]
        stock_exits = [s for s in to_sell if s not in defensive_etfs]
        if etf_exits:   lines.append(f"\n*SELL (defensive ETF exit):* {', '.join(etf_exits)}")
        if stock_exits: lines.append(f"*SELL:* {', '.join(stock_exits)}")
        if to_buy:      lines.append(f"*BUY:*  {', '.join(to_buy)}")
        if hold:    lines.append(f"*HOLD:* {', '.join(hold)}")
    else:
        lines.append(f"\n*BUY ALL:* {', '.join(target)}")

    lines.append(f"\n_Execute at open, 9:30–10:00 AM IST_")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Monthly momentum auto-runner")
    ap.add_argument("--capital",  type=float, default=300_000.0,
                    help="Current portfolio value for allocation display (default ₹3L momentum allocation)")
    ap.add_argument("--holdings", type=str, default="",
                    help="Comma-separated current holdings for buy/sell diff")
    ap.add_argument("--force",    action="store_true",
                    help="Skip last-trading-day guard and run regardless of date")
    ap.add_argument("--as-of",   type=str, default=None,
                    help="Override signal date YYYY-MM-DD")
    ap.add_argument("--no-notify", action="store_true",
                    help="Print report but suppress Telegram/email notification")
    ap.add_argument("--test",     action="store_true",
                    help="Alias for --force --no-notify: verify script runs without side effects")
    args = ap.parse_args()

    if args.test:
        args.force     = True
        args.no_notify = True

    today   = date.today()
    from_dt = date(today.year - 2, today.month, 1)
    to_dt   = today

    logger.info("Loading data ...")
    with store.db_conn() as conn:
        df = conn.execute("""
            SELECT symbol, dt, open, high, low, close, volume
            FROM adjusted_ohlcv
            WHERE dt >= ? AND dt <= ? AND source IN ('eod2', 'eod2_index')
            ORDER BY symbol, dt
        """, [from_dt, to_dt]).df()

    if df.empty:
        logger.error("No OHLCV data. Run: python scripts/ingest_data.py")
        sys.exit(1)

    df["dt"] = pd.to_datetime(df["dt"])
    ohlcv_all = {sym: grp.set_index("dt").sort_index()
                 for sym, grp in df.groupby("symbol")}

    nifty_df  = ohlcv_all.pop(settings.NIFTY50_SYMBOL, None)
    vix_df    = ohlcv_all.pop(settings.INDIA_VIX_SYMBOL, None)
    nifty_close = nifty_df["close"] if nifty_df is not None else pd.Series(dtype=float)
    vix_close   = vix_df["close"]   if vix_df is not None   else pd.Series(dtype=float)

    if nifty_close.empty:
        logger.error("Nifty 50 data missing")
        sys.exit(1)

    trading_dates = pd.DatetimeIndex(nifty_close.index)

    # Last-trading-day guard
    if not args.force and not args.as_of:
        if not _is_last_trading_day_of_month(today, trading_dates):
            logger.info("{} is not the last trading day of the month — skipping (use --force to override)", today)
            sys.exit(0)

    sector_df  = get_sector_map()
    sector_map = dict(zip(sector_df["symbol"], sector_df["sector"])) if not sector_df.empty else {}

    as_of = pd.Timestamp(args.as_of) if args.as_of else trading_dates[-1]
    logger.info("Signal date: {}  |  Universe: {} symbols  |  Capital: ₹{:,.0f}",
                as_of.date(), len(ohlcv_all), args.capital)

    # Import compute_signals from the forward test module (scripts/ on sys.path already)
    # Fetch GOLDBEES price for gold cash-fallback check (not in eod2 DB)
    gold_close: pd.Series | None = None
    try:
        import yfinance as yf
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = yf.download("GOLDBEES.NS", period="400d", interval="1d",
                              progress=False, auto_adjust=True)
        if not raw.empty:
            gold_close = raw["Close"].squeeze().dropna()
            gold_close.index = pd.to_datetime(gold_close.index).tz_localize(None)
            logger.info("Loaded {} days of GOLDBEES data", len(gold_close))
    except Exception as exc:
        logger.warning("Could not fetch GOLDBEES for gold check: {}", exc)

    _scripts_dir = str(Path(__file__).parent)
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    from forward_test_momentum_monthly import compute_signals, print_report  # noqa: PLC0415
    sig = compute_signals(
        ohlcv_dict=ohlcv_all,
        nifty_close=nifty_close,
        vix_close=vix_close if not vix_close.empty else None,
        sector_map=sector_map,
        as_of=as_of,
        gold_close=gold_close,
        composite_score=True,
    )

    current_holdings = [s.strip() for s in args.holdings.split(",") if s.strip()] or None
    print_report(sig, current_holdings=current_holdings, capital=args.capital)

    if not args.no_notify:
        msg = _format_telegram_msg(sig, current_holdings, args.capital)
        notifier = Notifier()
        ok = notifier.send_telegram(msg)
        logger.info("Telegram: {}", "sent" if ok else "not sent (check credentials)")

    # Persist signal to DuckDB for paper performance tracking
    _log_monthly_signal(sig, current_holdings, args.capital)

    # Save entry prices so live_orb.py can run per-stock stop-loss
    _log_entry_prices(sig, args.capital)

    logger.info("Monthly run complete.")


def _log_monthly_signal(sig: dict, holdings: list[str] | None, capital: float) -> None:
    """
    Save this month's signal to DuckDB monthly_signals table.
    Allows tracking paper portfolio performance month over month.
    """
    try:
        import json
        as_of_str = sig["as_of"].strftime("%Y-%m-%d") if hasattr(sig["as_of"], "strftime") else str(sig["as_of"])
        record = {
            "signal_date":   as_of_str,
            "in_market":     sig["in_market"],
            "defensive_mode": sig.get("defensive_mode", "cash"),
            "nifty_6m_ret":  round(float(sig.get("nifty_6m_ret", 0) or 0), 4),
            "gold_6m_ret":   round(float(sig.get("gold_6m_ret", float("nan")) or float("nan")), 4)
                             if sig.get("gold_6m_ret") == sig.get("gold_6m_ret") else None,
            "target":        json.dumps(sig.get("target", [])),
            "watchlist":     json.dumps(sig.get("watchlist", [])),
            "weights":       json.dumps({k: round(v, 4) for k, v in (sig.get("weights") or {}).items()}),
            "prior_holdings": json.dumps(holdings or []),
            "capital":       capital,
            "logged_at":     pd.Timestamp.now().isoformat(),
        }
        with store.db_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS monthly_signals (
                    signal_date   VARCHAR PRIMARY KEY,
                    in_market     BOOLEAN,
                    defensive_mode VARCHAR,
                    nifty_6m_ret  DOUBLE,
                    gold_6m_ret   DOUBLE,
                    target        VARCHAR,
                    watchlist     VARCHAR,
                    weights       VARCHAR,
                    prior_holdings VARCHAR,
                    capital       DOUBLE,
                    logged_at     VARCHAR
                )
            """)
            conn.execute("""
                INSERT INTO monthly_signals VALUES (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT (signal_date) DO UPDATE SET
                    in_market=excluded.in_market, defensive_mode=excluded.defensive_mode,
                    nifty_6m_ret=excluded.nifty_6m_ret, gold_6m_ret=excluded.gold_6m_ret,
                    target=excluded.target, watchlist=excluded.watchlist,
                    weights=excluded.weights,
                    prior_holdings=excluded.prior_holdings, capital=excluded.capital,
                    logged_at=excluded.logged_at
            """, list(record.values()))
        logger.info("Monthly signal logged to DB for {}", as_of_str)
    except Exception as exc:
        logger.warning("Could not log monthly signal to DB: {}", exc)


def _log_entry_prices(sig: dict, capital: float) -> None:
    """
    Save entry prices for the new momentum portfolio to momentum_entries table.
    Uses today's closing price as the proxy for tomorrow's execution price.
    Called immediately after month-end signal (market already closed at 6 PM).
    live_orb.py reads this table to run smart per-stock stop-loss.
    """
    if not sig.get("in_market") or not sig.get("target"):
        logger.info("Defensive mode — no entry prices to log")
        return

    import json
    target  = sig["target"]
    weights = sig.get("weights") or {}
    as_of   = sig["as_of"]
    as_of_str = as_of.strftime("%Y-%m-%d") if hasattr(as_of, "strftime") else str(as_of)

    try:
        with store.db_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS momentum_entries (
                    signal_date  VARCHAR,
                    symbol       VARCHAR,
                    entry_price  DOUBLE,
                    weight       DOUBLE,
                    qty          INTEGER,
                    hard_stop    DOUBLE,
                    capital      DOUBLE,
                    logged_at    VARCHAR,
                    PRIMARY KEY (signal_date, symbol)
                )
            """)

            # Latest close on or before signal date, per symbol
            rows = conn.execute("""
                SELECT a.symbol, a.close
                FROM adjusted_ohlcv a
                INNER JOIN (
                    SELECT symbol, MAX(dt) AS max_dt
                    FROM adjusted_ohlcv
                    WHERE symbol = ANY(?) AND dt <= ?
                    GROUP BY symbol
                ) b ON a.symbol = b.symbol AND a.dt = b.max_dt
            """, [target, as_of_str]).fetchall()

        price_map = {r[0]: float(r[1]) for r in rows}
        missing   = [s for s in target if s not in price_map]
        if missing:
            logger.warning("No closing price found for: {}", missing)

        with store.db_conn() as conn:
            for sym in target:
                price = price_map.get(sym)
                if not price or price <= 0:
                    continue
                w         = float(weights.get(sym, 1.0 / len(target)))
                qty       = max(1, int((capital * w) / price))
                hard_stop = round(price * 0.85, 2)   # -15% absolute floor
                conn.execute("""
                    INSERT INTO momentum_entries VALUES (?,?,?,?,?,?,?,?)
                    ON CONFLICT (signal_date, symbol) DO UPDATE SET
                        entry_price=excluded.entry_price, weight=excluded.weight,
                        qty=excluded.qty, hard_stop=excluded.hard_stop,
                        capital=excluded.capital, logged_at=excluded.logged_at
                """, [as_of_str, sym, price, w, qty, hard_stop, capital,
                      pd.Timestamp.now().isoformat()])

        logger.info("Entry prices saved for {}/{} stocks (signal_date={})",
                    len(price_map), len(target), as_of_str)
    except Exception as exc:
        logger.warning("Could not log entry prices: {}", exc)


if __name__ == "__main__":
    main()
