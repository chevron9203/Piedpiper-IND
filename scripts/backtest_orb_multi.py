"""
Multi-symbol ORB backtest — LONG, SHORT, and regime-switching COMBINED.

Configs compared:
  1. LONG BEFORE — gap=0.3% + no stock EMA  (old system, baseline)
  2. LONG NEW    — no gap filter + stock EMA20 alignment  ← LIVE SYSTEM
  3. SHORT NEW   — equity MIS short + stock EMA20 alignment (reference only — unprofitable at 200-symbol scale)
  4. LONG NEW ←LIVE — same as #2, confirms live config

Data loading (priority order):
  1. DuckDB intraday_ohlcv — populated by scripts/backfill_intraday.py (Angel One)
  2. yfinance fallback — only last 60 days, used when DB has no data for a symbol

Usage:
  python scripts/backtest_orb_multi.py                      # last 60 days, Nifty 50
  python scripts/backtest_orb_multi.py --from-date 2024-01-01  # 1.5 year backtest
  python scripts/backtest_orb_multi.py --capital 200000
  python scripts/backtest_orb_multi.py --quick              # Nifty 50 only (faster)

Run backfill first to enable long backtests:
  python scripts/backfill_intraday.py --quick   # Nifty 50, ~30 min via Angel One
  python scripts/backfill_intraday.py           # Nifty 200, ~2-3 hours
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.orb_intraday import (
    backtest_orb_on_candles,
    VOL_SURGE_MULT,
    STOP_FRAC,
    TARGET_MULT,
    RISK_PCT,
)

logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")

NIFTY50 = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "SBIN",
    "BHARTIARTL", "ITC", "KOTAKBANK", "LT", "AXISBANK", "ASIANPAINT", "MARUTI",
    "SUNPHARMA", "TITAN", "BAJFINANCE", "WIPRO", "HCLTECH", "ULTRACEMCO",
    "NTPC", "POWERGRID", "TECHM", "NESTLEIND", "M&M", "JSWSTEEL", "TATASTEEL",
    "INDUSINDBK", "BAJAJ-AUTO", "HDFCLIFE", "BRITANNIA", "GRASIM", "DIVISLAB",
    "CIPLA", "DRREDDY", "ONGC", "COALINDIA", "ADANIPORTS", "BPCL", "EICHERMOT",
    "APOLLOHOSP", "HINDALCO", "BAJAJFINSV", "HEROMOTOCO", "SHRIRAMFIN",
    "TATACONSUM", "SBILIFE", "VEDL",
]

YF_PERIOD_CAP = 59  # yfinance 15-min history cap in days


# ── Data loading ──────────────────────────────────────────────────────────────

def _symbols_from_db() -> set[str]:
    """Return set of symbols that have 15-min data in DuckDB."""
    try:
        from storage import store
        store.init_schema()
        with store.db_conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT symbol FROM intraday_ohlcv WHERE interval = '15m'"
            ).fetchall()
        return {r[0] for r in rows}
    except Exception:
        return set()


def _load_from_db(symbols: list[str], from_date: date, to_date: date) -> dict[str, pd.DataFrame]:
    """Load 15-min bars from DuckDB intraday_ohlcv."""
    from storage import store
    result: dict[str, pd.DataFrame] = {}
    from_dt = datetime.combine(from_date, datetime.min.time())
    to_dt   = datetime.combine(to_date,   datetime.max.time())
    for sym in symbols:
        try:
            df = store.load_intraday_ohlcv(sym, "15m", from_dt=from_dt, to_dt=to_dt)
            if not df.empty:
                df = df.between_time("09:15", "15:30").dropna(subset=["close"])
                if not df.empty:
                    result[sym] = df
        except Exception:
            pass
    logger.info("DB load: {}/{} symbols have 15-min data", len(result), len(symbols))
    return result


def _download_yf(symbols: list[str], period_days: int) -> dict[str, pd.DataFrame]:
    """Download last N days of 15-min bars from yfinance (max 59 days)."""
    import yfinance as yf
    period_days = min(period_days, YF_PERIOD_CAP)
    yf_syms = [s + ".NS" for s in symbols]
    logger.info("yfinance fallback: {}-day 15-min for {} symbols", period_days, len(yf_syms))
    result: dict[str, pd.DataFrame] = {}
    chunk_size = 60
    for i in range(0, len(yf_syms), chunk_size):
        chunk = yf_syms[i: i + chunk_size]
        try:
            raw = yf.download(chunk, period=f"{period_days}d", interval="15m",
                              group_by="ticker", progress=False, auto_adjust=True)
        except Exception as exc:
            logger.warning("yf download failed chunk: {}", exc)
            continue
        if raw.empty:
            continue
        for sym_yf in chunk:
            sym = sym_yf.replace(".NS", "")
            try:
                sub = raw[sym_yf] if sym_yf in raw.columns.get_level_values(0) else pd.DataFrame()
                if sub.empty or not isinstance(sub, pd.DataFrame):
                    continue
                sub = sub.copy()
                sub.columns = [c.lower() for c in sub.columns]
                sub.index   = pd.to_datetime(sub.index)
                if sub.index.tzinfo is not None:
                    sub.index = sub.index.tz_convert("Asia/Kolkata").tz_localize(None)
                sub = sub.between_time("09:15", "15:30").dropna(subset=["close"])
                if not sub.empty:
                    result[sym] = sub
            except Exception:
                pass
    logger.info("yfinance: got {}/{} symbols", len(result), len(yf_syms))
    return result


def _load_all_data(
    symbols: list[str], from_date: date, to_date: date,
    db_only: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    DB first (any date range) → yfinance fallback (last 59d only).
    If db_only=True, only use symbols that have full-history data in DuckDB.
    """
    db_syms     = _symbols_from_db()
    in_db       = [s for s in symbols if s in db_syms]
    not_in_db   = [s for s in symbols if s not in db_syms]

    all_data: dict[str, pd.DataFrame] = {}

    if in_db:
        all_data.update(_load_from_db(in_db, from_date, to_date))

    if not db_only:
        # yfinance fallback only works for the last 59 days
        days_back = (to_date - from_date).days
        if not_in_db:
            yf_data = _download_yf(not_in_db, min(days_back, YF_PERIOD_CAP))
            # Only keep if dates overlap with requested range
            cutoff = datetime.combine(from_date, datetime.min.time())
            for sym, df in yf_data.items():
                if not df.empty and df.index[-1] >= cutoff:
                    all_data[sym] = df

    logger.info("Total data: {} symbols | from {} to {}",
                len(all_data), from_date, to_date)
    return all_data


# ── Nifty daily close ─────────────────────────────────────────────────────────

def _nifty_daily_from_db(from_date: date, to_date: date) -> pd.Series:
    """Load Nifty 50 daily closes from DuckDB adjusted_ohlcv (eod2/eod2_index)."""
    try:
        from storage import store
        with store.db_conn() as conn:
            rows = conn.execute(
                "SELECT dt, close FROM adjusted_ohlcv "
                "WHERE symbol = 'Nifty 50' AND dt >= ? AND dt <= ? "
                "ORDER BY dt",
                [from_date, to_date],
            ).fetchall()
        if rows:
            s = pd.Series({pd.Timestamp(r[0]): float(r[1]) for r in rows})
            logger.info("Nifty daily (DB): {} days ({} → {})",
                        len(s), s.index[0].date(), s.index[-1].date())
            return s
    except Exception as exc:
        logger.warning("Nifty DB load failed: {}", exc)
    return pd.Series(dtype=float)


def _nifty_daily_from_yf(days: int = 60) -> pd.Series:
    """Nifty daily from yfinance — fallback."""
    try:
        import yfinance as yf
        df = yf.download("^NSEI", period=f"{days}d", interval="1d",
                         progress=False, auto_adjust=True)
        closes = df["Close"].squeeze().dropna()
        closes.index = pd.to_datetime(closes.index)
        if closes.index.tzinfo is not None:
            closes.index = closes.index.tz_localize(None)
        logger.info("Nifty daily (yfinance): {} days", len(closes))
        return closes
    except Exception as exc:
        logger.warning("Nifty yfinance failed: {}", exc)
        return pd.Series(dtype=float)


def _nifty_ema20(nifty_close: pd.Series) -> pd.Series:
    return nifty_close.ewm(span=20, adjust=False).mean()


# ── Backtest core ─────────────────────────────────────────────────────────────

def _run_config(
    all_data:         dict[str, pd.DataFrame],
    symbols:          list[str],
    capital:          float,
    nifty_close:      pd.Series,
    nifty_ema20:      pd.Series | None,
    direction:        str,
    label:            str,
    stock_ema_filter: bool        = True,
    gap_pct:          float       = 0.0,
    long_gap_pct:     float | None = None,  # separate gap for LONG leg of COMBINED
) -> dict:
    per_symbol: list[pd.DataFrame] = []

    for sym in symbols:
        df = all_data.get(sym)
        if df is None or df.empty:
            continue

        if direction == "COMBINED":
            effective_long_gap = long_gap_pct if long_gap_pct is not None else gap_pct
            long_t  = backtest_orb_on_candles(sym, df, capital=capital,
                        nifty_close=nifty_close, nifty_ema20=nifty_ema20,
                        gap_filter_pct=effective_long_gap, trail2=True, direction="LONG",
                        stock_ema_filter=stock_ema_filter)
            short_t = backtest_orb_on_candles(sym, df, capital=capital,
                        nifty_close=nifty_close, nifty_ema20=nifty_ema20,
                        gap_filter_pct=gap_pct, trail2=True, direction="SHORT",
                        stock_ema_filter=stock_ema_filter)
            parts = [t for t in [long_t, short_t] if not t.empty]
            if parts:
                per_symbol.append(pd.concat(parts, ignore_index=True))
        else:
            t = backtest_orb_on_candles(sym, df, capital=capital,
                    nifty_close=nifty_close,
                    nifty_ema20=nifty_ema20,
                    gap_filter_pct=gap_pct, trail2=True, direction=direction,
                    stock_ema_filter=stock_ema_filter)
            if not t.empty:
                per_symbol.append(t)

    if not per_symbol:
        return {"label": label, "trades": 0, "trade_days": 0, "trades_p_day": 0,
                "win_rate": 0, "avg_win": 0, "avg_loss": 0, "rr": 0,
                "net_pnl": 0, "avg_per_trade": 0, "targets": 0, "stops": 0,
                "squareoffs": 0, "symbols_hit": 0, "long_pnl": 0, "short_pnl": 0,
                "df": pd.DataFrame()}

    all_trades = pd.concat(per_symbol, ignore_index=True)
    all_trades["trade_date"] = pd.to_datetime(all_trades["trade_date"])
    capped = all_trades

    n       = len(capped)
    wins    = (capped["net_pnl"] > 0).sum()
    losses  = n - wins
    win_pct = wins / n * 100 if n > 0 else 0
    net_pnl = capped["net_pnl"].sum()
    avg_win  = capped.loc[capped["net_pnl"] > 0, "net_pnl"].mean() if wins   > 0 else 0
    avg_loss = capped.loc[capped["net_pnl"] < 0, "net_pnl"].mean() if losses > 0 else 0
    rr       = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    reasons      = capped["exit_reason"].value_counts().to_dict()
    trading_days = capped["trade_date"].nunique()
    long_pnl  = capped.loc[capped["direction"] == "LONG",  "net_pnl"].sum()
    short_pnl = capped.loc[capped["direction"] == "SHORT", "net_pnl"].sum()

    return {
        "label":         label,
        "trades":        n,
        "trade_days":    trading_days,
        "trades_p_day":  round(n / trading_days if trading_days else 0, 1),
        "win_rate":      round(win_pct, 1),
        "avg_win":       round(avg_win, 0),
        "avg_loss":      round(avg_loss, 0),
        "rr":            round(rr, 2),
        "net_pnl":       round(net_pnl, 0),
        "avg_per_trade": round(net_pnl / n if n else 0, 0),
        "targets":       reasons.get("target", 0),
        "stops":         reasons.get("stop", 0),
        "squareoffs":    reasons.get("squareoff", 0),
        "symbols_hit":   capped["symbol"].nunique(),
        "long_pnl":      round(long_pnl, 0),
        "short_pnl":     round(short_pnl, 0),
        "df":            capped,
    }


def _print_result(r: dict) -> None:
    sep = "─" * 68
    print(f"\n{sep}")
    print(f"  {r['label']}")
    print(sep)
    print(f"  Trades   : {r['trades']}  over {r['trade_days']} active days  "
          f"({r['trades_p_day']:.1f}/day)  |  {r['symbols_hit']} symbols")
    print(f"  Win rate : {r['win_rate']:.1f}%  |  RR {r['rr']:.2f}x  |  "
          f"Avg win ₹{r['avg_win']:+,.0f}  Avg loss ₹{r['avg_loss']:+,.0f}")
    print(f"  Exits    : {r['targets']} target  |  {r['stops']} stop  |  "
          f"{r['squareoffs']} squareoff")
    print(f"  Net P&L  : ₹{r['net_pnl']:+,.0f}  (avg ₹{r['avg_per_trade']:+,.0f}/trade)")
    if r["long_pnl"] != 0 or r["short_pnl"] != 0:
        print(f"  Split    : LONG ₹{r['long_pnl']:+,.0f}  |  SHORT ₹{r['short_pnl']:+,.0f}")


def _print_monthly_breakdown(df: pd.DataFrame, label: str) -> None:
    """Print month-by-month P&L for a config — useful for understanding consistency."""
    if df.empty:
        return
    df = df.copy()
    df["ym"] = pd.to_datetime(df["trade_date"]).dt.to_period("M")
    monthly = df.groupby("ym")["net_pnl"].agg(["sum", "count"]).reset_index()
    monthly.columns = ["month", "net_pnl", "trades"]
    print(f"\n  Monthly breakdown — {label}")
    print(f"  {'Month':<12} {'Trades':>6} {'Net P&L':>12} {'Cum P&L':>12}")
    cum = 0
    for _, row in monthly.iterrows():
        cum += row["net_pnl"]
        sign = "+" if row["net_pnl"] >= 0 else ""
        print(f"  {str(row['month']):<12} {int(row['trades']):>6} "
              f"₹{row['net_pnl']:>+9,.0f}    ₹{cum:>+9,.0f}")


def _get_universe() -> list[str]:
    try:
        from data.universe import fetch_nifty200
        df = fetch_nifty200()
        return df["symbol"].tolist()
    except Exception as exc:
        logger.warning("Nifty 200 fetch failed: {} — using Nifty 50", exc)
        return NIFTY50


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="ORB multi-symbol backtest")
    ap.add_argument("--capital",   type=float, default=100_000,
                    help="Intraday capital per trade (default ₹1L)")
    ap.add_argument("--quick",     action="store_true",
                    help="Nifty 50 only (faster)")
    ap.add_argument("--from-date", type=str, default=None,
                    help="Backtest start YYYY-MM-DD (default: 60 days ago). "
                         "Run backfill_intraday.py first for dates older than 60d.")
    ap.add_argument("--monthly",   action="store_true",
                    help="Show month-by-month P&L breakdown for COMBINED config")
    ap.add_argument("--db-only",   action="store_true",
                    help="Only use symbols with full history in DuckDB (skip yfinance fallback)")
    args = ap.parse_args()

    to_date   = date.today() - timedelta(days=1)   # yesterday (today not complete)
    if args.from_date:
        from_date = date.fromisoformat(args.from_date)
    else:
        from_date = to_date - timedelta(days=59)

    days_total = (to_date - from_date).days + 1

    symbols = NIFTY50 if args.quick else _get_universe()

    # ── Load 15-min bars ──────────────────────────────────────────────────
    all_data = _load_all_data(symbols, from_date, to_date, db_only=args.db_only)

    # ── Nifty daily close for regime filter ───────────────────────────────
    # Load extra 60 days before from_date for EMA warmup
    nifty_from = from_date - timedelta(days=60)
    nifty_close = _nifty_daily_from_db(nifty_from, to_date)
    if nifty_close.empty:
        nifty_close = _nifty_daily_from_yf(days=days_total + 60)

    if nifty_close.empty:
        logger.error("No Nifty data — cannot compute regime filter. Exiting.")
        sys.exit(1)

    nifty_ema = _nifty_ema20(nifty_close)

    # Regime days in the actual backtest window
    win_close = nifty_close[nifty_close.index >= pd.Timestamp(from_date)]
    win_ema   = nifty_ema[nifty_ema.index >= pd.Timestamp(from_date)]
    aligned   = win_close.align(win_ema, join="inner")
    bull_days = int((aligned[0] >= aligned[1]).sum())
    bear_days = int((aligned[0] <  aligned[1]).sum())

    # ── Print header ──────────────────────────────────────────────────────
    print(f"\n{'═'*68}")
    print(f"  ORB Backtest  {from_date} → {to_date}  ({days_total}d)")
    print(f"  Capital ₹{args.capital:,.0f}  |  {len(all_data)}/{len(symbols)} symbols with data")
    print(f"  Params: vol_surge={VOL_SURGE_MULT}x | stop={STOP_FRAC} | RR={TARGET_MULT}x | risk={RISK_PCT*100:.0f}%")
    print(f"  Regime: BULL {bull_days}d  BEAR {bear_days}d  "
          f"({bull_days/(bull_days+bear_days)*100:.0f}% bull)" if bull_days+bear_days > 0 else "")
    print(f"{'═'*68}")

    # ── Run configs ───────────────────────────────────────────────────────
    # (label, direction, use_nifty_ema, stock_ema_filter, gap_pct, long_gap_pct, descr)
    configs = [
        ("LONG BEFORE │ gap=0.3% noEMA",     "LONG",     False, False, 0.003, None,
         "LONG every day | gap 0.3% | no stock EMA  [OLD BASELINE]"),
        ("LONG NEW    │ no-gap + stockEMA",  "LONG",     True,  True,  0.0,   None,
         "LONG bull days | no gap filter | stock EMA20"),
        ("SHORT NEW   │ equity MIS",         "SHORT",    True,  True,  0.0,   None,
         "SHORT bear days | equity MIS | stock EMA20"),
        ("LONG NEW    │ no-gap+EMA ←LIVE",   "LONG",     True,  True,  0.0,   None,
         "LONG bull days | no gap filter | stock EMA20  ←LIVE SYSTEM"),
    ]

    results = []
    for label, dirn, use_regime, use_stock_ema, gap, long_gap, descr in configs:
        logger.info("Running: {}", descr)
        ema_arg = nifty_ema if use_regime else None
        r = _run_config(all_data, symbols, args.capital,
                        nifty_close, ema_arg, dirn, label,
                        stock_ema_filter=use_stock_ema, gap_pct=gap,
                        long_gap_pct=long_gap)
        results.append(r)
        _print_result(r)

    # Monthly breakdown for COMBINED
    combined_r = results[-1]
    if args.monthly and not combined_r["df"].empty:
        _print_monthly_breakdown(combined_r["df"], combined_r["label"])

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'═'*68}")
    print("  COMPARISON SUMMARY")
    print(f"{'═'*68}")
    print(f"  {'Config':<38} {'Trades':>6} {'Win%':>6} {'AvgPT':>8} {'RR':>5} {'NetPnL':>10}")
    print(f"  {'-'*38} {'-'*6} {'-'*6} {'-'*8} {'-'*5} {'-'*10}")
    for r in results:
        print(f"  {r['label']:<38} {r['trades']:>6} {r['win_rate']:>5.1f}% "
              f"₹{r['avg_per_trade']:>6,.0f} {r['rr']:>5.2f}x ₹{r['net_pnl']:>8,.0f}")
    print()
    print("  AvgPT = avg net P&L per trade | RR = avg_win / avg_loss")
    print("  For dates > 60d ago: run backfill_intraday.py first (Angel One)")
    if not args.from_date:
        print("  Tip: python scripts/backtest_orb_multi.py --from-date 2024-01-01 --monthly")


if __name__ == "__main__":
    main()
