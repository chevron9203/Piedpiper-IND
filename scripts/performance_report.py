"""
Performance report — monthly and yearly CAGR for all strategies.

Shows:
  1. Monthly Momentum (swing) — from backtest equity curve
  2. Intraday ORB — from live/paper trade log in DuckDB
  3. Combined portfolio view

Monthly return grid (calendar format):
        Jan    Feb    Mar   ...   Dec   Full Year
  2021  +3.2%  +1.8%  ...        ...   +25.6%
  2022  ...

Usage:
  python scripts/performance_report.py                # all strategies
  python scripts/performance_report.py --intraday     # intraday only
  python scripts/performance_report.py --swing        # swing only
  python scripts/performance_report.py --capital-swing 100000 --capital-intraday 50000
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))  # project root
sys.path.insert(0, str(Path(__file__).parent))          # scripts/

from config import settings
from data.universe import get_sector_map
from storage import store


# ── Helpers ────────────────────────────────────────────────────────────────────

def _cagr(start: float, end: float, years: float) -> float:
    if years <= 0 or start <= 0:
        return 0.0
    return (end / start) ** (1 / years) - 1


def _maxdd(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    return float(dd.min())


def _monthly_returns_grid(monthly_eq: pd.Series, start_capital: float) -> pd.DataFrame:
    """
    Build a month×year calendar grid of returns.
    monthly_eq: Series indexed by month-end dates, values = equity
    """
    if monthly_eq.empty:
        return pd.DataFrame()

    # Add starting capital as the first "equity point" before any trades
    first = monthly_eq.index[0]
    start_ts = pd.Timestamp(first.year, first.month, 1) - pd.DateOffset(months=1)
    eq = pd.concat([pd.Series([start_capital], index=[start_ts]), monthly_eq])
    eq.index = pd.to_datetime(eq.index)

    records = []
    for i in range(1, len(eq)):
        prev = eq.iloc[i - 1]
        curr = eq.iloc[i]
        ret  = curr / prev - 1.0 if prev > 0 else np.nan
        records.append({
            "year":  eq.index[i].year,
            "month": eq.index[i].month,
            "ret":   ret,
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    grid = df.pivot(index="year", columns="month", values="ret")
    _month_names = {
        1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
        7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec",
    }
    grid.columns = [_month_names[m] for m in grid.columns]

    # Annual return = product of monthly returns
    month_cols = list(grid.columns)
    grid["Year"] = grid[month_cols].apply(
        lambda row: np.prod([1 + r for r in row.dropna()]) - 1, axis=1
    )
    return grid


def _print_monthly_grid(grid: pd.DataFrame, label: str) -> None:
    if grid.empty:
        print(f"\n  {label}: No data available\n")
        return

    month_cols = [c for c in grid.columns if c not in ("", "Year")]

    print(f"\n  {label} — Monthly Returns")
    header = f"  {'Year':>5}  " + "  ".join(f"{m:>6}" for m in month_cols) + "  {'Year':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for yr, row in grid.iterrows():
        vals = []
        for m in month_cols:
            v = row.get(m, np.nan)
            if pd.isna(v):
                vals.append("      ")
            else:
                vals.append(f"{v*100:+6.1f}")
        yr_ret = row.get("Year", np.nan)
        yr_str = f"{yr_ret*100:+7.1f}%" if not pd.isna(yr_ret) else "       "
        print(f"  {yr:>5}  " + "  ".join(vals) + f"  {yr_str}")


# ── Swing / Monthly Momentum ───────────────────────────────────────────────────

def _get_swing_equity(capital: float) -> pd.Series:
    """Run the monthly momentum backtest and return the equity curve."""
    from_dt = date(2015, 1, 1)
    to_dt   = date.today()

    with store.db_conn() as conn:
        df = conn.execute("""
            SELECT symbol, dt, open, high, low, close, volume
            FROM adjusted_ohlcv
            WHERE dt >= ? AND dt <= ? AND source IN ('eod2', 'eod2_index')
            ORDER BY symbol, dt
        """, [from_dt, to_dt]).df()

    if df.empty:
        return pd.Series(dtype=float)

    df["dt"] = pd.to_datetime(df["dt"])
    ohlcv_all = {sym: grp.set_index("dt").sort_index() for sym, grp in df.groupby("symbol")}

    nifty_df = ohlcv_all.pop(settings.NIFTY50_SYMBOL, None)
    vix_df   = ohlcv_all.pop(settings.INDIA_VIX_SYMBOL, None)

    if nifty_df is None:
        return pd.Series(dtype=float)

    nifty_close = nifty_df["close"]
    vix_close   = vix_df["close"] if vix_df is not None else pd.Series(dtype=float)

    universe   = list(ohlcv_all.keys())
    sector_df  = get_sector_map()
    sector_map = dict(zip(sector_df["symbol"], sector_df["sector"])) if not sector_df.empty else {}

    from backtest_momentum_monthly import run_monthly_momentum, _load_goldbees  # noqa: PLC0415

    gold_close = _load_goldbees(from_dt, to_dt)

    eq_curve, _ = run_monthly_momentum(
        ohlcv_dict=ohlcv_all,
        nifty_close=nifty_close,
        universe_symbols=universe,
        vix_close=vix_close if not vix_close.empty else None,
        start_capital=capital,
        inv_vol_weight=True,
        defensive_mode="gold",
        gold_close=gold_close if not gold_close.empty else None,
        sector_map=sector_map,
    )
    return eq_curve


# ── Intraday ORB ──────────────────────────────────────────────────────────────

def _get_intraday_equity(capital: float, paper_only: bool = False) -> pd.Series:
    """Build equity curve from intraday trade log."""
    trades = store.load_intraday_trades(paper_only=paper_only)
    if trades.empty:
        return pd.Series(dtype=float)

    trades["trade_date"] = pd.to_datetime(trades["trade_date"])
    trades = trades.dropna(subset=["net_pnl"])

    # Daily P&L
    daily = trades.groupby("trade_date")["net_pnl"].sum()

    # Build equity curve
    equity = capital
    eq_pts: list[tuple] = []
    for dt, pnl in daily.items():
        equity += pnl
        eq_pts.append((dt, equity))

    if not eq_pts:
        return pd.Series(dtype=float)

    return pd.Series(
        [e for _, e in eq_pts],
        index=[d for d, _ in eq_pts],
        name="intraday_equity",
    )


def _run_intraday_backtest(capital: float) -> pd.Series:
    """
    Quick backtest on yfinance 15-min data for a subset of Nifty 50 stocks.
    Uses last 60 days (yfinance intraday limit).
    Returns daily equity curve.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not available — cannot run intraday backtest")
        return pd.Series(dtype=float)

    from strategies.orb_intraday import backtest_orb_on_candles  # noqa: PLC0415

    # Use 5 liquid stocks as a representative sample for quick backtest
    sample_symbols = [
        "RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "TCS.NS",
        "SBIN.NS", "AXISBANK.NS", "WIPRO.NS", "LT.NS", "BHARTIARTL.NS",
    ]

    print("\n  Running ORB backtest on yfinance 15-min data (last 60 days) ...")
    all_trades: list[pd.DataFrame] = []

    for sym_yf in sample_symbols:
        try:
            df = yf.download(sym_yf, period="60d", interval="15m", progress=False)
            if df.empty:
                continue
            # Normalise columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() for c in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]
            df.index = pd.to_datetime(df.index)
            # Filter to NSE market hours
            df = df.between_time("09:15", "15:30")
            sym_clean = sym_yf.replace(".NS", "")
            trades = backtest_orb_on_candles(sym_clean, df, capital=capital)
            if not trades.empty:
                all_trades.append(trades)
        except Exception as exc:
            logger.debug("ORB backtest failed for {}: {}", sym_yf, exc)

    if not all_trades:
        return pd.Series(dtype=float)

    combined = pd.concat(all_trades, ignore_index=True)
    combined["trade_date"] = pd.to_datetime(combined["trade_date"])
    daily = combined.groupby("trade_date")["net_pnl"].sum()

    equity = capital
    eq_pts = []
    for dt, pnl in daily.items():
        equity += pnl
        eq_pts.append((dt, equity))

    return pd.Series(
        [e for _, e in eq_pts],
        index=[d for d, _ in eq_pts],
        name="intraday_backtest_equity",
    )


# ── Report printing ────────────────────────────────────────────────────────────

def _print_summary(label: str, eq: pd.Series, capital: float) -> None:
    if eq.empty:
        print(f"\n  {label}: No data")
        return

    start_eq = pd.concat([pd.Series([capital], index=[eq.index[0] - pd.DateOffset(months=1)]), eq])
    years    = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr_val = _cagr(capital, float(eq.iloc[-1]), years)
    dd_val   = _maxdd(start_eq)

    print(f"\n  {label}")
    print(f"  {'─'*50}")
    print(f"  CAGR:   {cagr_val*100:+.1f}%   |   MaxDD: {dd_val*100:.1f}%")
    print(f"  Start:  ₹{capital:,.0f}   →   Final: ₹{float(eq.iloc[-1]):,.0f}")
    print(f"  Period: {eq.index[0].date()} → {eq.index[-1].date()} ({years:.1f} years)")


def _intraday_trade_stats(trades: pd.DataFrame) -> None:
    if trades.empty or trades["net_pnl"].isna().all():
        return
    closed = trades.dropna(subset=["net_pnl"])
    wins   = (closed["net_pnl"] > 0).sum()
    losses = (closed["net_pnl"] <= 0).sum()
    avg_win  = closed.loc[closed["net_pnl"] > 0, "net_pnl"].mean() if wins else 0
    avg_loss = closed.loc[closed["net_pnl"] <= 0, "net_pnl"].mean() if losses else 0
    print(f"\n  ORB Trade Stats:")
    print(f"    Total: {len(closed)} | Win: {wins} ({wins/max(len(closed),1)*100:.0f}%) | Loss: {losses}")
    print(f"    Avg Win: ₹{avg_win:+,.0f} | Avg Loss: ₹{avg_loss:+,.0f} | "
          f"RR: {abs(avg_win/avg_loss):.2f}x" if avg_loss != 0 else "")
    print(f"    Total P&L: ₹{closed['net_pnl'].sum():+,.0f}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Monthly + yearly CAGR for all strategies")
    ap.add_argument("--swing",    action="store_true", help="Show swing strategy only")
    ap.add_argument("--intraday", action="store_true", help="Show intraday strategy only")
    ap.add_argument("--capital-swing",    type=float, default=100_000.0)
    ap.add_argument("--capital-intraday", type=float, default=50_000.0)
    ap.add_argument("--paper-only",       action="store_true",
                    help="Intraday: show paper trades only")
    ap.add_argument("--backtest-intraday", action="store_true",
                    help="Run intraday ORB backtest on yfinance data")
    args = ap.parse_args()

    show_swing    = args.swing    or not (args.swing or args.intraday)
    show_intraday = args.intraday or not (args.swing or args.intraday)

    swing_eq  = pd.Series(dtype=float)
    intra_eq  = pd.Series(dtype=float)

    print("\n" + "=" * 70)
    print("  PIEDPIPER PERFORMANCE REPORT")
    print("  Generated:", date.today())
    print("=" * 70)

    # ── Swing / Monthly Momentum ──────────────────────────────────────────────
    if show_swing:
        print("\n[1] MONTHLY MOMENTUM (Swing) — inv-vol weighted + gold defensive")
        print("    Universe: Nifty 500 | Rebalance: Monthly | IS 2015-2020, OOS 2021-today")
        try:
            print("    Loading swing data ...")
            swing_eq = _get_swing_equity(args.capital_swing)
            _print_summary("Monthly Momentum CAGR", swing_eq, args.capital_swing)

            # Monthly return grid
            if not swing_eq.empty:
                grid = _monthly_returns_grid(swing_eq, args.capital_swing)
                _print_monthly_grid(grid, "Monthly Momentum")
        except Exception as exc:
            logger.error("Swing report failed: {}", exc)
            print(f"    ERROR: {exc}")

    # ── Intraday ORB ──────────────────────────────────────────────────────────
    if show_intraday:
        print("\n[2] INTRADAY ORB — Opening Range Breakout (Nifty 50 stocks)")
        print("    Capital: ₹{:,.0f} | Max 3 trades/day | Auto SQO 3:15 PM".format(
            args.capital_intraday))

        if args.backtest_intraday:
            print("    [BACKTEST on yfinance 15-min data, last 60 days]")
            intra_eq = _run_intraday_backtest(args.capital_intraday)
            label    = "Intraday ORB (60-day backtest)"
        else:
            intra_eq = _get_intraday_equity(args.capital_intraday, args.paper_only)
            label    = "Intraday ORB (live/paper trades)"

        if not intra_eq.empty:
            _print_summary(label, intra_eq, args.capital_intraday)
            # Monthly return grid for intraday
            if not intra_eq.empty:
                grid = _monthly_returns_grid(intra_eq.resample("ME").last().ffill(),
                                             args.capital_intraday)
                _print_monthly_grid(grid, "Intraday ORB")
            # Trade-level stats
            trades = store.load_intraday_trades(paper_only=args.paper_only)
            _intraday_trade_stats(trades)
        else:
            print("\n  No intraday trade data yet.")
            print("  → Start paper trading: python scripts/intraday_run.py")
            print("  → Or run backtest: python scripts/performance_report.py --backtest-intraday")

    # ── Combined ──────────────────────────────────────────────────────────────
    if show_swing and show_intraday and not swing_eq.empty and not intra_eq.empty:
        combined_capital = args.capital_swing + args.capital_intraday
        # Align to common dates and sum equity
        swing_daily  = swing_eq.resample("D").last().ffill()
        intra_daily  = intra_eq.resample("D").last().ffill()
        combined_idx = swing_daily.index.union(intra_daily.index)
        sw = swing_daily.reindex(combined_idx).ffill()
        it = intra_daily.reindex(combined_idx).ffill()
        combined_eq = (sw.fillna(args.capital_swing) + it.fillna(args.capital_intraday))

        print("\n[3] COMBINED PORTFOLIO")
        _print_summary("Combined (Swing + Intraday)", combined_eq, combined_capital)

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
