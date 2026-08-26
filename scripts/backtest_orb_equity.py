"""
Equity ORB Portfolio Backtest — 2020-2026
==========================================
Uses the same logic as the live intraday_run.py + strategies/orb_intraday.py:
  - 15-min opening range (9:15-9:30 bar)
  - Volume surge 3x 20-day avg opening-bar vol
  - Range 0.3-2.5%, skip 1-1.5% bucket
  - Stock EMA20 alignment filter
  - Nifty regime: LONG on bull days, SHORT on bear days
  - 2-tier trailing stop
  - Max 3 signals/day (top by vol_ratio)
  - Capital: ₹50,000 intraday | 3% risk per trade = ₹1,500 max risk/trade

Run:
  /usr/bin/python3 scripts/backtest_orb_equity.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
import pandas as pd
import numpy as np
from datetime import date
from loguru import logger

from strategies.orb_intraday import backtest_orb_on_candles

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH        = "data_store/piedpiper.duckdb"
CAPITAL        = 50_000.0    # intraday capital
MAX_SIGNALS    = 3           # max trades per day
FROM_DATE      = "2020-01-01"
TO_DATE        = "2026-08-26"
IS_END_YEAR    = 2023        # IS: 2020-2022, OOS: 2023-2026

logger.remove()
logger.add(sys.stderr, level="WARNING")   # suppress per-symbol INFO noise

def load_data():
    print(f"Loading intraday candles ({FROM_DATE} → {TO_DATE}) ...")
    conn = duckdb.connect(DB_PATH, read_only=True)

    # All 15-min equity candles
    df = conn.execute(f"""
        SELECT symbol, dt, open, high, low, close, volume
        FROM intraday_ohlcv
        WHERE dt >= '{FROM_DATE}' AND dt <= '{TO_DATE}'
        ORDER BY symbol, dt
    """).df()
    df["dt"] = pd.to_datetime(df["dt"])

    # Nifty daily close for regime filter
    nifty = conn.execute(f"""
        SELECT dt, close FROM adjusted_ohlcv
        WHERE symbol = 'Nifty 50' AND dt >= '{FROM_DATE}'
        ORDER BY dt
    """).df()
    nifty["dt"] = pd.to_datetime(nifty["dt"])

    conn.close()
    symbols = df["symbol"].unique().tolist()
    print(f"  {len(symbols)} symbols | {df['dt'].min().date()} → {df['dt'].max().date()}")
    return df, nifty, symbols


def run_portfolio_backtest(df_all, nifty_df, symbols):
    """
    Run per-symbol backtest, then apply max 3/day selection by vol_ratio.
    Returns a trades DataFrame with all selected trades.
    """
    nifty_close = nifty_df.set_index("dt")["close"].sort_index()
    nifty_ema20 = nifty_close.ewm(span=20, adjust=False).mean()

    all_trades = []
    total = len(symbols)

    for i, sym in enumerate(symbols):
        sym_df = df_all[df_all["symbol"] == sym].copy()
        sym_df = sym_df.set_index("dt").sort_index()
        sym_df = sym_df[["open", "high", "low", "close", "volume"]]

        if len(sym_df) < 100:
            continue

        # LONG on bull days
        long_t = backtest_orb_on_candles(
            sym, sym_df, capital=CAPITAL,
            nifty_close=nifty_close, nifty_ema20=nifty_ema20,
            direction="LONG", stock_ema_filter=True,
        )
        # SHORT on bear days
        short_t = backtest_orb_on_candles(
            sym, sym_df, capital=CAPITAL,
            nifty_close=nifty_close, nifty_ema20=nifty_ema20,
            direction="SHORT", stock_ema_filter=True,
        )

        for t in [long_t, short_t]:
            if not t.empty:
                all_trades.append(t)

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{total}] processed ...")

    if not all_trades:
        print("No trades generated.")
        return pd.DataFrame()

    trades = pd.concat(all_trades, ignore_index=True)
    trades["trade_date"] = pd.to_datetime(trades["trade_date"])

    # Apply max 3/day constraint: keep top 3 by vol_ratio per day
    selected = (
        trades.sort_values("vol_ratio", ascending=False)
              .groupby("trade_date", group_keys=False)
              .head(MAX_SIGNALS)
    )
    print(f"\n  Total signals: {len(trades)} | After top-{MAX_SIGNALS}/day: {len(selected)}")
    return selected.sort_values("trade_date").reset_index(drop=True)


def print_stats(trades, label, capital):
    if trades.empty:
        print(f"  {label}: No trades")
        return

    trades = trades.copy()
    trades["year"] = pd.to_datetime(trades["trade_date"]).dt.year

    total_pnl  = trades["net_pnl"].sum()
    n_trades   = len(trades)
    wins       = (trades["net_pnl"] > 0).sum()
    win_rate   = wins / n_trades * 100 if n_trades else 0
    avg_win    = trades.loc[trades["net_pnl"] > 0, "net_pnl"].mean() if wins else 0
    avg_loss   = trades.loc[trades["net_pnl"] < 0, "net_pnl"].mean() if (n_trades-wins) else 0
    total_cost = trades["charges"].sum()

    # Equity curve (add to starting capital)
    equity = capital + trades["net_pnl"].cumsum()
    peak   = equity.cummax()
    dd     = ((equity - peak) / peak * 100)
    max_dd = dd.min()

    years  = (pd.to_datetime(trades["trade_date"]).max() -
              pd.to_datetime(trades["trade_date"]).min()).days / 365.25
    final  = capital + total_pnl
    cagr   = (final / capital) ** (1 / max(years, 0.1)) - 1 if final > 0 else float("nan")

    print(f"\n{'═'*60}")
    print(f"  {label}")
    print(f"{'═'*60}")
    print(f"  Trades   : {n_trades} | Win rate: {win_rate:.1f}%")
    print(f"  Net P&L  : ₹{total_pnl:+,.0f} | Total cost: ₹{total_cost:,.0f}")
    print(f"  Avg win  : ₹{avg_win:+,.0f} | Avg loss: ₹{avg_loss:+,.0f}")
    print(f"  CAGR     : {cagr:+.1f}%  (on ₹{capital:,.0f})")
    print(f"  Max DD   : {max_dd:.1f}%")

    by_reason = trades.groupby("exit_reason")["net_pnl"].agg(["count", "sum", "mean"])
    print(f"\n  Exit breakdown:")
    for reason, row in by_reason.iterrows():
        print(f"    {reason:<12}: {int(row['count']):>4} trades  avg ₹{row['mean']:+,.0f}  total ₹{row['sum']:+,.0f}")

    print(f"\n  Year-by-year (top-{MAX_SIGNALS}/day, ₹{CAPITAL:,.0f} capital):")
    by_year = trades.groupby("year")["net_pnl"].sum()
    for yr, pnl in by_year.items():
        bar = "█" * min(25, int(abs(pnl) / 2000))
        sign = "+" if pnl >= 0 else ""
        print(f"    {yr}: ₹{sign}{pnl:>10,.0f}  {bar}")

    # IS / OOS split
    is_t  = trades[trades["year"] <= IS_END_YEAR]
    oos_t = trades[trades["year"] >  IS_END_YEAR]

    def cagr_slice(t, cap, label2):
        if t.empty: return
        pnl_s = t["net_pnl"].sum()
        yrs   = (pd.to_datetime(t["trade_date"]).max() -
                 pd.to_datetime(t["trade_date"]).min()).days / 365.25
        fin   = cap + pnl_s
        c     = (fin / cap) ** (1/max(yrs, 0.1)) - 1 if fin > 0 else float("nan")
        print(f"  {label2}: CAGR {c:+.1f}%  total P&L ₹{pnl_s:+,.0f}  ({len(t)} trades)")

    print()
    cagr_slice(is_t,  capital, f"IS  (2020–{IS_END_YEAR})")
    cagr_slice(oos_t, capital, f"OOS ({IS_END_YEAR+1}–2026)")


def main():
    print("=" * 60)
    print("  Equity ORB Portfolio Backtest (2020–2026)")
    print("  15-min bars | max 3 signals/day | ₹50K capital")
    print("=" * 60)

    df_all, nifty_df, symbols = load_data()
    print(f"\nRunning per-symbol backtest ({len(symbols)} symbols) ...")
    trades = run_portfolio_backtest(df_all, nifty_df, symbols)

    if trades.empty:
        print("No trades to analyse.")
        return

    # Overall
    print_stats(trades, "ORB LONG + SHORT combined", CAPITAL)

    # Split by direction
    long_t  = trades[trades["direction"] == "LONG"]
    short_t = trades[trades["direction"] == "SHORT"]
    if not long_t.empty:
        print_stats(long_t,  "LONG only  (bull regime days)", CAPITAL)
    if not short_t.empty:
        print_stats(short_t, "SHORT only (bear regime days)", CAPITAL)

    # Best / worst symbols
    by_sym = trades.groupby("symbol")["net_pnl"].agg(["sum", "count"]).sort_values("sum", ascending=False)
    print(f"\n  Top-10 symbols by P&L:")
    for sym, row in by_sym.head(10).iterrows():
        print(f"    {sym:<15}: ₹{row['sum']:+,.0f}  ({int(row['count'])} trades)")
    print(f"  Bottom-5 symbols:")
    for sym, row in by_sym.tail(5).iterrows():
        print(f"    {sym:<15}: ₹{row['sum']:+,.0f}  ({int(row['count'])} trades)")


if __name__ == "__main__":
    main()
