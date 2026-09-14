"""
ORB — HONEST single-account backtest
=====================================
Fixes the three inflators in backtest_orb_equity/improvements:
  1. ONE compounding equity curve (not a sum of fixed-notional bets).
  2. Position size off CURRENT equity; cap concurrent positions + MIS margin.
  3. Slippage on every market fill + gap-through stop fills.

LONG-only (matches the live bull-regime system). 2020-2026, ₹2L, Nifty-200.

Run:  .venv/bin/python scripts/backtest_orb_realistic.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
import numpy as np
import pandas as pd
from loguru import logger
logger.remove(); logger.add(sys.stderr, level="ERROR")

from strategies.orb_intraday import backtest_orb_on_candles, _intraday_charges

DB_PATH   = "data_store/piedpiper.duckdb"
CAPITAL   = 200_000.0
FROM_DATE = "2020-01-01"
TO_DATE   = "2026-08-26"
IS_END    = 2023          # IS 2020-2023, OOS 2024-2026
MAX_PER_DAY = 3
LEVERAGE    = 5.0         # MIS intraday buying power
SLIPPAGE    = 0.0010      # 0.10% adverse per side on market fills


def load():
    print("Loading intraday + Nifty ...", flush=True)
    conn = duckdb.connect(DB_PATH, read_only=True)
    df = conn.execute(f"""
        SELECT symbol, dt, open, high, low, close, volume FROM intraday_ohlcv
        WHERE dt >= '{FROM_DATE}' AND dt <= '{TO_DATE}' ORDER BY symbol, dt
    """).df()
    df["dt"] = pd.to_datetime(df["dt"])
    nifty = conn.execute(f"""
        SELECT dt, close FROM adjusted_ohlcv
        WHERE symbol='Nifty 50' AND dt >= '{FROM_DATE}' ORDER BY dt
    """).df()
    nifty["dt"] = pd.to_datetime(nifty["dt"])
    conn.close()
    nc = nifty.set_index("dt")["close"].sort_index()
    ne = nc.ewm(span=20, adjust=False).mean()
    print(f"  {df['symbol'].nunique()} symbols", flush=True)
    return df, nc, ne


def gen_trades(df, nc, ne, slippage, gap):
    out = []
    syms = df["symbol"].unique().tolist()
    for i, sym in enumerate(syms):
        sd = df[df["symbol"] == sym].set_index("dt").sort_index()[["open","high","low","close","volume"]]
        if len(sd) < 100:
            continue
        t = backtest_orb_on_candles(
            sym, sd, capital=CAPITAL, nifty_close=nc, nifty_ema20=ne,
            direction="LONG", stock_ema_filter=True,
            slippage_pct=slippage, gap_through=gap,
        )
        if not t.empty:
            out.append(t)
        if (i + 1) % 50 == 0:
            print(f"    [{i+1}/{len(syms)}]", flush=True)
    trades = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    if not trades.empty:
        trades = trades[trades["direction"] == "LONG"].copy()
        trades["trade_date"] = pd.to_datetime(trades["trade_date"])
        trades["risk_share"] = trades["entry_price"] - trades["stop_loss"]
        trades = trades[trades["risk_share"] > 0]
    return trades


def simulate(trades, risk_pct):
    """Single compounding account. Returns daily equity series + stats."""
    equity = CAPITAL
    ruin = False
    dates = sorted(trades["trade_date"].unique())
    curve, ntr, wins = {}, 0, 0
    for d in dates:
        day = (trades[trades["trade_date"] == d]
               .sort_values("vol_ratio", ascending=False).head(MAX_PER_DAY))
        used, day_pnl = 0.0, 0.0
        for _, r in day.iterrows():
            qty = int((equity * risk_pct) / r["risk_share"])
            free = equity * LEVERAGE - used
            if qty * r["entry_price"] > free:
                qty = int(free / r["entry_price"]) if r["entry_price"] > 0 else 0
            if qty < 1:
                continue
            gross   = (r["exit_price"] - r["entry_price"]) * qty
            charges = _intraday_charges(r["entry_price"] * qty, r["exit_price"] * qty)
            net     = gross - charges
            day_pnl += net
            used    += qty * r["entry_price"]
            ntr += 1
            wins += 1 if net > 0 else 0
        equity += day_pnl
        if equity <= 0:
            ruin = True
            curve[d] = equity
            break
        curve[d] = equity
    eq = pd.Series(curve).sort_index()
    return eq, ntr, wins, ruin


def _cagr(eq, start):
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.1)
    return (eq.iloc[-1] / start) ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else float("nan")


def _maxdd(eq):
    peak = eq.cummax()
    return ((eq - peak) / peak).min()


def report(label, trades, risk_pct):
    eq, ntr, wins, ruin = simulate(trades, risk_pct)
    if eq.empty:
        print(f"\n{label}: no trades"); return
    is_eq  = eq[eq.index.year <= IS_END]
    oos_eq = eq[eq.index.year >  IS_END]
    print(f"\n{'─'*70}\n  {label}\n{'─'*70}")
    if ruin:
        print("  ⚠️  ACCOUNT BLEW UP (equity hit 0) — strategy is not survivable at this risk")
    print(f"  Trades taken: {ntr} | Win rate: {wins/ntr*100:.1f}%" if ntr else "  no trades")
    print(f"  Final equity: ₹{eq.iloc[-1]:,.0f}  (from ₹{CAPITAL:,.0f})")
    print(f"  Full  : CAGR {_cagr(eq, CAPITAL)*100:+.1f}%  |  MaxDD {_maxdd(eq)*100:.1f}%")
    if not is_eq.empty:
        print(f"  IS20-23 : CAGR {_cagr(is_eq, CAPITAL)*100:+.1f}%  |  MaxDD {_maxdd(is_eq)*100:.1f}%")
    if not oos_eq.empty:
        oos0 = is_eq.iloc[-1] if not is_eq.empty else CAPITAL
        print(f"  OOS24-26: CAGR {_cagr(oos_eq, oos0)*100:+.1f}%  |  MaxDD {_maxdd(oos_eq)*100:.1f}%")
    yr = eq.resample("YE").last()
    prev = CAPITAL
    print("  Year-end equity:")
    for ts, v in yr.items():
        print(f"    {ts.year}: ₹{v:>11,.0f}  ({(v/prev-1)*100:+.1f}%)")
        prev = v


def main():
    df, nc, ne = load()
    print("\nGenerating CLEAN-fill trades (slippage=0, no gap-through) ...", flush=True)
    clean  = gen_trades(df, nc, ne, slippage=0.0,      gap=False)
    print("Generating HONEST-fill trades (0.10%/side + gap-through stops) ...", flush=True)
    honest = gen_trades(df, nc, ne, slippage=SLIPPAGE, gap=True)
    print(f"\n  clean trades: {len(clean)} | honest trades: {len(honest)}")

    print(f"\n{'='*70}\n  ORB — HONEST SINGLE-ACCOUNT RESULTS  (₹2L, LONG-only, compounding)\n{'='*70}")
    report("A. Clean fills, 3% risk/trade  (isolates summing→compounding fix)", clean,  0.03)
    report("B. Honest fills (slip+gap), 3% risk/trade", honest, 0.03)
    report("C. Honest fills, 1.5% risk/trade  (saner sizing)", honest, 0.015)
    report("D. Honest fills, 1.0% risk/trade  (conservative)", honest, 0.010)


if __name__ == "__main__":
    main()
