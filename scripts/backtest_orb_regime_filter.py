"""
ORB Trend-Strength Filter Test
================================
Run baseline LONG-only ORB once. Then post-filter trades by Nifty
short-term momentum on the trade date. Tests:

  Baseline : Nifty close > EMA20  (current)
  Filter A : + Nifty 5d return > 0%
  Filter B : + Nifty 5d return > 0.5%
  Filter C : + Nifty 5d AND 10d return both > 0%
  Filter D : + Nifty 5d return > 0% AND previous day was up (2-day confirm)
  Filter E : + Nifty 20d return > 0%  (medium-term trend)

Capital ₹2,00,000 | LONG only | Max 3/day | 2020-2026
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
import pandas as pd
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="ERROR")

from strategies.orb_intraday import backtest_orb_on_candles

DB_PATH   = "data_store/piedpiper.duckdb"
CAPITAL   = 200_000.0
MAX_SIGS  = 3
FROM_DATE = "2020-01-01"
TO_DATE   = "2026-08-26"


# ── Load data ────────────────────────────────────────────────────────────────
def load_data():
    print("Loading data ...", flush=True)
    conn = duckdb.connect(DB_PATH, read_only=True)
    df = conn.execute(f"""
        SELECT symbol, dt, open, high, low, close, volume
        FROM intraday_ohlcv
        WHERE dt >= '{FROM_DATE}' AND dt <= '{TO_DATE}'
        ORDER BY symbol, dt
    """).df()
    df["dt"] = pd.to_datetime(df["dt"])

    nifty = conn.execute(f"""
        SELECT dt, close FROM adjusted_ohlcv
        WHERE symbol = 'Nifty 50' AND source = 'eod2_index'
        AND dt >= '2019-06-01'
        ORDER BY dt
    """).df()
    nifty["dt"] = pd.to_datetime(nifty["dt"])
    conn.close()
    print(f"  {df['symbol'].nunique()} symbols | Nifty rows: {len(nifty)}", flush=True)
    return df, nifty


# ── Build Nifty regime table ─────────────────────────────────────────────────
def build_nifty_regime(nifty_df: pd.DataFrame) -> pd.DataFrame:
    n = nifty_df.set_index("dt")["close"].sort_index()
    reg = pd.DataFrame(index=n.index)
    reg["close"]      = n
    reg["ema20"]      = n.ewm(span=20, adjust=False).mean()
    reg["ret_1d"]     = n.pct_change(1)
    reg["ret_5d"]     = n.pct_change(5)
    reg["ret_10d"]    = n.pct_change(10)
    reg["ret_20d"]    = n.pct_change(20)
    reg["above_ema20"]= (reg["close"] > reg["ema20"])
    reg["prev_up"]    = reg["ret_1d"].shift(1) > 0      # yesterday was up
    reg = reg.dropna()
    return reg


# ── Portfolio selection ───────────────────────────────────────────────────────
def select_top3(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades
    trades = trades.copy()
    trades["trade_date"] = pd.to_datetime(trades["trade_date"])
    return (
        trades.sort_values("vol_ratio", ascending=False)
              .groupby("trade_date", group_keys=False)
              .head(MAX_SIGS)
              .sort_values("trade_date")
              .reset_index(drop=True)
    )


# ── Stats ────────────────────────────────────────────────────────────────────
def stats(trades: pd.DataFrame, label: str, cap: float = CAPITAL) -> dict:
    if trades.empty:
        print(f"\n  {label}: No trades")
        return {}

    trades = trades.copy()
    trades["year"] = pd.to_datetime(trades["trade_date"]).dt.year

    n      = len(trades)
    wins   = (trades["net_pnl"] > 0).sum()
    total  = trades["net_pnl"].sum()
    avg_w  = trades.loc[trades["net_pnl"] > 0, "net_pnl"].mean() if wins else 0
    avg_l  = trades.loc[trades["net_pnl"] < 0, "net_pnl"].mean() if (n - wins) else 0

    equity = cap + trades["net_pnl"].cumsum()
    peak   = equity.cummax()
    max_dd = ((equity - peak) / peak * 100).min()

    yrs    = (pd.to_datetime(trades["trade_date"]).max() -
              pd.to_datetime(trades["trade_date"]).min()).days / 365.25
    final  = cap + total
    cagr   = (final / cap) ** (1 / max(yrs, 0.1)) - 1 if final > 0 else float("nan")

    by_yr  = trades.groupby("year")["net_pnl"].sum()

    print(f"\n{'─'*64}")
    print(f"  {label}")
    print(f"{'─'*64}")
    print(f"  Trades {n} | Win {wins/n:.1%} | Avg win ₹{avg_w:+,.0f} | Avg loss ₹{avg_l:+,.0f}")
    print(f"  Net P&L ₹{total:+,.0f} | CAGR {cagr:+.1f}% | MaxDD {max_dd:.1f}%")

    return {"label": label, "n": n, "total": total, "cagr": cagr,
            "max_dd": max_dd, "win_pct": wins/n, "by_yr": by_yr.to_dict()}


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 64)
    print("  ORB Trend-Strength Filter Test | ₹2L | 2020–2026")
    print("=" * 64)

    df_all, nifty_df = load_data()
    regime = build_nifty_regime(nifty_df)

    nc = nifty_df.set_index("dt")["close"].sort_index()
    ne = nc.ewm(span=20, adjust=False).mean()
    symbols = df_all["symbol"].unique().tolist()

    # ── Single baseline run ───────────────────────────────────────────────
    print(f"\nRunning baseline backtest ({len(symbols)} symbols) ...", flush=True)
    all_trades = []
    for i, sym in enumerate(symbols):
        sym_df = df_all[df_all["symbol"] == sym].copy()
        sym_df = sym_df.set_index("dt").sort_index()[["open", "high", "low", "close", "volume"]]
        if len(sym_df) < 100:
            continue
        t = backtest_orb_on_candles(
            sym, sym_df, capital=CAPITAL,
            nifty_close=nc, nifty_ema20=ne,
            direction="LONG", stock_ema_filter=True,
        )
        if not t.empty:
            all_trades.append(t)
        if (i + 1) % 40 == 0:
            print(f"  [{i+1}/{len(symbols)}] ...", flush=True)

    raw = pd.concat(all_trades, ignore_index=True)
    raw["trade_date"] = pd.to_datetime(raw["trade_date"])

    # Select max 3/day — this IS the baseline
    baseline = select_top3(raw[raw["direction"] == "LONG"])

    # ── Join Nifty regime data onto each trade ────────────────────────────
    reg_daily = regime[["ret_5d", "ret_10d", "ret_20d", "ret_1d", "prev_up"]].copy()
    reg_daily.index = pd.to_datetime(reg_daily.index).normalize()

    baseline["trade_day"] = baseline["trade_date"].dt.normalize()
    baseline = baseline.join(reg_daily, on="trade_day", how="left")

    # ── Apply filter variants ─────────────────────────────────────────────
    filters = {
        "BASELINE  current (Nifty > EMA20 only)":
            baseline,

        "FILTER A  + Nifty 5d return > 0%":
            baseline[baseline["ret_5d"] > 0],

        "FILTER B  + Nifty 5d return > 0.5%":
            baseline[baseline["ret_5d"] > 0.005],

        "FILTER C  + Nifty 5d AND 10d both > 0%":
            baseline[(baseline["ret_5d"] > 0) & (baseline["ret_10d"] > 0)],

        "FILTER D  + Nifty 5d > 0% AND prev day up":
            baseline[(baseline["ret_5d"] > 0) & (baseline["prev_up"] == True)],

        "FILTER E  + Nifty 20d return > 0%":
            baseline[baseline["ret_20d"] > 0],
    }

    results = {}
    for label, trades in filters.items():
        r = stats(trades, label)
        if r:
            results[label] = r

    # ── Year-by-year comparison ───────────────────────────────────────────
    print(f"\n\n{'='*64}")
    print("  YEAR-BY-YEAR  (₹2L capital)")
    print(f"{'='*64}")

    keys = list(results.keys())
    short = ["Baseline", "A:5d>0", "B:5d>0.5%", "C:5d+10d", "D:5d+prev", "E:20d>0"]

    header = f"  {'Year':<6}" + "".join(f"  {s:>11}" for s in short)
    print(header)
    print("  " + "─" * 78)

    base_yr = results[keys[0]]["by_yr"] if keys else {}
    all_years = sorted(base_yr.keys())

    for yr in all_years:
        row = f"  {yr:<6}"
        for k in keys:
            p = results[k]["by_yr"].get(yr, 0)
            row += f"  ₹{p:>9,.0f}"
        print(row)

    print("  " + "─" * 78)
    print(f"  {'TOTAL':<6}", end="")
    for k in keys:
        print(f"  ₹{results[k]['total']:>9,.0f}", end="")
    print()

    print(f"  {'CAGR':<6}", end="")
    for k in keys:
        print(f"  {results[k]['cagr']:>+10.1f}%", end="")
    print()

    print(f"  {'MaxDD':<6}", end="")
    for k in keys:
        print(f"  {results[k]['max_dd']:>+10.1f}%", end="")
    print()

    print(f"  {'Trades':<6}", end="")
    for k in keys:
        print(f"  {results[k]['n']:>11,}", end="")
    print()

    print(f"  {'WinRate':<6}", end="")
    for k in keys:
        print(f"  {results[k]['win_pct']:>+10.1%}", end="")
    print()

    print(f"\n{'='*64}")

    # ── Summary insight ───────────────────────────────────────────────────
    if len(results) >= 2:
        base_total = results[keys[0]]["total"]
        print("\n  DELTA vs Baseline:")
        for k in keys[1:]:
            delta = results[k]["total"] - base_total
            pct   = delta / base_total * 100
            yr_2025 = results[k]["by_yr"].get(2025, 0) - results[keys[0]]["by_yr"].get(2025, 0)
            yr_2021 = results[k]["by_yr"].get(2021, 0) - results[keys[0]]["by_yr"].get(2021, 0)
            s = results[k]["label"].split()[0] + " " + results[k]["label"].split()[1]
            print(f"  {s:<14}  Total {delta:>+10,.0f} ({pct:>+.1f}%)  "
                  f"| 2025: {yr_2025:>+9,.0f}  | 2021: {yr_2021:>+9,.0f}")

    print(f"\n{'='*64}")


if __name__ == "__main__":
    main()
