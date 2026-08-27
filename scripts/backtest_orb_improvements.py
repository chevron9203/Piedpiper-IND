"""
ORB Improvement Tests — all 4 variants in one pass
====================================================
Runs two backtests per symbol (TARGET_MULT=1.5 and 1.0), then derives:

  Baseline  : current settings (1.5x target, 3x vol, 0.3% min range)
  Test A    : Fixed 1.0x target (take profits faster)
  Test B    : 5x vol surge required for large-range stocks (range > 1.5%)
  Test C    : 0.5% minimum range (drop tiny 0.3-0.5% range trades)
  Test D    : Adaptive target — use 1.5x when rolling 30d hit rate >= 22%, else 1.0x
  Test E    : B + C + D combined

Capital: ₹2,00,000 | LONG only | Max 3 signals/day | 2020–2026
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
import pandas as pd
import numpy as np
from datetime import timedelta
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="ERROR")

import strategies.orb_intraday as orb_mod
from strategies.orb_intraday import backtest_orb_on_candles

# ── Config ──────────────────────────────────────────────────────────────────
DB_PATH       = "data_store/piedpiper.duckdb"
CAPITAL       = 200_000.0
MAX_SIGS      = 3
FROM_DATE     = "2020-01-01"
TO_DATE       = "2026-08-26"
IS_END        = 2023

ADAPTIVE_WINDOW   = 30    # days rolling window for hit rate
ADAPTIVE_THRESH   = 0.22  # switch to 1.0x when hit rate < this
LARGE_RANGE_PCT   = 0.015 # range > 1.5% = "large range bucket"
HIGH_VOL_MULT     = 5.0   # vol surge req for large-range (Test B)
MIN_RANGE_NEW     = 0.005 # 0.5% minimum range (Test C)


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
        WHERE symbol='Nifty 50' AND dt >= '{FROM_DATE}' ORDER BY dt
    """).df()
    nifty["dt"] = pd.to_datetime(nifty["dt"])
    conn.close()
    print(f"  {df['symbol'].nunique()} symbols loaded", flush=True)
    return df, nifty


# ── Per-symbol backtest with given target_mult ───────────────────────────────
def run_symbol(sym, sym_df, nc, ne, target_mult: float) -> pd.DataFrame:
    orig = orb_mod.TARGET_MULT
    orb_mod.TARGET_MULT = target_mult
    try:
        t = backtest_orb_on_candles(
            sym, sym_df, capital=CAPITAL,
            nifty_close=nc, nifty_ema20=ne,
            direction="LONG", stock_ema_filter=True,
        )
    finally:
        orb_mod.TARGET_MULT = orig
    return t


# ── Portfolio selection (max 3/day by vol_ratio) ─────────────────────────────
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


# ── Stats printer ─────────────────────────────────────────────────────────────
def stats(trades: pd.DataFrame, label: str, cap: float = CAPITAL):
    if trades.empty:
        print(f"\n  {label}: No trades")
        return

    trades = trades.copy()
    trades["year"] = pd.to_datetime(trades["trade_date"]).dt.year

    n       = len(trades)
    wins    = (trades["net_pnl"] > 0).sum()
    total   = trades["net_pnl"].sum()
    costs   = trades["charges"].sum() if "charges" in trades.columns else 0
    avg_w   = trades.loc[trades["net_pnl"] > 0, "net_pnl"].mean() if wins else 0
    avg_l   = trades.loc[trades["net_pnl"] < 0, "net_pnl"].mean() if (n - wins) else 0

    equity  = cap + trades["net_pnl"].cumsum()
    peak    = equity.cummax()
    max_dd  = ((equity - peak) / peak * 100).min()

    years   = (pd.to_datetime(trades["trade_date"]).max() -
               pd.to_datetime(trades["trade_date"]).min()).days / 365.25
    final   = cap + total
    cagr    = (final / cap) ** (1 / max(years, 0.1)) - 1 if final > 0 else float("nan")

    by_yr   = trades.groupby("year")["net_pnl"].sum()

    print(f"\n{'─'*64}")
    print(f"  {label}")
    print(f"{'─'*64}")
    print(f"  Trades {n} | Win {wins/n:.1%} | RR {abs(avg_w/avg_l):.2f}x | "
          f"Avg win ₹{avg_w:+,.0f} | Avg loss ₹{avg_l:+,.0f}")
    print(f"  Net P&L ₹{total:+,.0f} | Charges ₹{costs:,.0f} | CAGR {cagr:+.1f}% | MaxDD {max_dd:.1f}%")
    print(f"  {'Year':<6} {'P&L':>14}  {'% on ₹2L':>9}  {'vs Baseline':>12}")

    return {"label": label, "total": total, "cagr": cagr, "max_dd": max_dd,
            "win_rate": wins/n, "n": n, "by_yr": by_yr.to_dict()}


def print_year_row(by_yr, baseline_yr, cap):
    for yr in sorted(by_yr.keys()):
        p   = by_yr[yr]
        bp  = baseline_yr.get(yr, 0)
        diff = p - bp
        sign = "+" if diff >= 0 else ""
        print(f"  {yr:<6} ₹{p:>12,.0f}  {p/cap:>+8.0%}   {sign}₹{diff:,.0f}")


# ── Adaptive target logic ─────────────────────────────────────────────────────
def build_adaptive_trades(base15: pd.DataFrame, base10: pd.DataFrame) -> pd.DataFrame:
    """
    For each trade date, compute rolling 30-day hit rate from baseline (1.5x).
    If hit_rate < ADAPTIVE_THRESH → use 1.0x P&L, else use 1.5x P&L.
    """
    if base15.empty or base10.empty:
        return base15

    b15 = base15.copy().sort_values("trade_date").reset_index(drop=True)
    b10 = base10.copy().sort_values("trade_date").reset_index(drop=True)

    # Build date→hit_rate mapping from 1.5x run
    b15["is_target"] = (b15["exit_reason"] == "target").astype(int)
    dates = sorted(b15["trade_date"].dt.date.unique())

    date_hit_rate = {}
    for d in dates:
        window_start = d - timedelta(days=ADAPTIVE_WINDOW)
        prior = b15[b15["trade_date"].dt.date < d]
        prior = prior[prior["trade_date"].dt.date >= window_start]
        if len(prior) >= 5:
            date_hit_rate[d] = prior["is_target"].mean()
        else:
            date_hit_rate[d] = 0.30  # not enough history → assume good, use 1.5x

    # Merge 1.0x P&L into baseline frame
    b10_slim = b10[["symbol", "trade_date", "net_pnl", "charges"]].rename(
        columns={"net_pnl": "pnl_10", "charges": "charges_10"}
    )
    merged = b15.merge(b10_slim, on=["symbol", "trade_date"], how="left")

    # Decide which P&L to use per trade
    def pick_pnl(row):
        d = row["trade_date"].date()
        hr = date_hit_rate.get(d, 0.30)
        if hr < ADAPTIVE_THRESH:
            pnl  = row["pnl_10"] if pd.notna(row["pnl_10"]) else row["net_pnl"]
            cost = row["charges_10"] if pd.notna(row["charges_10"]) else row["charges"]
        else:
            pnl  = row["net_pnl"]
            cost = row["charges"]
        return pd.Series({"net_pnl": pnl, "charges": cost,
                          "hit_rate_used": hr, "mult_used": 1.0 if hr < ADAPTIVE_THRESH else 1.5})

    result = merged.apply(pick_pnl, axis=1)
    adaptive = b15.copy()
    adaptive["net_pnl"] = result["net_pnl"]
    adaptive["charges"] = result["charges"]
    return adaptive


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 64)
    print("  ORB Improvement Tests | ₹2L capital | LONG-only | 2020-2026")
    print("=" * 64)

    df_all, nifty_df = load_data()
    nc = nifty_df.set_index("dt")["close"].sort_index()
    ne = nc.ewm(span=20, adjust=False).mean()
    symbols = df_all["symbol"].unique().tolist()

    # ── Run two backtests per symbol (1.5x and 1.0x target) ──────────────
    print(f"\nRunning per-symbol backtests (2 × {len(symbols)} symbols) ...", flush=True)
    all_15, all_10 = [], []

    for i, sym in enumerate(symbols):
        sym_df = df_all[df_all["symbol"] == sym].copy()
        sym_df = sym_df.set_index("dt").sort_index()[["open", "high", "low", "close", "volume"]]
        if len(sym_df) < 100:
            continue

        t15 = run_symbol(sym, sym_df, nc, ne, target_mult=1.5)
        t10 = run_symbol(sym, sym_df, nc, ne, target_mult=1.0)

        if not t15.empty:
            all_15.append(t15)
        if not t10.empty:
            all_10.append(t10)

        if (i + 1) % 40 == 0:
            print(f"  [{i+1}/{len(symbols)}] ...", flush=True)

    base15_raw = pd.concat(all_15, ignore_index=True)
    base10_raw = pd.concat(all_10, ignore_index=True)
    base15_raw["trade_date"] = pd.to_datetime(base15_raw["trade_date"])
    base10_raw["trade_date"] = pd.to_datetime(base10_raw["trade_date"])

    print(f"  Raw signals: {len(base15_raw)} (1.5x) | {len(base10_raw)} (1.0x)", flush=True)

    # ── Apply max-3/day selection ─────────────────────────────────────────
    baseline = select_top3(base15_raw[base15_raw["direction"] == "LONG"])
    t_a      = select_top3(base10_raw[base10_raw["direction"] == "LONG"])

    # ── Test B: 5x vol requirement for large-range stocks ─────────────────
    # Keep large-range only if vol_ratio >= 5.0
    if "range_pct" in baseline.columns:
        b_filt = baseline[
            (baseline["range_pct"] <= LARGE_RANGE_PCT) |
            (baseline["vol_ratio"] >= HIGH_VOL_MULT)
        ]
    else:
        # range_pct not returned — approximate: avg_win scales with range, proxy via entry_price
        # Fall back: filter on vol_ratio >= 5 overall (conservative)
        b_filt = baseline[baseline["vol_ratio"] >= HIGH_VOL_MULT]
    t_b = b_filt.copy()

    # ── Test C: 0.5% minimum range ────────────────────────────────────────
    if "range_pct" in baseline.columns:
        t_c = baseline[baseline["range_pct"] >= MIN_RANGE_NEW].copy()
    else:
        # Without range_pct we can't filter — skip Test C gracefully
        t_c = pd.DataFrame()

    # ── Test D: Adaptive target ────────────────────────────────────────────
    base15_long = select_top3(base15_raw[base15_raw["direction"] == "LONG"])
    base10_long = select_top3(base10_raw[base10_raw["direction"] == "LONG"])
    t_d = build_adaptive_trades(base15_long, base10_long)

    # ── Test E: B + D combined ─────────────────────────────────────────────
    if "range_pct" in t_d.columns:
        t_e_base = t_d[
            (t_d["range_pct"] <= LARGE_RANGE_PCT) |
            (t_d["vol_ratio"] >= HIGH_VOL_MULT)
        ].copy()
    else:
        t_e_base = t_d[t_d["vol_ratio"] >= HIGH_VOL_MULT].copy()
    t_e = t_e_base

    # ── Print results ──────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    results = {}

    for label, trades in [
        ("BASELINE  (1.5x target, 3x vol, 0.3% min)", baseline),
        ("TEST A    (Fixed 1.0x target — take faster)", t_a),
        ("TEST B    (5x vol for large-range stocks)", t_b),
        ("TEST C    (0.5% min range, drop tiny ranges)", t_c),
        ("TEST D    (Adaptive target — auto-adjust)", t_d),
        ("TEST E    (B + D combined)", t_e),
    ]:
        r = stats(trades, label)
        if r:
            results[label] = r

    # ── Side-by-side year comparison ─────────────────────────────────────
    print(f"\n\n{'='*64}")
    print("  YEAR-BY-YEAR COMPARISON  (₹2L capital)")
    print(f"{'='*64}")

    base_yr = results.get("BASELINE  (1.5x target, 3x vol, 0.3% min)", {}).get("by_yr", {})

    header = f"  {'Year':<6} {'Baseline':>12} {'TestA(1.0x)':>13} {'TestD(Adapt)':>14} {'TestE(B+D)':>12}"
    print(header)
    print(f"  {'─'*62}")

    a_yr = results.get("TEST A    (Fixed 1.0x target — take faster)", {}).get("by_yr", {})
    d_yr = results.get("TEST D    (Adaptive target — auto-adjust)", {}).get("by_yr", {})
    e_yr = results.get("TEST E    (B + D combined)", {}).get("by_yr", {})

    for yr in sorted(base_yr.keys()):
        b  = base_yr.get(yr, 0)
        a  = a_yr.get(yr, 0)
        d  = d_yr.get(yr, 0)
        e  = e_yr.get(yr, 0)
        print(f"  {yr:<6} ₹{b:>10,.0f}  ₹{a:>10,.0f}  ₹{d:>11,.0f}  ₹{e:>10,.0f}")

    print(f"\n  {'TOTAL':<6}", end="")
    for key in ["BASELINE  (1.5x target, 3x vol, 0.3% min)",
                "TEST A    (Fixed 1.0x target — take faster)",
                "TEST D    (Adaptive target — auto-adjust)",
                "TEST E    (B + D combined)"]:
        t = results.get(key, {}).get("total", 0)
        print(f"  ₹{t:>10,.0f}", end="")
    print()

    print(f"\n  {'CAGR':<6}", end="")
    for key in ["BASELINE  (1.5x target, 3x vol, 0.3% min)",
                "TEST A    (Fixed 1.0x target — take faster)",
                "TEST D    (Adaptive target — auto-adjust)",
                "TEST E    (B + D combined)"]:
        c = results.get(key, {}).get("cagr", float("nan"))
        print(f"  {c:>+11.1%}   ", end="")
    print()

    print(f"\n  {'MaxDD':<6}", end="")
    for key in ["BASELINE  (1.5x target, 3x vol, 0.3% min)",
                "TEST A    (Fixed 1.0x target — take faster)",
                "TEST D    (Adaptive target — auto-adjust)",
                "TEST E    (B + D combined)"]:
        d = results.get(key, {}).get("max_dd", float("nan"))
        print(f"  {d:>+11.1f}%   ", end="")
    print()

    print(f"\n{'='*64}")
    print("  Note: Test C skipped if range_pct column not in trade output.")
    print("  Test B uses vol_ratio >= 5.0 as large-range proxy if range_pct unavailable.")


if __name__ == "__main__":
    main()
