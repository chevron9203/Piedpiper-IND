#!/usr/bin/python3
"""
Momentum parameter optimisation — monthly rebalance with real data.

Tests all combinations of:
  top_n    : 5, 8, 10, 12  (number of positions)
  sizing   : equal, inverse-vol (60-day realised vol, annualised)

Fixed for all runs:
  - Composite score: 0.4×12m + 0.3×6m + 0.2×3m + 0.1×1m  (same windows as live system)
  - Regime filter  : Nifty 50 > 200d EMA  →  in market; else park in LIQUIDBEES (6.5% pa)
  - Sector cap     : max 3 stocks per sector
  - Price filter   : close > ₹100
  - Momentum cap   : composite score ≤ 80% (exclude pump-and-dump)
  - OOS period     : 2021-01-01 → 2026-07-24  (5.5 years, out-of-sample)
  - IS period      : 2016-01-01 → 2020-12-31  (for context, not decision)
  - Capital        : ₹300,000
  - Fill price     : month-end CLOSE (conservative — open next day would be better)
  - Costs          : same as live system (₹40 brokerage + ₹25.50 DP + 0.1% STT)

Baseline reference (from previous OOS backtest):
  top-10, equal weight, composite+200ema: +27.6% CAGR, -13.1% MaxDD, Sharpe ~1.8
"""

from __future__ import annotations
import sys
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb

DB_PATH        = Path(__file__).parent.parent / "data_store" / "piedpiper.duckdb"
START_CAPITAL  = 300_000.0
NIFTY_EMA_SPAN = 200
MIN_PRICE      = 100.0
MAX_SCORE_CAP  = 0.80
MAX_PER_SECTOR = 3
LIQUID_RATE_PA = 0.065   # 6.5% pa LIQUIDBEES proxy
INV_VOL_LOOKBACK = 60    # trading days for realised vol

COMPOSITE_WINDOWS = [(253, 0.40), (127, 0.30), (64, 0.20), (22, 0.10)]
MIN_HISTORY = max(s for s, _ in COMPOSITE_WINDOWS) + 20  # 273 trading days

BROKERAGE_BUY        = 40.0
BROKERAGE_SELL       = 40.0
DP_CHARGE            = 25.50
STT_BUY_PCT          = 0.001
STT_SELL_PCT         = 0.001
EXCHANGE_CHARGE_PCT  = 0.00003

IS_START  = pd.Timestamp("2016-01-01")
IS_END    = pd.Timestamp("2020-12-31")
OOS_START = pd.Timestamp("2021-01-01")
OOS_END   = pd.Timestamp("2026-07-24")


def _cost(value: float, side: str) -> float:
    c = BROKERAGE_BUY if side == "buy" else BROKERAGE_SELL
    if side == "sell":
        c += DP_CHARGE
    c += value * ((STT_BUY_PCT if side == "buy" else STT_SELL_PCT) + EXCHANGE_CHARGE_PCT)
    return c


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    equity = con.execute("""
        SELECT symbol, dt, close
        FROM adjusted_ohlcv
        WHERE source = 'eod2' AND dt >= '2015-01-01'
        ORDER BY symbol, dt
    """).df()
    nifty_df = con.execute("""
        SELECT dt, close FROM adjusted_ohlcv
        WHERE symbol = 'Nifty 50' AND dt >= '2015-01-01'
        ORDER BY dt
    """).df()
    con.close()

    equity["dt"]   = pd.to_datetime(equity["dt"])
    nifty_df["dt"] = pd.to_datetime(nifty_df["dt"])

    close_wide = equity.pivot(index="dt", columns="symbol", values="close").sort_index()
    nifty_s    = nifty_df.set_index("dt")["close"].sort_index()
    return close_wide, nifty_s


def load_sector_map() -> dict[str, str]:
    """Load sector map from parquet if it exists; return empty dict otherwise."""
    candidates = [
        Path(__file__).parent.parent / "data" / "sector_map.parquet",
        Path(__file__).parent.parent / "data_store" / "sector_map.parquet",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_parquet(p)
            if "symbol" in df.columns and "sector" in df.columns:
                return dict(zip(df["symbol"], df["sector"]))
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Signals
# ─────────────────────────────────────────────────────────────────────────────

def compute_composite(close_wide: pd.DataFrame) -> pd.DataFrame:
    composite = pd.DataFrame(0.0, index=close_wide.index, columns=close_wide.columns)
    valid     = pd.DataFrame(True,  index=close_wide.index, columns=close_wide.columns)
    for shift, weight in COMPOSITE_WINDOWS:
        past   = close_wide.shift(shift)
        r      = close_wide / past - 1.0
        composite += weight * r
        valid &= past.notna()
    composite = composite.where(valid)
    composite = composite.where(close_wide > MIN_PRICE)
    composite = composite.where(composite <= MAX_SCORE_CAP)
    return composite


def compute_regime(nifty_s: pd.Series, all_dates: pd.DatetimeIndex) -> pd.Series:
    nifty_aligned = nifty_s.reindex(all_dates, method="ffill")
    ema200 = nifty_aligned.ewm(span=NIFTY_EMA_SPAN, adjust=False).mean()
    return (nifty_aligned > ema200).rename("regime")


def month_end_dates(all_dates: pd.DatetimeIndex,
                    start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    dates_in_range = all_dates[(all_dates >= start) & (all_dates <= end)]
    if len(dates_in_range) == 0:
        return []
    series = pd.Series(dates_in_range, index=dates_in_range)
    return [v for v in series.resample("ME").last() if pd.notna(v)]


# ─────────────────────────────────────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────────────────────────────────────

def inv_vol_weights(close_wide: pd.DataFrame, symbols: list[str],
                    as_of: pd.Timestamp, lookback: int = INV_VOL_LOOKBACK) -> dict[str, float]:
    """
    Compute inverse-volatility weights for a list of symbols.
    vol = annualised 60-day realised vol of daily returns.
    """
    ivols: dict[str, float] = {}
    for sym in symbols:
        hist = close_wide[sym].dropna().loc[:as_of].tail(lookback)
        if len(hist) < 20:
            ivols[sym] = 1.0
        else:
            daily_vol   = hist.pct_change().dropna().std()
            annual_vol  = max(daily_vol * np.sqrt(252), 0.05)   # 5% floor
            ivols[sym]  = 1.0 / annual_vol

    total = sum(ivols.values())
    if total == 0:
        n = len(symbols)
        return {s: 1.0 / n for s in symbols}
    return {s: v / total for s, v in ivols.items()}


def run_monthly(
    close_wide:   pd.DataFrame,
    composite:    pd.DataFrame,
    regime:       pd.Series,
    sector_map:   dict[str, str],
    top_n:        int   = 10,
    use_inv_vol:  bool  = False,
    start_capital: float = START_CAPITAL,
    sim_start:    pd.Timestamp = OOS_START,
    sim_end:      pd.Timestamp = OOS_END,
) -> tuple[pd.Series, list[dict]]:
    """
    Monthly momentum simulation.
    Returns (equity_curve, trade_log).
    Fills at month-end CLOSE price (conservative).
    """
    all_dates  = close_wide.index
    reb_dates  = month_end_dates(all_dates, sim_start, sim_end)

    if not reb_dates:
        return pd.Series(dtype=float), []

    portfolio: dict[str, dict] = {}   # {sym: {qty, entry_price}}
    cash         = start_capital
    liquid_cash  = 0.0
    equity_pts   = {}
    trades: list[dict] = []

    prev_in_market = True

    for reb_ts in reb_dates:
        today_regime  = bool(regime.get(reb_ts, True))
        today_scores  = composite.loc[reb_ts] if reb_ts in composite.index else pd.Series(dtype=float)
        today_close   = close_wide.loc[reb_ts] if reb_ts in close_wide.index else pd.Series(dtype=float)

        # Accrue liquid return from last rebalance
        if liquid_cash > 0:
            prev_idx = [d for d in reb_dates if d < reb_ts]
            if prev_idx:
                days = (reb_ts - prev_idx[-1]).days
                liquid_cash *= (1 + LIQUID_RATE_PA / 365) ** days

        # ── Select target portfolio ──────────────────────────────────────
        target: list[str] = []
        if today_regime:
            scored = today_scores.dropna().sort_values(ascending=False)
            sector_count: dict[str, int] = {}
            for sym in scored.index:
                if len(target) >= top_n:
                    break
                sec = sector_map.get(sym, "Other")
                if sector_count.get(sec, 0) >= MAX_PER_SECTOR:
                    continue
                target.append(sym)
                sector_count[sec] = sector_count.get(sec, 0) + 1

        # ── Exits ────────────────────────────────────────────────────────
        to_sell = [s for s in portfolio if s not in target]
        for sym in to_sell:
            px = float(today_close.get(sym, portfolio[sym]["entry_price"]))
            if pd.isna(px) or px <= 0:
                continue
            qty   = portfolio[sym]["qty"]
            value = qty * px
            cost  = _cost(value, "sell")
            if today_regime:
                cash       += value - cost
            else:
                liquid_cash += value - cost
            trades.append({"date": reb_ts, "action": "SELL", "symbol": sym,
                           "qty": qty, "price": px})
            del portfolio[sym]

        # Regime exit: convert remaining cash to liquid
        if not today_regime and prev_in_market:
            liquid_cash += cash
            cash = 0.0

        # Regime re-entry: convert liquid to cash
        if today_regime and not prev_in_market and liquid_cash > 0:
            cash        += liquid_cash
            liquid_cash  = 0.0

        # ── Entries ──────────────────────────────────────────────────────
        to_buy = [s for s in target if s not in portfolio]
        if to_buy and today_regime:
            # Compute weights
            all_target = list(portfolio.keys()) + to_buy
            if use_inv_vol:
                weights = inv_vol_weights(close_wide, all_target, reb_ts)
            else:
                n = len(all_target)
                weights = {s: 1.0 / n for s in all_target}

            # Total equity to rebalance (current positions + available cash)
            port_value = cash
            for sym, p in portfolio.items():
                px_now = float(today_close.get(sym, p["entry_price"]))
                port_value += p["qty"] * px_now if not pd.isna(px_now) else p["qty"] * p["entry_price"]

            for sym in to_buy:
                w   = weights.get(sym, 1.0 / len(all_target))
                px  = float(today_close.get(sym, np.nan))
                if pd.isna(px) or px <= 0:
                    continue
                alloc = port_value * w
                qty   = int(alloc / px)
                if qty == 0:
                    continue
                cost  = _cost(qty * px, "buy")
                spend = qty * px + cost
                if spend > cash + 1:
                    qty = int((cash - cost - 1) / px) if cash > cost + px else 0
                    if qty <= 0:
                        continue
                    spend = qty * px + _cost(qty * px, "buy")
                cash -= spend
                portfolio[sym] = {"qty": qty, "entry_price": px}
                trades.append({"date": reb_ts, "action": "BUY", "symbol": sym,
                               "qty": qty, "price": px})

        # ── MTM equity ────────────────────────────────────────────────────
        total = cash + liquid_cash
        for sym, p in portfolio.items():
            px = float(today_close.get(sym, p["entry_price"]))
            if not pd.isna(px):
                total += p["qty"] * px
        equity_pts[reb_ts] = total
        prev_in_market = today_regime

    equity_s = pd.Series(equity_pts).sort_index()
    return equity_s, trades


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def _cagr(eq: pd.Series) -> float:
    if len(eq) < 2:
        return 0.0
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    return (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1.0 if yrs > 0 else 0.0


def _maxdd(eq: pd.Series) -> float:
    peak = eq.expanding().max()
    return float(((eq - peak) / peak).min())


def _sharpe(eq: pd.Series, rf: float = 0.065) -> float:
    if len(eq) < 4:
        return 0.0
    # Use monthly returns for Sharpe (equity sampled monthly)
    rets   = eq.pct_change().dropna()
    excess = rets - rf / 12
    return float(excess.mean() / excess.std() * np.sqrt(12)) if excess.std() > 0 else 0.0


def summarise(label: str, eq: pd.Series, trades: list[dict]) -> dict:
    cagr   = _cagr(eq)
    mdd    = _maxdd(eq)
    sharpe = _sharpe(eq)
    net    = eq.iloc[-1] - START_CAPITAL
    buys   = [t for t in trades if t["action"] == "BUY"]
    months = (eq.index[-1] - eq.index[0]).days / 30.44 if not eq.empty else 1
    tpm    = len(buys) / months
    return {
        "label": label, "cagr": cagr, "mdd": mdd, "sharpe": sharpe,
        "net_pnl": net, "trades_per_month": tpm, "equity": eq, "trades": trades,
    }


def year_row(eq: pd.Series) -> str:
    parts = []
    for yr in range(eq.index[0].year, eq.index[-1].year + 1):
        yr_eq  = eq[eq.index.year == yr]
        if yr_eq.empty:
            continue
        prev   = eq[eq.index.year < yr]
        start  = float(prev.iloc[-1]) if not prev.empty else float(yr_eq.iloc[0])
        end    = float(yr_eq.iloc[-1])
        ret    = (end / start - 1) * 100 if start > 0 else 0
        parts.append(f"{yr}:{ret:+.0f}%")
    return "  ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading data …")
    close_wide, nifty_s = load_data()
    sector_map = load_sector_map()
    print(f"  {close_wide.shape[1]} symbols | {len(close_wide)} trading days "
          f"| sector map: {len(sector_map)} stocks")

    print("Computing composite scores …")
    composite = compute_composite(close_wide)

    print("Computing 200d EMA regime …")
    regime    = compute_regime(nifty_s, close_wide.index)

    # OOS regime summary
    oos_regime = regime[(regime.index >= OOS_START) & (regime.index <= OOS_END)]
    print(f"  OOS bull market: {oos_regime.mean()*100:.0f}% of month-ends\n")

    # ── Test matrix ──────────────────────────────────────────────────────────
    configs = [
        (5,  False, "Top-5   equal-weight"),
        (8,  False, "Top-8   equal-weight"),
        (10, False, "Top-10  equal-weight  ← CURRENT LIVE"),
        (12, False, "Top-12  equal-weight"),
        (5,  True,  "Top-5   inv-vol"),
        (8,  True,  "Top-8   inv-vol"),
        (10, True,  "Top-10  inv-vol"),
        (12, True,  "Top-12  inv-vol"),
    ]

    print("═"*80)
    print("  MONTHLY MOMENTUM PARAMETER OPTIMISATION")
    print(f"  OOS: {OOS_START.date()} → {OOS_END.date()}   Capital ₹{START_CAPITAL:,.0f}")
    print("  Composite score | Nifty 200d EMA regime | sector cap 3 | costs included")
    print("═"*80)

    results = []
    for top_n, inv_vol, label in configs:
        eq, trades = run_monthly(
            close_wide=close_wide,
            composite=composite,
            regime=regime,
            sector_map=sector_map,
            top_n=top_n,
            use_inv_vol=inv_vol,
            sim_start=OOS_START,
            sim_end=OOS_END,
        )
        r = summarise(label, eq, trades)
        results.append(r)
        print(f"\n  {label}")
        print(f"  CAGR {r['cagr']*100:+.1f}%  MaxDD {r['mdd']*100:.1f}%  "
              f"Sharpe {r['sharpe']:.2f}  Trades/mo {r['trades_per_month']:.1f}")
        print(f"  {year_row(eq)}")

    # ── Comparison table ─────────────────────────────────────────────────────
    print("\n" + "═"*80)
    print("  COMPARISON TABLE")
    print("═"*80)
    print(f"  {'Config':<40}  {'CAGR':>7}  {'MaxDD':>7}  {'Sharpe':>7}  {'Trades/mo':>10}")
    print(f"  {'─'*40}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*10}")
    # Previous baseline reference
    print(f"  {'Prev OOS baseline (reported)':<40}  {'+27.6%':>7}  {'-13.1%':>7}  {'~1.8':>7}  {'~2':>10}")
    for r in results:
        marker = " ◄" if "CURRENT" in r["label"] else ""
        print(f"  {r['label']:<40}  {r['cagr']*100:>+6.1f}%  {r['mdd']*100:>6.1f}%  "
              f"{r['sharpe']:>7.2f}  {r['trades_per_month']:>9.1f}{marker}")

    # ── Best by metric ────────────────────────────────────────────────────────
    best_cagr   = max(results, key=lambda x: x["cagr"])
    best_sharpe = max(results, key=lambda x: x["sharpe"])
    best_mdd    = min(results, key=lambda x: x["mdd"])   # least negative = best

    print("\n" + "═"*80)
    print("  WINNERS")
    print("═"*80)
    print(f"  Best CAGR  : {best_cagr['label']}  → {best_cagr['cagr']*100:+.1f}%")
    print(f"  Best Sharpe: {best_sharpe['label']}  → {best_sharpe['sharpe']:.2f}")
    print(f"  Best MaxDD : {best_mdd['label']}  → {best_mdd['mdd']*100:.1f}%")

    # ── Recommendation ────────────────────────────────────────────────────────
    print("\n" + "═"*80)
    print("  YEAR-BY-YEAR FOR EACH CONFIG")
    print("═"*80)
    for r in results:
        print(f"  {r['label']}")
        eq = r["equity"]
        print(f"  {'Year':>6}  {'Start ₹':>12}  {'End ₹':>12}  {'Return':>8}")
        for yr in range(eq.index[0].year, eq.index[-1].year + 1):
            yr_eq = eq[eq.index.year == yr]
            if yr_eq.empty:
                continue
            prev  = eq[eq.index.year < yr]
            start = float(prev.iloc[-1]) if not prev.empty else float(yr_eq.iloc[0])
            end   = float(yr_eq.iloc[-1])
            ret   = (end / start - 1) * 100 if start > 0 else 0
            print(f"  {yr:>6}  ₹{start:>10,.0f}  ₹{end:>10,.0f}  {ret:>+7.1f}%")
        print()


if __name__ == "__main__":
    main()
