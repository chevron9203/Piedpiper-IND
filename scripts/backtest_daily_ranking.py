#!/usr/bin/python3
"""
backtest_daily_ranking.py

Pure daily ranking-based momentum — NO monthly calendar.

Logic:
  - Every trading day: compute composite score for all universe stocks
  - Signal on day D → execute at open of day D+1 (no look-ahead)
  - ENTRY: stock enters today's top-10 AND a slot is open → BUY next open
  - HOLD:  rank stays <= buffer_zone → keep holding
  - EXIT:  rank drops > buffer_zone for exit_days consecutive days
             AND position has been held >= min_hold_days → SELL next open
  - REGIME: Nifty 50 < 200d EMA → sell everything → park in LIQUIDBEES (6.5% pa)
             Nifty 50 > 200d EMA → resume normal operation

Test matrix (buffer_zone, exit_days, min_hold_days):
  Config A: 12, 3, 5
  Config B: 15, 3, 5   ← primary
  Config C: 20, 3, 5
  Config D: 15, 5, 5
  Config E: 15, 3, 10

Baseline reference (monthly composite + 200d EMA):
  IS (2016-2020):  ~+22% CAGR    OOS (2021-2026): +27.6% CAGR, -13.1% MaxDD
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb

DB_PATH       = Path(__file__).parent.parent / "data_store" / "piedpiper.duckdb"
START_CAPITAL = 300_000.0
MAX_POSITIONS = 10
LIQUID_RATE_PA = 0.065          # 6.5% annualised for LIQUIDBEES proxy
NIFTY_EMA_SPAN = 200
MIN_PRICE      = 100.0          # skip penny stocks
MAX_SCORE_CAP  = 0.80           # exclude pump-and-dump

# Composite windows: (lookback_shift_days, weight)
# score = Σ weight × (close[today] / close[today - shift] − 1)
# Same formula as backtest_momentum_monthly.py COMPOSITE_WINDOWS
COMPOSITE_WINDOWS = [
    (253, 0.40),   # ≈12m
    (127, 0.30),   # ≈6m
    ( 64, 0.20),   # ≈3m
    ( 22, 0.10),   # ≈1m
]
MIN_HISTORY = max(s for s, _ in COMPOSITE_WINDOWS) + 20   # ~273 trading days

# Transaction costs (identical to backtest_momentum_monthly.py)
BROKERAGE_BUY        = 40.0
BROKERAGE_SELL       = 40.0
DP_CHARGE            = 25.50
STT_BUY_PCT          = 0.001
STT_SELL_PCT         = 0.001
EXCHANGE_CHARGE_PCT  = 0.00003


def _trade_cost(value: float, side: str) -> float:
    cost = BROKERAGE_BUY if side == "buy" else BROKERAGE_SELL
    if side == "sell":
        cost += DP_CHARGE
    stt = STT_BUY_PCT if side == "buy" else STT_SELL_PCT
    cost += value * (stt + EXCHANGE_CHARGE_PCT)
    return cost


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    con = duckdb.connect(str(DB_PATH), read_only=True)

    equity = con.execute("""
        SELECT symbol, dt, open, close
        FROM adjusted_ohlcv
        WHERE source = 'eod2'
          AND dt >= '2018-01-01'
          AND open  > 0
          AND close > 0
        ORDER BY symbol, dt
    """).df()

    nifty_df = con.execute("""
        SELECT dt, close
        FROM adjusted_ohlcv
        WHERE symbol = 'Nifty 50'
          AND dt >= '2018-01-01'
        ORDER BY dt
    """).df()
    con.close()

    equity["dt"]   = pd.to_datetime(equity["dt"])
    nifty_df["dt"] = pd.to_datetime(nifty_df["dt"])

    close_wide = equity.pivot(index="dt", columns="symbol", values="close").sort_index()
    open_wide  = equity.pivot(index="dt", columns="symbol", values="open").sort_index()

    nifty_s    = nifty_df.set_index("dt")["close"].sort_index()

    return close_wide, open_wide, nifty_s


# ─────────────────────────────────────────────────────────────────────────────
# Signal computation (vectorised)
# ─────────────────────────────────────────────────────────────────────────────

def compute_scores_and_ranks(
    close_wide: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return composite score and rank DataFrames."""
    composite = pd.DataFrame(0.0, index=close_wide.index, columns=close_wide.columns)
    valid_mask = pd.DataFrame(True, index=close_wide.index, columns=close_wide.columns)

    for shift, weight in COMPOSITE_WINDOWS:
        past = close_wide.shift(shift)
        r = close_wide / past - 1.0
        composite += weight * r
        valid_mask &= past.notna()

    # Apply quality filters
    composite = composite.where(valid_mask)
    composite = composite.where(close_wide > MIN_PRICE)
    composite = composite.where(composite <= MAX_SCORE_CAP)

    # Rank: rank 1 = strongest momentum
    ranks = composite.rank(axis=1, ascending=False, method="min", na_option="bottom")
    return composite, ranks


def compute_regime(nifty_s: pd.Series, all_dates: pd.DatetimeIndex) -> pd.Series:
    """True = Nifty > 200d EMA (bull); False = bear (go to cash)."""
    nifty_aligned = nifty_s.reindex(all_dates, method="ffill")
    ema200 = nifty_aligned.ewm(span=NIFTY_EMA_SPAN, adjust=False).mean()
    return (nifty_aligned > ema200).rename("regime")


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio simulation
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(
    close_wide:   pd.DataFrame,
    open_wide:    pd.DataFrame,
    ranks:        pd.DataFrame,
    regime:       pd.Series,
    buffer_zone:  int   = 15,
    exit_days:    int   = 3,
    min_hold_days: int  = 5,
    max_positions: int  = MAX_POSITIONS,
    start_capital: float = START_CAPITAL,
    oos_start: pd.Timestamp = pd.Timestamp("2021-01-01"),
    oos_end:   pd.Timestamp = pd.Timestamp("2026-08-25"),
) -> tuple[pd.Series, list[dict]]:
    """
    Simulate pure daily ranking portfolio.

    Returns:
        equity_curve : pd.Series  date → total_equity
        trades       : list of trade dicts
    """
    sim_dates = close_wide.index[
        (close_wide.index >= oos_start) & (close_wide.index <= oos_end)
    ].tolist()

    if not sim_dates:
        return pd.Series(dtype=float), []

    # Portfolio: symbol → {qty, entry_price, entry_idx, days_below_buffer}
    portfolio: dict[str, dict] = {}
    cash         = start_capital
    liquid_cash  = 0.0          # regime-exit proceeds earning LIQUID_RATE_PA
    liquid_since = sim_dates[0]

    equity_curve: dict[pd.Timestamp, float] = {}
    trades: list[dict] = []

    prev_regime = bool(regime.get(sim_dates[0], True))

    for i, d in enumerate(sim_dates):
        today_regime = bool(regime.get(d, True))
        today_ranks  = ranks.loc[d]  if d in ranks.index  else pd.Series(dtype=float)
        today_close  = close_wide.loc[d] if d in close_wide.index else pd.Series(dtype=float)

        # ── Accrue LIQUIDBEES proxy return ────────────────────────────────
        if liquid_cash > 0 and i > 0:
            prev_d = sim_dates[i - 1]
            days_elapsed = (d - prev_d).days
            liquid_cash *= (1.0 + LIQUID_RATE_PA / 365) ** days_elapsed

        # ── Update consecutive-days-below-buffer counter ──────────────────
        for sym in portfolio:
            sym_rank = float(today_ranks.get(sym, 9_999))
            if sym_rank > buffer_zone:
                portfolio[sym]["days_below_buffer"] += 1
            else:
                portfolio[sym]["days_below_buffer"] = 0

        # ── Generate signals for execution at D+1 open ───────────────────
        pending_sells: list[str] = []
        pending_buys:  list[str] = []

        if not today_regime:
            # Regime exit: dump all equity, move to liquid
            pending_sells = list(portfolio.keys())

        else:
            # Ranking exits
            for sym, p in portfolio.items():
                hold_days = i - p["entry_idx"]
                if (p["days_below_buffer"] >= exit_days
                        and hold_days >= min_hold_days):
                    pending_sells.append(sym)

            # Buy signals: top-max_positions stocks with open slots
            held_after_sells = set(portfolio.keys()) - set(pending_sells)
            slots_open = max_positions - len(held_after_sells)

            if slots_open > 0:
                # Rank valid stocks (not already held or being sold)
                try:
                    top_stocks = (
                        today_ranks
                        .dropna()
                        .sort_values()
                        .head(max_positions * 2)  # look a bit wider for candidates
                        .index.tolist()
                    )
                except Exception:
                    top_stocks = []

                filled = 0
                for sym in top_stocks:
                    if filled >= slots_open:
                        break
                    sym_rank = float(today_ranks.get(sym, 9_999))
                    if sym_rank > max_positions:
                        break  # sorted → done
                    if sym in held_after_sells:
                        continue
                    pending_buys.append(sym)
                    filled += 1

        # ── Execute at D+1 open ───────────────────────────────────────────
        if i + 1 < len(sim_dates):
            exec_d = sim_dates[i + 1]
            if exec_d not in open_wide.index:
                # Skip if no open prices available (rare holiday gap)
                _record_mtm(equity_curve, d, portfolio, today_close, cash, liquid_cash)
                prev_regime = today_regime
                continue

            exec_opens = open_wide.loc[exec_d]

            # ── Sells ────────────────────────────────────────────────────
            for sym in pending_sells:
                if sym not in portfolio:
                    continue
                px = float(exec_opens.get(sym, today_close.get(sym, np.nan)))
                if pd.isna(px) or px <= 0:
                    continue
                qty   = portfolio[sym]["qty"]
                value = qty * px
                cost  = _trade_cost(value, "sell")
                net   = value - cost

                if not today_regime:
                    liquid_cash  += net
                    liquid_since  = exec_d
                else:
                    cash += net

                trades.append({
                    "date": exec_d, "action": "SELL", "symbol": sym,
                    "qty": qty, "price": px, "value": value, "cost": cost,
                })
                del portfolio[sym]

            # ── Regime ON → resume (move liquid → cash) ───────────────────
            if today_regime and not prev_regime and liquid_cash > 0:
                cash        += liquid_cash
                liquid_cash  = 0.0

            # ── Buys ─────────────────────────────────────────────────────
            if pending_buys and today_regime:
                n_buys = len(pending_buys)
                # Spread available cash equally across pending buys
                per_pos = cash / n_buys if n_buys > 0 else 0

                for sym in pending_buys:
                    px = float(exec_opens.get(sym, np.nan))
                    if pd.isna(px) or px <= 0:
                        continue
                    alloc = min(per_pos, cash)
                    if alloc < 2_000:
                        continue
                    qty = int(alloc / px)
                    if qty == 0:
                        continue
                    cost  = _trade_cost(qty * px, "buy")
                    spend = qty * px + cost
                    if spend > cash + 1:
                        qty   = int((cash - cost - 1) / px)
                        if qty <= 0:
                            continue
                        spend = qty * px + _trade_cost(qty * px, "buy")
                    cash -= spend
                    portfolio[sym] = {
                        "qty":               qty,
                        "entry_price":       px,
                        "entry_idx":         i + 1,
                        "days_below_buffer": 0,
                    }
                    trades.append({
                        "date": exec_d, "action": "BUY", "symbol": sym,
                        "qty": qty, "price": px, "value": qty * px, "cost": cost,
                    })

        # ── MTM equity ────────────────────────────────────────────────────
        _record_mtm(equity_curve, d, portfolio, today_close, cash, liquid_cash)
        prev_regime = today_regime

    equity_s = pd.Series(equity_curve).sort_index()
    return equity_s, trades


def _record_mtm(
    equity_curve: dict,
    d: pd.Timestamp,
    portfolio: dict,
    today_close: pd.Series,
    cash: float,
    liquid_cash: float,
) -> None:
    total = cash + liquid_cash
    for sym, p in portfolio.items():
        px = float(today_close.get(sym, p["entry_price"]))
        if not pd.isna(px) and px > 0:
            total += p["qty"] * px
    equity_curve[d] = total


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def _cagr(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return 0.0
    return (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0


def _max_dd(equity: pd.Series) -> float:
    peak = equity.expanding().max()
    dd   = (equity - peak) / peak
    return float(dd.min())


def _sharpe(equity: pd.Series, rf: float = 0.065) -> float:
    if len(equity) < 20:
        return 0.0
    returns = equity.pct_change().dropna()
    excess  = returns - rf / 252
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(252))


def summarise(label: str, equity: pd.Series, trades: list[dict]) -> None:
    cagr   = _cagr(equity)
    mdd    = _max_dd(equity)
    sharpe = _sharpe(equity)

    buy_trades = [t for t in trades if t["action"] == "BUY"]
    n_trades   = len(buy_trades)

    # Avg hold duration
    hold_days_list = []
    sell_map = {t["symbol"]: [] for t in trades if t["action"] == "SELL"}
    for t in trades:
        if t["action"] == "SELL":
            sell_map[t["symbol"]].append(t["date"])

    trade_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    if not trade_df.empty:
        buys  = trade_df[trade_df["action"] == "BUY"].copy()
        sells = trade_df[trade_df["action"] == "SELL"].copy()
        for _, row in buys.iterrows():
            later_sells = sells[(sells["symbol"] == row["symbol"]) & (sells["date"] > row["date"])]
            if not later_sells.empty:
                hold = (later_sells.iloc[0]["date"] - row["date"]).days
                hold_days_list.append(hold)

    avg_hold = np.mean(hold_days_list) if hold_days_list else 0

    # Trades per month
    if n_trades > 0 and len(equity) > 20:
        months = (equity.index[-1] - equity.index[0]).days / 30.44
        tpm    = n_trades / months
    else:
        tpm = 0.0

    final_equity = equity.iloc[-1] if not equity.empty else START_CAPITAL
    net_pnl      = final_equity - START_CAPITAL

    print(f"\n{'─'*64}")
    print(f"  {label}")
    print(f"{'─'*64}")
    print(f"  CAGR        : {cagr*100:+.1f}%")
    print(f"  MaxDD       : {mdd*100:.1f}%")
    print(f"  Sharpe      : {sharpe:.2f}")
    print(f"  Net P&L     : ₹{net_pnl:,.0f}")
    print(f"  Final equity: ₹{final_equity:,.0f}  (started ₹{START_CAPITAL:,.0f})")
    print(f"  Buy trades  : {n_trades}  ({tpm:.1f}/month)")
    print(f"  Avg hold    : {avg_hold:.0f} calendar days")


def year_by_year(equity: pd.Series) -> None:
    if equity.empty:
        return
    print(f"  {'Year':>6}   {'Start ₹':>12}   {'End ₹':>12}   {'Return':>8}")
    for yr in range(equity.index[0].year, equity.index[-1].year + 1):
        yr_eq = equity[equity.index.year == yr]
        if yr_eq.empty:
            continue
        prev_yr_last = equity[equity.index.year < yr]
        start = float(prev_yr_last.iloc[-1]) if not prev_yr_last.empty else float(yr_eq.iloc[0])
        end   = float(yr_eq.iloc[-1])
        ret   = (end / start - 1) * 100 if start > 0 else 0
        print(f"  {yr:>6}   ₹{start:>10,.0f}   ₹{end:>10,.0f}   {ret:>+7.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    OOS_START = pd.Timestamp("2021-01-01")
    OOS_END   = pd.Timestamp("2026-08-25")

    print("Loading data from DuckDB …")
    close_wide, open_wide, nifty_s = load_data()
    print(f"  Loaded {close_wide.shape[1]} symbols | {len(close_wide)} trading days "
          f"({close_wide.index[0].date()} → {close_wide.index[-1].date()})")

    print("Computing composite scores and ranks …")
    _scores, ranks = compute_scores_and_ranks(close_wide)

    print("Computing Nifty 200d EMA regime …")
    regime = compute_regime(nifty_s, close_wide.index)
    bull_pct = regime[regime.index >= OOS_START].mean() * 100
    print(f"  OOS bull regime: {bull_pct:.0f}% of trading days")

    # ── Test configurations ──────────────────────────────────────────────
    configs = [
        ("Config A  buffer=12 exit=3d min_hold=5d",  12, 3,  5),
        ("Config B  buffer=15 exit=3d min_hold=5d",  15, 3,  5),
        ("Config C  buffer=20 exit=3d min_hold=5d",  20, 3,  5),
        ("Config D  buffer=15 exit=5d min_hold=5d",  15, 5,  5),
        ("Config E  buffer=15 exit=3d min_hold=10d", 15, 3, 10),
    ]

    print("\n" + "═"*64)
    print("  PURE DAILY RANKING BACKTEST — OOS 2021-01-01 → 2026-08-25")
    print(f"  Capital ₹{START_CAPITAL:,.0f} | Max {MAX_POSITIONS} positions")
    print("  Reference: monthly composite +27.6% CAGR / -13.1% MaxDD")
    print("═"*64)

    results = []
    for label, buf, exits, hold in configs:
        print(f"\nRunning: {label} …", end="", flush=True)
        equity, trades = run_simulation(
            close_wide   = close_wide,
            open_wide    = open_wide,
            ranks        = ranks,
            regime       = regime,
            buffer_zone  = buf,
            exit_days    = exits,
            min_hold_days = hold,
            oos_start    = OOS_START,
            oos_end      = OOS_END,
        )
        print(" done")
        summarise(label, equity, trades)
        year_by_year(equity)
        results.append((label, equity, trades))

    # ── Comparison table ────────────────────────────────────────────────
    print("\n" + "═"*64)
    print("  COMPARISON SUMMARY")
    print("═"*64)
    print(f"  {'Config':<45}  {'CAGR':>7}  {'MaxDD':>7}  {'Sharpe':>7}  {'Trades/mo':>10}")
    print(f"  {'─'*45}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*10}")
    # Reference row
    print(f"  {'Monthly baseline (composite+200ema)':<45}  {'+27.6%':>7}  {'-13.1%':>7}  {'~1.8':>7}  {'~2':>10}")
    for label, equity, trades in results:
        cagr   = _cagr(equity)
        mdd    = _max_dd(equity)
        sharpe = _sharpe(equity)
        buys   = [t for t in trades if t["action"] == "BUY"]
        months = (equity.index[-1] - equity.index[0]).days / 30.44 if not equity.empty else 1
        tpm    = len(buys) / months
        short_label = label[:45]
        print(f"  {short_label:<45}  {cagr*100:>+6.1f}%  {mdd*100:>6.1f}%  {sharpe:>7.2f}  {tpm:>9.1f}")


if __name__ == "__main__":
    main()
