"""
NSE-S1 rule-based swing strategy backtest.

Walk-forward:
  IS  2015-01-01 → 2020-12-31
  OOS 2021-01-01 → present

Features:
  - ATR-based stops (2×) and targets (4×), adjusted to T+1 fill price
  - Risk-based position sizing (2% equity, 3% for score=5)
  - 126-day equity SMA filter: if equity < SMA → halve risk (ported from fxpiper)
  - Full Indian cost model (STT, stamp duty, brokerage, slippage)
  - Portfolio caps: 5 positions, 80% deployed, 2 per sector
  - Max hold: 15 trading days

Cash accounting:
  entry: cash -= qty * entry_price + entry_side_costs
  exit:  cash += qty * exit_price  - exit_side_costs
  net_pnl = qty*(exit-entry) - total_round_trip_costs

Usage:
  python scripts/backtest_nse_s1.py
  python scripts/backtest_nse_s1.py --capital 200000 --risk 0.02 --eq-filter 126
  python scripts/backtest_nse_s1.py --eq-filter 0  (disable equity filter)
"""
from __future__ import annotations


import argparse
import collections
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from config.cost_model import per_side_cost
from storage import store
from data.universe import get_sector_map, get_nifty100_symbols
from signals.nse_s1_generator import generate_signals


# ── constants ─────────────────────────────────────────────────────────────────

IS_END_YEAR     = 2020
MAX_POSITIONS   = 5
MAX_DEPLOYED    = 0.80
MAX_PER_SECTOR  = 2
MAX_HOLD_DAYS   = 25      # trading days
BASE_RISK_PCT   = 0.02
SCORE5_MULT     = 1.5     # applied when score=4 (max score in revised 4-gate system)
EQ_FILTER_DAYS  = 126
EQ_FILTER_SCALE = 0.5


# ── data loading ──────────────────────────────────────────────────────────────

def _load_universe_ohlcv(from_dt: date, to_dt: date) -> tuple[dict[str, pd.DataFrame], list[str]]:
    with store.db_conn() as conn:
        symbols = [
            r[0] for r in conn.execute(
                "SELECT DISTINCT symbol FROM adjusted_ohlcv "
                "WHERE symbol NOT LIKE '% %' ORDER BY symbol"
            ).fetchall()
        ]
    ohlcv: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = store.load_adjusted_ohlcv(sym, from_date=from_dt, to_date=to_dt)
        if not df.empty and len(df) >= 200:
            ohlcv[sym] = df
    logger.info("Loaded {} equity symbols", len(ohlcv))
    return ohlcv, list(ohlcv.keys())


def _load_index(symbol: str, from_dt: date, to_dt: date) -> pd.DataFrame:
    df = store.load_adjusted_ohlcv(symbol, from_date=from_dt, to_date=to_dt)
    if df.empty:
        logger.warning("Index '{}' not in DuckDB — run ingest_data.py first", symbol)
    return df


# ── position sizing ───────────────────────────────────────────────────────────

def _size_qty(
    equity: float,
    risk_pct: float,
    entry_price: float,
    stop_dist: float,
    cash: float,
    slot_pct: float = MAX_DEPLOYED / MAX_POSITIONS,  # per-position value cap as fraction of equity
) -> int:
    if stop_dist <= 0 or entry_price <= 0:
        return 0
    qty_risk  = int(equity * risk_pct / stop_dist)
    qty_slot  = int(equity * slot_pct / entry_price)
    qty_cash  = int(cash / entry_price)
    return max(0, min(qty_risk, qty_slot, qty_cash))


# ── exit logic ────────────────────────────────────────────────────────────────

def _exit_on_bar(bar: dict, stop: float, target: float) -> tuple[float | None, str]:
    """Return (exit_price, reason) or (None, '') if no exit triggered."""
    o, h, l = bar["open"], bar["high"], bar["low"]
    if o <= stop:
        return o, "stop_gap"
    if l <= stop:
        return stop, "stop"
    if o >= target:
        return o, "target_gap"
    if h >= target:
        return target, "target"
    return None, ""


# ── metrics ───────────────────────────────────────────────────────────────────

def _metrics(eq: pd.Series, trades: list[dict]) -> dict:
    if eq.empty or len(eq) < 2:
        return {}
    ret = eq.pct_change().dropna()
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.01)
    cagr   = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    sharpe = (ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0.0
    max_dd = ((eq - eq.cummax()) / eq.cummax()).min()
    wins   = [t for t in trades if t["net_pnl"] > 0]
    losses = [t for t in trades if t["net_pnl"] <= 0]
    wr     = len(wins) / len(trades) if trades else 0
    aw     = float(np.mean([t["net_pnl"] for t in wins]))   if wins   else 0.0
    al     = float(np.mean([t["net_pnl"] for t in losses])) if losses else 0.0
    pf     = (sum(t["net_pnl"] for t in wins) / abs(sum(t["net_pnl"] for t in losses))
              if losses and sum(t["net_pnl"] for t in losses) != 0 else float("inf"))
    return dict(cagr=cagr, sharpe=sharpe, max_dd=max_dd, trades=len(trades),
                win_rate=wr, avg_win=aw, avg_loss=al, pf=pf)


def _year_returns(eq: pd.Series) -> dict[int, float]:
    out: dict[int, float] = {}
    for yr in sorted(eq.index.year.unique()):
        s = eq[eq.index.year == yr]
        if len(s) >= 2:
            out[int(yr)] = float(s.iloc[-1] / s.iloc[0] - 1)
    return out


# ── simulation ────────────────────────────────────────────────────────────────

def run_backtest(
    signals: pd.DataFrame,
    ohlcv_dict: dict[str, pd.DataFrame],
    sector_map: dict[str, str],
    nifty100_syms: set[str],
    start_capital: float,
    base_risk_pct: float,
    eq_filter_days: int,
    max_hold_days: int = MAX_HOLD_DAYS,
    sl_mult: float = 2.5,
    tp_mult: float = 5.0,
    trail_mult: float = 0.0,   # 0 = disabled; >0 = ATR trail from daily high
    max_gap_pct: float = 0.0,  # 0 = no filter; 0.01 = skip if gap > 1%
    slot_pct: float = MAX_DEPLOYED / MAX_POSITIONS,  # per-position value cap (default 16%)
) -> tuple[pd.Series, list[dict]]:
    """
    Simulate NSE-S1 trades.

    Signal on day T  →  enter at open of day T+1.
    Stop/target distances preserved from the signal bar's ATR, anchored to entry price.
    """
    if signals.empty:
        return pd.Series(dtype=float), []

    # signals grouped by date for fast lookup
    sig_by_date: dict[pd.Timestamp, list[dict]] = {}
    for _, row in signals.iterrows():
        sig_by_date.setdefault(pd.Timestamp(row["date"]), []).append(row.to_dict())

    all_dates = sorted(set().union(*[df.index for df in ohlcv_dict.values()]))
    all_dates = pd.DatetimeIndex(all_dates)

    def _bar(sym: str, dt: pd.Timestamp) -> dict | None:
        df = ohlcv_dict.get(sym)
        if df is None or dt not in df.index:
            return None
        r = df.loc[dt]
        return {"open": float(r["open"]), "high": float(r["high"]),
                "low":  float(r["low"]),  "close": float(r["close"])}

    cash: float = start_capital
    open_pos: dict[str, dict] = {}   # sym → position
    eq_curve: list[tuple[pd.Timestamp, float]] = []
    trades:   list[dict] = []

    eq_deque: collections.deque = collections.deque(maxlen=max(eq_filter_days, 1))
    vol_scale = 1.0

    for i, dt in enumerate(all_dates):

        # ── 1. Process exits (SL / target / time) ────────────────────────────
        for sym in list(open_pos.keys()):
            pos = open_pos[sym]
            bar = _bar(sym, dt)
            if bar is None:
                continue

            exit_price, reason = _exit_on_bar(bar, pos["stop"], pos["target"])

            if exit_price is None:
                # Breakeven stop: raise to entry when 1R profit reached (intrabar OK)
                stop_dist  = pos["entry_price"] - pos["stop"]
                be_trigger = pos["entry_price"] + stop_dist
                if bar["high"] >= be_trigger:
                    new_stop = max(pos["stop"], pos["entry_price"])
                    if new_stop != pos["stop"]:
                        pos["stop"] = new_stop
                        exit_price, reason = _exit_on_bar(bar, pos["stop"], pos["target"])
                        if exit_price is not None:
                            reason = "be_stop"

            # ATR trail: update EOD only (applies from NEXT bar), never re-checks today
            if exit_price is None and trail_mult > 0:
                new_trail = bar["high"] - pos["trail_dist"]
                pos["stop"] = max(pos["stop"], new_trail)

            days_held = i - pos["entry_i"]
            if exit_price is None and days_held >= max_hold_days:
                exit_price = bar["close"]
                reason = "time"

            if exit_price is not None:
                tier = "nifty100" if sym in nifty100_syms else "midcap100"
                exit_c = per_side_cost(pos["qty"], exit_price, "SELL", tier)
                gross  = pos["qty"] * (exit_price - pos["entry_price"])
                total_c = pos["entry_cost"] + exit_c
                cash   += pos["qty"] * exit_price - exit_c
                trades.append({
                    "symbol":      sym,
                    "entry_date":  pos["entry_date"],
                    "exit_date":   dt,
                    "entry_price": pos["entry_price"],
                    "exit_price":  exit_price,
                    "qty":         pos["qty"],
                    "gross_pnl":   round(gross, 2),
                    "costs":       round(total_c, 2),
                    "net_pnl":     round(gross - total_c, 2),
                    "exit_reason": reason,
                    "score":       pos["score"],
                })
                del open_pos[sym]

        # ── 2. Open new positions from previous day's signals ─────────────────
        if i > 0:
            prev_dt  = all_dates[i - 1]
            # Sort: highest score first only; RS in any direction hurts (tested)
            day_sigs = sorted(
                sig_by_date.get(prev_dt, []),
                key=lambda x: x["score"],
                reverse=True,
            )

            # Compute equity for sizing (using today's open for open positions)
            mtm = sum(
                ((_bar(s, dt) or {}).get("open") or p["entry_price"]) * p["qty"]
                for s, p in open_pos.items()
            )
            equity = cash + mtm

            for sig in day_sigs:
                sym = sig["symbol"]
                if sym in open_pos:
                    continue
                if len(open_pos) >= MAX_POSITIONS:
                    break
                sec = sector_map.get(sym, "Other")
                if sum(1 for s in open_pos
                       if sector_map.get(s, "Other") == sec) >= MAX_PER_SECTOR:
                    continue

                bar = _bar(sym, dt)
                if bar is None:
                    continue
                entry_p = bar["open"]
                if entry_p <= 0:
                    continue

                # Gap filter: skip if open gaps too far above signal close
                sig_close = float(sig["close"])
                if max_gap_pct > 0 and entry_p > sig_close * (1 + max_gap_pct):
                    continue

                # Stop/target: use pre-computed distances from signal (e.g. box floor)
                # when available; otherwise fall back to ATR-multiple (S1 default).
                atr_val = float(sig["atr"])
                if "stop_dist" in sig and pd.notna(sig.get("stop_dist")):
                    raw_stop_dist = float(sig["stop_dist"])
                    raw_tp_dist   = float(sig.get("tp_dist", tp_mult * atr_val))
                    # Anchor to actual fill price (not signal close)
                    scale = entry_p / float(sig["close"]) if float(sig["close"]) > 0 else 1.0
                    stop_dist = raw_stop_dist * scale
                    tp_dist   = raw_tp_dist * scale
                else:
                    stop_dist = sl_mult * atr_val
                    tp_dist   = tp_mult * atr_val
                actual_stop   = entry_p - stop_dist
                actual_target = entry_p + tp_dist
                if actual_stop <= 0 or stop_dist <= 0:
                    continue

                risk_mult = SCORE5_MULT if int(sig["score"]) >= 4 else 1.0
                eff_risk  = base_risk_pct * risk_mult * vol_scale
                qty = _size_qty(equity, eff_risk, entry_p, stop_dist, cash, slot_pct)
                if qty <= 0:
                    continue

                tier = "nifty100" if sym in nifty100_syms else "midcap100"
                entry_c = per_side_cost(qty, entry_p, "BUY", tier)
                outflow = qty * entry_p + entry_c
                if outflow > cash:
                    qty = max(0, int((cash - entry_c) / entry_p))
                    if qty <= 0:
                        continue
                    entry_c = per_side_cost(qty, entry_p, "BUY", tier)
                    outflow = qty * entry_p + entry_c

                cash -= outflow
                open_pos[sym] = {
                    "entry_price": entry_p,
                    "entry_date":  dt,
                    "entry_i":     i,
                    "stop":        actual_stop,
                    "target":      actual_target,
                    "trail_dist":  trail_mult * atr_val,  # 0 when trail_mult==0
                    "qty":         qty,
                    "score":       int(sig["score"]),
                    "entry_cost":  entry_c,
                }

        # ── 3. Mark-to-market ─────────────────────────────────────────────────
        mtm    = sum(
            ((_bar(s, dt) or {}).get("close") or p["entry_price"]) * p["qty"]
            for s, p in open_pos.items()
        )
        equity = cash + mtm
        eq_curve.append((dt, equity))

        # ── 4. Update equity filter ───────────────────────────────────────────
        if eq_filter_days > 0:
            eq_deque.append(equity)
            if len(eq_deque) >= eq_filter_days:
                vol_scale = 1.0 if equity >= float(np.mean(eq_deque)) else EQ_FILTER_SCALE
            else:
                vol_scale = 1.0

    # Close remaining open positions at last bar's close
    last_dt = all_dates[-1]
    for sym, pos in list(open_pos.items()):
        bar = _bar(sym, last_dt)
        ep  = bar["close"] if bar else pos["entry_price"]
        tier   = "nifty100" if sym in nifty100_syms else "midcap100"
        exit_c = per_side_cost(pos["qty"], ep, "SELL", tier)
        gross  = pos["qty"] * (ep - pos["entry_price"])
        total_c = pos["entry_cost"] + exit_c
        trades.append({
            "symbol":      sym,
            "entry_date":  pos["entry_date"],
            "exit_date":   last_dt,
            "entry_price": pos["entry_price"],
            "exit_price":  ep,
            "qty":         pos["qty"],
            "gross_pnl":   round(gross, 2),
            "costs":       round(total_c, 2),
            "net_pnl":     round(gross - total_c, 2),
            "exit_reason": "end_of_backtest",
            "score":       pos["score"],
        })

    eq_series = pd.Series(
        [e for _, e in eq_curve],
        index=pd.DatetimeIndex([d for d, _ in eq_curve]),
        name="equity",
    )
    return eq_series, trades


# ── report ────────────────────────────────────────────────────────────────────

def _print_report(
    eq: pd.Series,
    trades: list[dict],
    start_capital: float,
    is_end: int,
    label: str = "",
) -> None:
    print(f"\n{'='*62}")
    print(f"  NSE-S1 Backtest  {label}")
    print(f"  Capital: ₹{start_capital:,.0f} | {eq.index[0].date()} → {eq.index[-1].date()}")
    print(f"{'='*62}")

    periods = [
        (f"IS  ({eq.index.year.min()}–{is_end})",
         eq[eq.index.year <= is_end],
         [t for t in trades if pd.Timestamp(t["entry_date"]).year <= is_end]),
        (f"OOS ({is_end+1}–{eq.index.year.max()})",
         eq[eq.index.year > is_end],
         [t for t in trades if pd.Timestamp(t["entry_date"]).year > is_end]),
    ]
    for period, peq, ptrades in periods:
        if peq.empty:
            continue
        m = _metrics(peq, ptrades)
        print(f"\n{period}:")
        print(f"  CAGR     : {m.get('cagr', 0)*100:+.1f}%")
        print(f"  MaxDD    : {m.get('max_dd', 0)*100:.1f}%")
        print(f"  Sharpe   : {m.get('sharpe', 0):.2f}")
        print(f"  Trades   : {m.get('trades', 0)}")
        print(f"  Win%     : {m.get('win_rate', 0)*100:.1f}%")
        print(f"  Avg Win  : ₹{m.get('avg_win', 0):,.0f}  |  Avg Loss: ₹{m.get('avg_loss', 0):,.0f}")
        print(f"  P-Factor : {m.get('pf', 0):.2f}")

    print(f"\nYear-by-year net returns:")
    for yr, ret in sorted(_year_returns(eq).items()):
        tag = " [IS]" if yr <= is_end else " [OOS]"
        bar = ("+" if ret >= 0 else "-") * min(int(abs(ret) * 40), 40)
        print(f"  {yr}: {ret*100:+6.1f}%  {bar}{tag}")

    if trades:
        # Exit breakdown with win/loss per exit type
        from collections import defaultdict
        by_type: dict = defaultdict(lambda: {"n": 0, "w": 0, "pnl": 0.0})
        for t in trades:
            er = t["exit_reason"]
            by_type[er]["n"] += 1
            if t["net_pnl"] > 0:
                by_type[er]["w"] += 1
            by_type[er]["pnl"] += t["net_pnl"]
        print(f"\nExit type breakdown:")
        for er, d in sorted(by_type.items()):
            pct = d['w'] / d['n'] * 100 if d['n'] else 0
            print(f"  {er:<16} {d['n']:>4} trades  {pct:>5.1f}% win  ₹{d['pnl']:>8,.0f}")
        print(f"Total net P&L  : ₹{sum(t['net_pnl'] for t in trades):,.0f}")
        print(f"Final equity   : ₹{eq.iloc[-1]:,.0f}")
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="NSE-S1 rule-based swing backtest")
    ap.add_argument("--capital",       type=float, default=settings.STARTING_VIRTUAL_CAPITAL)
    ap.add_argument("--risk",          type=float, default=BASE_RISK_PCT,
                    help="Base risk per trade, fraction (default 0.02 = 2%%)")
    ap.add_argument("--eq-filter",     type=int,   default=EQ_FILTER_DAYS,
                    help="Equity SMA days for risk scaling (0 = off)")
    ap.add_argument("--is-end",        type=int,   default=IS_END_YEAR)
    ap.add_argument("--from-year",     type=int,   default=2015)
    ap.add_argument("--to-year",       type=int,   default=2025)
    ap.add_argument("--vix-threshold", type=float, default=20.0)
    ap.add_argument("--max-hold",      type=int,   default=MAX_HOLD_DAYS,
                    help="Max hold in trading days (default 25)")
    ap.add_argument("--sl-mult",       type=float, default=2.5,
                    help="Stop-loss ATR multiplier (default 2.5)")
    ap.add_argument("--tp-mult",       type=float, default=5.0,
                    help="Take-profit ATR multiplier (default 5.0)")
    # Signal generation tuning
    ap.add_argument("--rsi-low",       type=float, default=55.0)
    ap.add_argument("--rsi-high",      type=float, default=80.0)
    ap.add_argument("--vol-mult",      type=float, default=2.0)
    ap.add_argument("--min-score",     type=int,   default=4)
    ap.add_argument("--bo-window",     type=int,   default=63,
                    help="Breakout lookback window in trading days")
    ap.add_argument("--trail-mult",    type=float, default=0.0,
                    help="ATR trail multiplier (0=disabled, use breakeven stop instead)")
    ap.add_argument("--max-gap",       type=float, default=0.0,
                    help="Skip entry if open gaps > this fraction above signal close (0=off)")
    ap.add_argument("--nifty-ema-mid", type=int,   default=0,
                    help="Additional Nifty EMA filter in G1 (0=off, e.g. 100)")
    ap.add_argument("--slot-pct",      type=float, default=MAX_DEPLOYED / MAX_POSITIONS,
                    help=f"Per-position value cap as fraction of equity (default {MAX_DEPLOYED/MAX_POSITIONS:.2f} = {MAX_DEPLOYED/MAX_POSITIONS*100:.0f}%%)")
    args = ap.parse_args()

    from_dt = date(args.from_year, 1, 1)
    to_dt   = date(args.to_year, 12, 31)

    print(f"\nLoading data {args.from_year}–{args.to_year} ...")
    ohlcv_dict, symbols = _load_universe_ohlcv(from_dt, to_dt)
    nifty_df = _load_index(settings.NIFTY50_SYMBOL,   from_dt, to_dt)
    vix_df   = _load_index(settings.INDIA_VIX_SYMBOL, from_dt, to_dt)

    sector_df     = get_sector_map()
    sector_map    = dict(zip(sector_df["symbol"], sector_df["sector"]))
    nifty100_syms = get_nifty100_symbols()

    if nifty_df.empty:
        print("ERROR: Nifty 50 data missing.  Run: python scripts/ingest_data.py")
        sys.exit(1)
    if vix_df.empty:
        print("WARNING: India VIX data missing — VIX gate disabled (treated as regime-open)")
        vix_df = pd.DataFrame(
            {"open": 0, "high": 0, "low": 0, "close": 0, "volume": 0},
            index=nifty_df.index,
        )

    print(f"Generating signals for {len(symbols)} symbols ...")
    signals = generate_signals(
        ohlcv_dict=ohlcv_dict,
        nifty_df=nifty_df,
        vix_df=vix_df,
        universe_symbols=symbols,
        vix_threshold=args.vix_threshold,
        rsi_low=args.rsi_low,
        rsi_high=args.rsi_high,
        vol_mult=args.vol_mult,
        min_score=args.min_score,
        breakout_window=args.bo_window,
    )
    n_sig = len(signals)
    n_dates = signals["date"].nunique() if not signals.empty else 0
    print(f"  {n_sig} signals across {n_dates} signal dates")

    # Optional secondary Nifty EMA filter applied post-signal-gen
    if args.nifty_ema_mid > 0 and not signals.empty:
        nifty_ema_mid = nifty_df["close"].ewm(span=args.nifty_ema_mid, adjust=False).mean()
        nifty_pass = nifty_df["close"] > nifty_ema_mid
        nifty_pass = nifty_pass.reindex(pd.DatetimeIndex(signals["date"].unique()), method="ffill")
        allowed = set(nifty_pass[nifty_pass].index)
        before = len(signals)
        signals = signals[signals["date"].isin(allowed)].reset_index(drop=True)
        print(f"  Nifty EMA({args.nifty_ema_mid}) filter: {before} → {len(signals)} signals")

    print("Running simulation ...")
    eq_curve, trades = run_backtest(
        signals=signals,
        ohlcv_dict=ohlcv_dict,
        sector_map=sector_map,
        nifty100_syms=nifty100_syms,
        start_capital=args.capital,
        base_risk_pct=args.risk,
        eq_filter_days=args.eq_filter,
        max_hold_days=args.max_hold,
        sl_mult=args.sl_mult,
        tp_mult=args.tp_mult,
        trail_mult=args.trail_mult,
        max_gap_pct=args.max_gap,
        slot_pct=args.slot_pct,
    )

    trail_str = f" trail×{args.trail_mult}" if args.trail_mult > 0 else ""
    gap_str   = f" gap<{args.max_gap*100:.1f}%" if args.max_gap > 0 else ""
    ema_str   = f" EMA_mid={args.nifty_ema_mid}" if args.nifty_ema_mid > 0 else ""
    slot_str  = f" slot={args.slot_pct*100:.0f}%"
    label = (f"| risk={args.risk*100:.0f}% | VIX<{args.vix_threshold:.0f} "
             f"| hold={args.max_hold}d | SL×{args.sl_mult} TP×{args.tp_mult}"
             f"{trail_str}{gap_str}{ema_str}{slot_str}")
    _print_report(eq_curve, trades, args.capital, args.is_end, label)


if __name__ == "__main__":
    main()
