"""
NSE-S1 daily forward-test runner.

Run after market close (after 15:30 IST) once EOD2 data is updated.
Generates signals for today → print tomorrow's entry candidates.
Also checks any paper positions from Ledger A for exit conditions.

Usage:
  python scripts/forward_test_nse_s1.py
  python scripts/forward_test_nse_s1.py --dry-run   (print only, don't save to DB)
  python scripts/forward_test_nse_s1.py --capital 200000
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
from config.cost_model import per_side_cost
from storage import store
from data.universe import get_sector_map, get_nifty100_symbols
from signals.nse_s1_generator import generate_signals


MODEL_VERSION = "nse_s1_v1"


# ── data helpers ──────────────────────────────────────────────────────────────

def _load_recent_ohlcv(lookback_days: int = 300) -> dict[str, pd.DataFrame]:
    from_dt = date.today() - timedelta(days=lookback_days)
    with store.db_conn() as conn:
        symbols = [
            r[0] for r in conn.execute(
                "SELECT DISTINCT symbol FROM adjusted_ohlcv "
                "WHERE symbol NOT LIKE '% %' ORDER BY symbol"
            ).fetchall()
        ]
    ohlcv: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = store.load_adjusted_ohlcv(sym, from_date=from_dt)
        if not df.empty and len(df) >= 100:
            ohlcv[sym] = df
    return ohlcv


def _load_index_recent(symbol: str, lookback_days: int = 300) -> pd.DataFrame:
    from_dt = date.today() - timedelta(days=lookback_days)
    return store.load_adjusted_ohlcv(symbol, from_date=from_dt)


def _load_open_paper_positions() -> pd.DataFrame:
    """Load all open paper positions from Ledger A."""
    with store.db_conn() as conn:
        df = conn.execute(
            "SELECT * FROM ledger_a WHERE status = 'open'"
        ).df()
    return df


# ── signal → DB ───────────────────────────────────────────────────────────────

def _save_today_signals(today_sigs: pd.DataFrame, signal_date: date) -> None:
    """Persist today's NSE-S1 signals to the signals table."""
    rows = []
    for _, r in today_sigs.iterrows():
        rows.append({
            "dt":           signal_date,
            "symbol":       r["symbol"],
            "direction":    "long",
            "confidence":   float(r["score"]) / 5.0,   # normalise score 4-5 → 0.8-1.0
            "entry_low":    float(r["close"]),          # today's close = reference entry
            "entry_high":   float(r["close"]) * 1.005, # 0.5% gap tolerance
            "stop_loss":    float(r["stop_loss"]),
            "target":       float(r["target"]),
            "max_hold_days": 15,
            "position_size": None,
            "model_version": MODEL_VERSION,
        })
    store.save_signals(rows)
    logger.info("Saved {} NSE-S1 signals to DB for {}", len(rows), signal_date)


# ── exit check ────────────────────────────────────────────────────────────────

def _check_exits(
    open_positions: pd.DataFrame,
    ohlcv_dict: dict[str, pd.DataFrame],
    today: date,
    nifty100_syms: set[str],
) -> None:
    """Print exit alerts for any open paper positions."""
    if open_positions.empty:
        return

    today_ts = pd.Timestamp(today)
    alerts = []

    for _, pos in open_positions.iterrows():
        sym = pos["symbol"]
        df  = ohlcv_dict.get(sym)
        if df is None or today_ts not in df.index:
            continue

        bar  = df.loc[today_ts]
        stop = float(pos["stop_loss"])
        tgt  = float(pos["target"])
        o, h, l, c = float(bar["open"]), float(bar["high"]), float(bar["low"]), float(bar["close"])

        if o <= stop:
            alerts.append(f"  STOP-GAP  {sym}: exit at open {o:.2f} (stop {stop:.2f})")
        elif l <= stop:
            alerts.append(f"  STOP      {sym}: stop hit at {stop:.2f} (low {l:.2f})")
        elif o >= tgt:
            alerts.append(f"  TARGET-GAP {sym}: exit at open {o:.2f} (target {tgt:.2f})")
        elif h >= tgt:
            alerts.append(f"  TARGET    {sym}: target hit at {tgt:.2f} (high {h:.2f})")

        # Days held
        entry_dt = pd.Timestamp(pos["entry_date"])
        sym_idx  = df.index
        days_held = len(sym_idx[(sym_idx >= entry_dt) & (sym_idx <= today_ts)])
        if days_held >= 15:
            alerts.append(f"  TIME-EXIT {sym}: {days_held} trading days held, exit at close {c:.2f}")

    if alerts:
        print("\n** EXIT ALERTS (act at tomorrow's open) **")
        for a in alerts:
            print(a)
    else:
        print("\nNo exit alerts for open positions today.")


# ── print signals ─────────────────────────────────────────────────────────────

def _print_signals(
    sigs: pd.DataFrame,
    capital: float,
    nifty100_syms: set[str],
    sector_map: dict[str, str],
    base_risk: float = 0.02,
) -> None:
    if sigs.empty:
        print("\nNo NSE-S1 signals for tomorrow's entry.")
        return

    print(f"\n{'='*62}")
    print(f"  NSE-S1 Signals  →  enter at tomorrow's open")
    print(f"  Reference capital: ₹{capital:,.0f} | base risk: {base_risk*100:.0f}%")
    print(f"{'='*62}")
    print(f"  {'Symbol':<14} {'Scr':>3}  {'Close':>8}  {'Stop':>8}  {'Target':>8}  "
          f"{'Risk%':>6}  {'~Qty':>6}  {'Sector'}")
    print(f"  {'-'*14} {'-'*3}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*12}")

    for _, r in sigs.iterrows():
        sym        = r["symbol"]
        score      = int(r["score"])
        close      = float(r["close"])
        stop       = float(r["stop_loss"])
        target     = float(r["target"])
        stop_dist  = close - stop
        risk_pct   = base_risk * (1.5 if score >= 5 else 1.0)
        risk_amt   = capital * risk_pct
        qty        = int(risk_amt / stop_dist) if stop_dist > 0 else 0
        sector     = sector_map.get(sym, "Other")
        tier       = "nifty100" if sym in nifty100_syms else "midcap100"
        print(f"  {sym:<14} {score:>3}  {close:>8.2f}  {stop:>8.2f}  {target:>8.2f}  "
              f"{risk_pct*100:>5.1f}%  {qty:>6}  {sector}")

    max_gap_pct = 0.003
    print(f"\n** ENTRY FILTER (apply at tomorrow's open) **")
    print(f"   Skip any signal where open > signal_close × {1+max_gap_pct:.4f}")
    print(f"   i.e., only enter if tomorrow's open is within {max_gap_pct*100:.1f}% of today's close.")
    print(f"\nNote: qty is approximate (uses current close; actual entry = tomorrow's open)")
    print(f"      Re-anchor stop & target to actual fill: stop=fill−2.5×ATR, target=fill+5×ATR.")
    print(f"      STCG at 20% applies to gains held < 12 months.")
    print()


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="NSE-S1 daily forward test")
    ap.add_argument("--capital",   type=float, default=settings.STARTING_VIRTUAL_CAPITAL)
    ap.add_argument("--risk",      type=float, default=0.02)
    ap.add_argument("--dry-run",   action="store_true", help="Don't save signals to DB")
    ap.add_argument("--lookback",  type=int,   default=300,
                    help="Days of history to load (default 300)")
    ap.add_argument("--vix-threshold", type=float, default=20.0)
    args = ap.parse_args()

    today = date.today()
    print(f"\nNSE-S1 Forward Test  —  {today}")
    print("Loading data ...")

    ohlcv_dict  = _load_recent_ohlcv(lookback_days=args.lookback)
    nifty_df    = _load_index_recent(settings.NIFTY50_SYMBOL,   lookback_days=args.lookback)
    vix_df      = _load_index_recent(settings.INDIA_VIX_SYMBOL, lookback_days=args.lookback)
    sector_df   = get_sector_map()
    sector_map  = dict(zip(sector_df["symbol"], sector_df["sector"]))
    nifty100    = get_nifty100_symbols()

    if nifty_df.empty:
        print("ERROR: Nifty 50 data not found.  Run: python scripts/ingest_data.py")
        sys.exit(1)
    if vix_df.empty:
        logger.warning("India VIX data not found — VIX gate treated as open")
        vix_df = pd.DataFrame(
            {"open": 0, "high": 0, "low": 0, "close": 0, "volume": 0},
            index=nifty_df.index,
        )

    symbols = list(ohlcv_dict.keys())
    print(f"  {len(symbols)} symbols loaded")

    # Generate signals (all historical — we only care about today's row)
    signals = generate_signals(
        ohlcv_dict=ohlcv_dict,
        nifty_df=nifty_df,
        vix_df=vix_df,
        universe_symbols=symbols,
        vix_threshold=args.vix_threshold,
    )

    # Latest signal date in the data
    if not ohlcv_dict:
        print("No OHLCV data available.")
        return
    latest_date = max(df.index[-1] for df in ohlcv_dict.values())
    latest_ts   = pd.Timestamp(latest_date)

    # Additional regime filter: Nifty must be above its 50d EMA (validated in backtest)
    nifty_ema50 = nifty_df["close"].ewm(span=50, adjust=False).mean()
    nifty_latest = nifty_df["close"].iloc[-1]
    nifty_ema50_latest = nifty_ema50.iloc[-1]
    regime_ok = nifty_latest > nifty_ema50_latest
    print(f"  Regime check: Nifty {nifty_latest:.0f} vs EMA50 {nifty_ema50_latest:.0f} → "
          f"{'OPEN ✓' if regime_ok else 'BLOCKED (Nifty < 50d EMA)'}")

    today_sigs = signals[signals["date"] == latest_ts].copy() if not signals.empty else pd.DataFrame()

    if not regime_ok:
        print("\n** No signals today — Nifty is below its 50-day EMA. Regime filter active. **")
        today_sigs = pd.DataFrame()

    # Print entry candidates for tomorrow
    _print_signals(today_sigs, args.capital, nifty100, sector_map, base_risk=args.risk)

    # Check exits on open paper positions
    print("Checking open paper positions ...")
    open_pos = _load_open_paper_positions()
    if open_pos.empty:
        print("  No open NSE-S1 paper positions in Ledger A.")
    else:
        print(f"  {len(open_pos)} open paper position(s):")
        for _, p in open_pos.iterrows():
            print(f"    {p['symbol']:14}  entry={p['entry_price']:.2f}  "
                  f"stop={p['stop_loss']:.2f}  target={p['target']:.2f}")
        _check_exits(open_pos, ohlcv_dict, today, nifty100)

    # Persist signals
    if not args.dry_run and not today_sigs.empty:
        _save_today_signals(today_sigs, latest_date.date() if hasattr(latest_date, 'date') else today)
        print(f"Saved {len(today_sigs)} signals to DB.")
    elif args.dry_run:
        print("(dry-run: signals not saved to DB)")


if __name__ == "__main__":
    main()
