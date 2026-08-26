"""
Monthly Momentum forward runner — run at month-end to get next portfolio.

This uses the same logic as backtest_momentum_monthly.py but:
  - Uses today's data (or latest available)
  - Shows current regime status (in/out of market)
  - Shows current portfolio + new target
  - Shows what to buy and sell

Run after market close on the last trading day of each month:
  python scripts/forward_test_momentum_monthly.py

For a specific date (e.g. checking last month-end):
  python scripts/forward_test_momentum_monthly.py --as-of 2026-07-31
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from data.universe import get_sector_map
from reporting.notifier import Notifier
from storage import store

# ── defaults (locked to grid-search optimal: 2008-2026 IS/OOS) ─────────────────
TOP_N               = 10
LOOKBACK_MO         = 12
SKIP_MO             = 1
ABS_LOOKBACK        = 6
VIX_REDUCE          = 25.0
VIX_EXIT            = 35.0
MIN_PRICE           = 100.0
MAX_SECTOR          = 3
MAX_MOM_CAP         = 0.80
REQUIRE_NIFTY_200EMA = True     # go to GOLDBEES when Nifty < 200d EMA
DEFENSIVE_MODE      = "gold"    # park cash in GOLDBEES (IS+18.8%/OOS+31.5% vs cash IS+9.1%)
WATCHLIST_N     = 25      # wider candidate pool for mid-month entries
COMPOSITE       = True    # weighted 12m×0.4 + 6m×0.3 + 3m×0.2 + 1m×0.1


def _load_adjusted_ohlcv_all(from_dt: date, to_dt: date) -> dict[str, pd.DataFrame]:
    with store.db_conn() as conn:
        df = conn.execute("""
            SELECT symbol, dt, open, high, low, close, volume
            FROM adjusted_ohlcv
            WHERE dt >= ? AND dt <= ? AND source = 'eod2'
            ORDER BY symbol, dt
        """, [from_dt, to_dt]).df()
    if df.empty:
        return {}
    df["dt"] = pd.to_datetime(df["dt"])
    return {sym: grp.set_index("dt").sort_index()
            for sym, grp in df.groupby("symbol")}


def _load_index(symbol: str, from_dt: date, to_dt: date) -> pd.Series:
    with store.db_conn() as conn:
        df = conn.execute("""
            SELECT dt, close FROM adjusted_ohlcv
            WHERE symbol = ? AND dt >= ? AND dt <= ?
            ORDER BY dt
        """, [symbol, from_dt, to_dt]).df()
    if df.empty:
        return pd.Series(dtype=float)
    df["dt"] = pd.to_datetime(df["dt"])
    return df.set_index("dt")["close"]


def compute_signals(
    ohlcv_dict: dict[str, pd.DataFrame],
    nifty_close: pd.Series,
    vix_close: pd.Series | None,
    sector_map: dict,
    as_of: pd.Timestamp,
    top_n: int = TOP_N,
    lookback_bdays: int = LOOKBACK_MO * 21,
    skip_bdays: int = SKIP_MO * 21,
    abs_lookback_bdays: int = ABS_LOOKBACK * 21,
    vix_reduce: float = VIX_REDUCE,
    vix_exit: float = VIX_EXIT,
    min_price: float = MIN_PRICE,
    max_per_sector: int = MAX_SECTOR,
    max_mom_cap: float = MAX_MOM_CAP,
    require_200d_ema: bool = True,
    require_nifty_200ema: bool = REQUIRE_NIFTY_200EMA,
    defensive_mode: str = DEFENSIVE_MODE,
    gold_close: pd.Series | None = None,   # GOLDBEES/gold price for cash-fallback
    composite_score: bool = COMPOSITE,
) -> dict:
    """Compute the month-end signal for the forward test."""

    def _ema(s: pd.Series, span: int) -> pd.Series:
        return s.ewm(span=span, adjust=False).mean()

    # ── Regime checks ────────────────────────────────────────────────────────
    nifty_hist = nifty_close.loc[:as_of]
    nifty_now  = float(nifty_hist.iloc[-1]) if not nifty_hist.empty else np.nan
    nifty_ema200 = float(_ema(nifty_close, 200).reindex(nifty_hist.index).iloc[-1]) if not nifty_hist.empty else np.nan

    # Absolute momentum: Nifty 6-month return
    abs_ok = True
    nifty_6m_ret = np.nan
    if len(nifty_hist) > abs_lookback_bdays + 5:
        nifty_then    = float(nifty_hist.iloc[-(abs_lookback_bdays + 1)])
        nifty_6m_ret  = nifty_now / nifty_then - 1.0
        abs_ok        = nifty_6m_ret > 0

    # Nifty 200-day EMA regime filter (faster crash exit than 6m momentum)
    nifty_below_200ema = (
        require_nifty_200ema
        and not np.isnan(nifty_ema200)
        and not np.isnan(nifty_now)
        and nifty_now < nifty_ema200
    )

    # Gold 6m return — if gold is also bearish when we'd go defensive → use cash instead
    gold_6m_ret  = np.nan
    effective_defensive = defensive_mode
    if gold_close is not None and not gold_close.empty:
        gold_hist = gold_close.loc[:as_of].dropna()
        if len(gold_hist) >= abs_lookback_bdays + 5:
            gold_6m_ret = float(gold_hist.iloc[-1]) / float(gold_hist.iloc[-(abs_lookback_bdays + 1)]) - 1.0
            if gold_6m_ret < 0 and defensive_mode == "gold":
                effective_defensive = "cash"
                logger.info("Gold 6m return {:.1f}% < 0 — switching defensive to CASH", gold_6m_ret * 100)

    # VIX
    in_market    = abs_ok and not nifty_below_200ema
    effective_n  = top_n
    vix_now      = np.nan
    vix_status   = "N/A"
    if vix_close is not None and not vix_close.empty:
        vix_hist = vix_close.loc[:as_of]
        if not vix_hist.empty:
            vix_now = float(vix_hist.iloc[-1])
            if vix_now > vix_exit:
                in_market  = False
                vix_status = f"CASH FORCED (VIX {vix_now:.1f} > {vix_exit:.0f})"
            elif vix_now > vix_reduce:
                effective_n = max(1, top_n // 2)
                vix_status  = f"REDUCED ({effective_n} positions, VIX {vix_now:.1f} > {vix_reduce:.0f})"
            else:
                vix_status = f"Normal ({vix_now:.1f})"

    # ── Pre-compute 200d EMAs ──────────────────────────────────────────────
    ema200: dict[str, float] = {}
    if require_200d_ema:
        for sym, df in ohlcv_dict.items():
            hist = df["close"].loc[:as_of]
            if len(hist) >= 150:
                ema200[sym] = float(_ema(hist, 200).iloc[-1])

    # ── Rank stocks by momentum ────────────────────────────────────────────
    scores: dict[str, float] = {}
    meta: dict[str, dict]   = {}
    inv_vols: dict[str, float] = {}

    if in_market:
        for sym, df in ohlcv_dict.items():
            hist = df["close"].loc[:as_of].dropna()
            if len(hist) < lookback_bdays + 10:
                continue

            price_now = float(hist.iloc[-1])
            if price_now < min_price:
                continue

            if require_200d_ema and sym in ema200:
                if price_now < ema200[sym]:
                    continue

            if composite_score:
                # Need 231 bars minimum (12m window - 1m skip = 231 bdays)
                if len(hist) < 241:
                    continue
                r12 = price_now / float(hist.iloc[-231]) - 1.0   # 12m minus 1m skip
                r6  = price_now / float(hist.iloc[-105]) - 1.0   # 6m minus 1m skip
                r3  = price_now / float(hist.iloc[-42])  - 1.0   # 3m minus 1m skip
                r1  = price_now / float(hist.iloc[-21])  - 1.0   # 1m (no skip)
                score = 0.4*r12 + 0.3*r6 + 0.2*r3 + 0.1*r1
            else:
                r12   = price_now / float(hist.iloc[-(lookback_bdays - skip_bdays + 1)]) - 1.0
                score = r12
                r6    = price_now / float(hist.iloc[-105]) - 1.0 if len(hist) >= 105 else float("nan")
                r3    = price_now / float(hist.iloc[-42])  - 1.0 if len(hist) >= 42  else float("nan")
                r1    = price_now / float(hist.iloc[-21])  - 1.0 if len(hist) >= 21  else float("nan")
            if score > max_mom_cap:
                continue

            # 20-day realized vol for inv-vol weight
            daily_returns = hist.tail(22).pct_change().dropna()
            inv_vols[sym] = max(float(daily_returns.std()), 0.001)

            # 1m, 3m, 6m returns for display (already computed above for composite score)
            ret_1m = r1
            ret_3m = r3
            ret_6m = r6

            scores[sym] = score
            meta[sym]   = {"price": price_now, "score_12m": score,
                           "ret_1m": ret_1m, "ret_3m": ret_3m, "ret_6m": ret_6m,
                           "ema200": ema200.get(sym, np.nan)}

        # Rank and pick top_n with sector cap
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        sector_count: dict[str, int] = {}
        target_syms: list[str] = []
        for sym, sc in ranked:
            sec = sector_map.get(sym, "Other")
            if sector_count.get(sec, 0) >= max_per_sector:
                continue
            target_syms.append(sym)
            sector_count[sec] = sector_count.get(sec, 0) + 1
            if len(target_syms) >= effective_n:
                break
    else:
        target_syms = []

    # ── Watchlist: top-25 with relaxed sector cap (for daily entry triggers) ──
    watchlist_syms: list[str] = []
    if in_market and scores:
        wl_sector_count: dict[str, int] = {}
        for sym, sc in ranked:
            sec = sector_map.get(sym, "Other")
            if wl_sector_count.get(sec, 0) >= 5:   # relaxed cap: 5 per sector
                continue
            watchlist_syms.append(sym)
            wl_sector_count[sec] = wl_sector_count.get(sec, 0) + 1
            if len(watchlist_syms) >= WATCHLIST_N:
                break

    # Compute inv-vol weights for the final target
    if target_syms and inv_vols:
        raw = {sym: inv_vols[sym] for sym in target_syms if sym in inv_vols}
        total = sum(1.0/v for v in raw.values()) if raw else 1
        weights = {sym: (1.0/raw[sym]) / total for sym in target_syms if sym in raw}
    else:
        n = len(target_syms) or 1
        weights = {sym: 1.0/n for sym in target_syms}

    return {
        "as_of": as_of,
        "in_market": in_market,
        "nifty_now": nifty_now,
        "nifty_ema200": nifty_ema200,
        "nifty_below_200ema": nifty_below_200ema,
        "nifty_6m_ret": nifty_6m_ret,
        "gold_6m_ret": gold_6m_ret,
        "vix_now": vix_now,
        "vix_status": vix_status,
        "effective_n": effective_n,
        "target": target_syms,
        "weights": weights,
        "meta": meta,
        "all_scores": scores,
        "watchlist": watchlist_syms,
        "defensive_mode": effective_defensive,
    }


def print_report(sig: dict, current_holdings: list[str] | None = None, capital: float = 100_000) -> None:
    as_of = sig["as_of"]
    print(f"\n{'='*70}")
    print(f"  Monthly Momentum Signal — {as_of.strftime('%d %b %Y')}")
    print(f"{'='*70}")

    # Regime
    print(f"\n  REGIME:")
    nifty = sig["nifty_now"]
    n200  = sig["nifty_ema200"]
    n6m   = sig["nifty_6m_ret"]
    print(f"    Nifty:       {nifty:,.0f}  |  200d EMA: {n200:,.0f}  |  "
          f"{'ABOVE ✓' if nifty > n200 else 'BELOW ✗'}")
    print(f"    Nifty 6m:    {n6m*100:+.1f}%  "
          f"({'positive ✓' if n6m > 0 else 'NEGATIVE ✗ → cash'})")
    gold_6m = sig.get("gold_6m_ret", float("nan"))
    if not (gold_6m != gold_6m):   # not nan
        print(f"    Gold 6m:     {gold_6m*100:+.1f}%  "
              f"({'positive ✓' if gold_6m > 0 else 'NEGATIVE ✗ → CASH not GOLDBEES'})")
    print(f"    VIX:         {sig['vix_status']}")
    print(f"    Status:      {'IN MARKET' if sig['in_market'] else '⚠ CASH — DO NOT ENTER'}")

    if not sig["in_market"]:
        def_mode = sig.get("defensive_mode", "cash")
        n200_trigger = sig.get("nifty_below_200ema", False)
        reason = "Nifty below 200d EMA" if n200_trigger else "Negative 6m momentum"
        if def_mode == "gold":
            action = "Hold / move to GOLDBEES (gold ETF)"
        elif def_mode == "split":
            action = "50% GOLDBEES + 50% liquid fund"
        else:
            action = "Stay in cash / liquid fund"
        print(f"\n  ACTION: {action}")
        print(f"  Reason: {reason}. Re-check at end of next month.\n")
        print("="*70)
        return

    target = sig["target"]
    weights = sig["weights"]
    meta   = sig["meta"]

    print(f"\n  TARGET PORTFOLIO ({len(target)} stocks):")
    print(f"  {'Symbol':<14} {'12m %':>8} {'6m %':>7} {'3m %':>7} {'1m %':>7} "
          f"{'Price':>8} {'Weight':>7} {'₹ Alloc':>10}")
    print(f"  {'-'*68}")
    for sym in target:
        m  = meta.get(sym, {})
        w  = weights.get(sym, 1/len(target))
        alloc = capital * w
        print(f"  {sym:<14} "
              f"{m.get('score_12m', 0)*100:>7.1f}% "
              f"{m.get('ret_6m', 0)*100:>6.1f}% "
              f"{m.get('ret_3m', 0)*100:>6.1f}% "
              f"{m.get('ret_1m', 0)*100:>6.1f}% "
              f"₹{m.get('price', 0):>7,.0f} "
              f"{w*100:>6.1f}% "
              f"₹{alloc:>9,.0f}")

    # What to do
    if current_holdings is not None:
        to_sell = [s for s in current_holdings if s not in target]
        to_buy  = [s for s in target if s not in current_holdings]
        hold    = [s for s in target if s in current_holdings]

        print(f"\n  ACTIONS:")
        if to_sell:
            print(f"    SELL ({len(to_sell)}): {', '.join(to_sell)}")
        if to_buy:
            print(f"    BUY  ({len(to_buy)}): {', '.join(to_buy)}")
        if hold:
            print(f"    HOLD ({len(hold)}): {', '.join(hold)}")
    else:
        print(f"\n  ACTIONS: Buy all {len(target)} stocks at above weights")

    # Top 20 full ranking (for transparency)
    print(f"\n  TOP 20 MOMENTUM RANKING (12-1m return, all qualifying stocks):")
    print(f"  {'#':>3} {'Symbol':<14} {'12m %':>8} {'Sector':<18}")
    sector_map_local = {}  # will be passed separately
    ranked_all = sorted(sig["all_scores"].items(), key=lambda x: x[1], reverse=True)[:20]
    for i, (sym, sc) in enumerate(ranked_all, 1):
        marker = "✓" if sym in target else " "
        print(f"  {i:>3}{marker} {sym:<14} {sc*100:>7.1f}%")

    # Watchlist: stocks 11-25 available for mid-month entry
    watchlist = sig.get("watchlist", [])
    extra = [s for s in watchlist if s not in target]
    if extra:
        print(f"  WATCHLIST (mid-month entry candidates, #{len(target)+1}–{len(watchlist)}):")
        for sym in extra:
            m = sig.get("meta", {}).get(sym, {})
            print(f"    {sym:<14} {m.get('score_12m', 0)*100:>7.1f}% 12m")
        print()

    print(f"\n{'='*70}\n")


def _format_telegram(sig: dict, current_holdings: list[str] | None, capital: float) -> str:
    """Format a concise Telegram message for the monthly momentum signal."""
    as_of = sig["as_of"]
    lines = [f"*Piedpiper Monthly Signal — {as_of.strftime('%d %b %Y')}*\n"]

    n6m = sig["nifty_6m_ret"]
    vix = sig["vix_status"]
    status = "IN MARKET" if sig["in_market"] else "CASH"
    lines.append(f"Regime: *{status}*")
    lines.append(f"Nifty 6m: {n6m*100:+.1f}%  |  VIX: {vix}")

    if not sig["in_market"]:
        lines.append("\n_Stay in cash. Re-check next month-end._")
        return "\n".join(lines)

    target  = sig["target"]
    weights = sig["weights"]
    meta    = sig["meta"]

    lines.append(f"\n*Portfolio ({len(target)} stocks):*")
    for sym in target:
        m     = meta.get(sym, {})
        w     = weights.get(sym, 1 / len(target))
        alloc = capital * w
        lines.append(f"  {sym}: {w*100:.0f}% (₹{alloc:,.0f})  {m.get('score_12m',0)*100:+.0f}% 12m")

    if current_holdings is not None:
        to_sell = [s for s in current_holdings if s not in target]
        to_buy  = [s for s in target if s not in current_holdings]
        if to_sell:
            lines.append(f"\n*SELL:* {', '.join(to_sell)}")
        if to_buy:
            lines.append(f"*BUY:* {', '.join(to_buy)}")
        hold = [s for s in target if s in current_holdings]
        if hold:
            lines.append(f"*HOLD:* {', '.join(hold)}")
    else:
        lines.append(f"\n*BUY ALL:* {', '.join(target)}")

    lines.append(f"\n_Execute at open, 9:30-10:00 AM IST_")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Monthly Momentum forward signal")
    ap.add_argument("--as-of",    type=str,   default=None,
                    help="Signal date YYYY-MM-DD (default: latest available data)")
    ap.add_argument("--capital",  type=float, default=100_000,
                    help="Current portfolio value for allocation display (default 100000)")
    ap.add_argument("--holdings", type=str,   default="",
                    help="Comma-separated list of current holdings for buy/sell diff")
    ap.add_argument("--top-n",    type=int,   default=TOP_N)
    ap.add_argument("--no-ema200", action="store_true")
    ap.add_argument("--notify",   action="store_true",
                    help="Send signal via Telegram (requires TELEGRAM_BOT_TOKEN in .env)")
    args = ap.parse_args()

    # Date range: need 15 months of data for 12-1 scoring
    to_dt   = date.today()
    from_dt = date(to_dt.year - 2, to_dt.month, 1)  # 2 years back for warmup

    print("Loading data...")
    ohlcv_dict  = _load_adjusted_ohlcv_all(from_dt, to_dt)
    nifty_close = _load_index(settings.NIFTY50_SYMBOL,   from_dt, to_dt)
    vix_close   = _load_index(settings.INDIA_VIX_SYMBOL, from_dt, to_dt)

    if not ohlcv_dict:
        print("ERROR: No OHLCV data. Run: python scripts/ingest_data.py")
        sys.exit(1)

    # Filter out index symbols
    universe_symbols = [s for s in ohlcv_dict
                        if s not in (settings.NIFTY50_SYMBOL, settings.INDIA_VIX_SYMBOL)]
    ohlcv_dict = {s: ohlcv_dict[s] for s in universe_symbols}

    sector_df  = get_sector_map()
    sector_map = dict(zip(sector_df["symbol"], sector_df["sector"])) if not sector_df.empty else {}

    # Determine as-of date
    all_dates = sorted(nifty_close.index)
    if args.as_of:
        as_of = pd.Timestamp(args.as_of)
    else:
        as_of = all_dates[-1] if all_dates else pd.Timestamp(to_dt)

    print(f"Signal date: {as_of.strftime('%d %b %Y')} | Universe: {len(ohlcv_dict)} symbols")

    sig = compute_signals(
        ohlcv_dict=ohlcv_dict,
        nifty_close=nifty_close,
        vix_close=vix_close if not vix_close.empty else None,
        sector_map=sector_map,
        as_of=as_of,
        top_n=args.top_n,
        require_200d_ema=not args.no_ema200,
    )

    current_holdings = [s.strip() for s in args.holdings.split(",") if s.strip()] or None
    print_report(sig, current_holdings=current_holdings, capital=args.capital)

    if args.notify:
        msg = _format_telegram(sig, current_holdings, args.capital)
        notifier = Notifier()
        ok = notifier.send_telegram(msg)
        if ok:
            print("Telegram notification sent.")
        else:
            print("Telegram send failed (check TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID in .env)")


if __name__ == "__main__":
    main()
