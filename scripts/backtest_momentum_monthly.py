"""
Monthly Momentum Rotation backtest for NSE.

Empirical basis (BacktestIndia, 18-year NSE data):
  - Quality-Momentum: 17.95% CAGR, Sharpe 0.86 (vs Nifty 10.42%)
  - Monthly-rebalance momentum: 15-20% CAGR pre-tax
  - Multi-factor (momentum + quality proxy): outperforms single-factor

Strategy rules:
  1. At each month-end rebalance date:
     a. Rank all universe stocks by momentum score (default: 12-1 month return;
        --composite uses weighted 1m+3m+6m+12m composite)
     b. Absolute momentum filter: if Nifty trailing return < abs_threshold, go to cash
     c. Buy top_n stocks from top sectors (--top-sectors N to enable sector filter)
     d. Size by equal weight OR inverse volatility (--inv-vol)
     e. Sell stocks that drop out of top_n
  2. Costs: ₹40 brokerage + ₹25 DP per sell leg (buys ₹40 + STT 0.1%)
  3. Fill at month-end close price

Enhancements vs baseline (all optional via flags):
  --composite    : rank by weighted composite (12m×0.4 + 6m×0.3 + 3m×0.2 + 1m×0.1)
                   more robust across regimes than single 12-1m window
  --inv-vol      : inverse-volatility position sizing (risk parity)
                   lower-vol stocks get more capital → reduces drawdown
  --top-sectors N: sector momentum filter — only pick from top N sectors
                   by 3-month average return (sector rotation layer)

Usage:
  python scripts/backtest_momentum_monthly.py
  python scripts/backtest_momentum_monthly.py --composite --inv-vol
  python scripts/backtest_momentum_monthly.py --composite --inv-vol --top-sectors 5
  python scripts/backtest_momentum_monthly.py --top-n 10 --lookback 12 --skip 1
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
from storage import store

# ── constants ─────────────────────────────────────────────────────────────────
BROKERAGE_BUY  = 40.0    # ₹ per buy leg
BROKERAGE_SELL = 40.0    # ₹ per sell leg
DP_CHARGE      = 25.50   # ₹ per sell (demat)
STT_BUY_PCT    = 0.001   # 0.1% STT on delivery buy
STT_SELL_PCT   = 0.001   # 0.1% STT on delivery sell
EXCHANGE_CHARGE_PCT = 0.00003  # NSE + SEBI + stamp (approx)
SLIPPAGE_PCT   = 0.0010  # 0.10% adverse market impact/spread per leg (was MISSING — inflated returns)


def _trade_cost(value: float, side: str) -> float:
    """One-sided transaction cost: brokerage + STT + DP (sell only) + exchange + slippage."""
    cost = BROKERAGE_BUY if side == "buy" else BROKERAGE_SELL
    if side == "sell":
        cost += DP_CHARGE
    stt_pct = STT_BUY_PCT if side == "buy" else STT_SELL_PCT
    cost += value * (stt_pct + EXCHANGE_CHARGE_PCT + SLIPPAGE_PCT)
    return cost


def _load_adjusted_ohlcv(from_dt: date, to_dt: date) -> dict[str, pd.DataFrame]:
    """
    Load all symbol OHLCV from DuckDB.
    Uses both eod2 (2015+) and yfinance (2008-2014 backfill) rows.
    When both sources have data for the same (symbol, dt), eod2 wins.
    """
    with store.db_conn() as conn:
        # Check whether yfinance rows exist at all
        has_yf = conn.execute(
            "SELECT COUNT(*) FROM adjusted_ohlcv WHERE source = 'yfinance' LIMIT 1"
        ).fetchone()[0] > 0

        if has_yf:
            # Deduplicate: eod2 rows override yfinance rows for same (symbol, dt)
            df = conn.execute("""
                SELECT symbol, dt, open, high, low, close, volume
                FROM (
                    SELECT *, ROW_NUMBER() OVER (
                        PARTITION BY symbol, dt
                        ORDER BY CASE source WHEN 'eod2' THEN 0 ELSE 1 END
                    ) AS rn
                    FROM adjusted_ohlcv
                    WHERE dt >= ? AND dt <= ?
                )
                WHERE rn = 1
                ORDER BY symbol, dt
            """, [from_dt, to_dt]).df()
        else:
            df = conn.execute("""
                SELECT symbol, dt, open, high, low, close, volume
                FROM adjusted_ohlcv
                WHERE dt >= ? AND dt <= ? AND source = 'eod2'
                ORDER BY symbol, dt
            """, [from_dt, to_dt]).df()

    if df.empty:
        return {}

    df["dt"] = pd.to_datetime(df["dt"])
    ohlcv = {}
    boundary = pd.Timestamp("2015-01-01")

    for sym, grp in df.groupby("symbol"):
        g = grp.set_index("dt").sort_index()

        # Price splice normalisation: yfinance and eod2 may have different adjustment
        # bases. Find the ratio at the 2015 boundary and rescale the yfinance segment.
        yf_seg  = g[g.index <  boundary]
        eod_seg = g[g.index >= boundary]
        if not yf_seg.empty and not eod_seg.empty:
            yf_last   = yf_seg["close"].iloc[-1]
            eod_first = eod_seg["close"].iloc[0]
            if yf_last > 0 and eod_first > 0:
                scale = eod_first / yf_last
                for col in ["open", "high", "low", "close"]:
                    g.loc[g.index < boundary, col] *= scale

        ohlcv[sym] = g
    return ohlcv


def _load_index(symbol: str, from_dt: date, to_dt: date) -> pd.Series:
    """
    Load index close series from DuckDB.
    Falls back to yfinance if DuckDB has no data for the pre-2015 range.
    """
    with store.db_conn() as conn:
        df = conn.execute("""
            SELECT dt, close FROM adjusted_ohlcv
            WHERE symbol = ? AND dt >= ? AND dt <= ?
            ORDER BY dt
        """, [symbol, from_dt, to_dt]).df()

    if df.empty:
        # Try yfinance directly for index symbols
        yf_map = {"Nifty 50": "^NSEI", "India VIX": "^INDIAVIX"}
        yf_sym = yf_map.get(symbol)
        if yf_sym:
            try:
                import yfinance as yf
                raw = yf.download(yf_sym, start=from_dt.isoformat(),
                                  end=to_dt.isoformat(), progress=False, auto_adjust=True)
                if not raw.empty:
                    close = raw["Close"]
                    if isinstance(close, pd.DataFrame):
                        close = close.iloc[:, 0]
                    close.index = pd.to_datetime(close.index).normalize()
                    close.index.name = "dt"
                    return close.dropna()
            except Exception as exc:
                logger.warning("yfinance fallback for {}: {}", symbol, exc)
        return pd.Series(dtype=float)

    df["dt"] = pd.to_datetime(df["dt"])
    return df.set_index("dt")["close"]


def _load_goldbees(from_dt: date, to_dt: date) -> pd.Series:
    """Fetch GOLDBEES.NS (gold ETF) daily close from yfinance."""
    try:
        import yfinance as yf
        df = yf.download("GOLDBEES.NS", start=from_dt.isoformat(),
                         end=to_dt.isoformat(), progress=False)
        if df.empty:
            logger.warning("GOLDBEES.NS: no data from yfinance")
            return pd.Series(dtype=float)
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close.index = pd.to_datetime(close.index).normalize()
        return close.sort_index().rename("GOLDBEES")
    except Exception as exc:
        logger.warning("GOLDBEES fetch failed: {}", exc)
        return pd.Series(dtype=float)


def _month_end_dates(from_dt: date, to_dt: date, trading_dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Return list of last trading day of each month within range."""
    all_dates = trading_dates[(trading_dates >= pd.Timestamp(from_dt))
                               & (trading_dates <= pd.Timestamp(to_dt))]
    by_month = pd.Series(all_dates, index=all_dates).resample("ME").last()
    return [d for d in by_month if pd.notna(d)]


def run_monthly_momentum(
    ohlcv_dict: dict[str, pd.DataFrame],
    nifty_close: pd.Series,
    universe_symbols: list[str],
    vix_close: pd.Series | None = None,
    start_capital: float = 100_000.0,
    top_n: int = 10,
    lookback_months: int = 12,
    skip_months: int = 1,
    abs_momentum: bool = True,
    abs_lookback_months: int = 6,
    abs_threshold: float = 0.0,
    # quality proxy filters
    max_atr_pct: float = 0.05,
    min_price: float = 100.0,
    max_momentum_cap: float = 0.80,
    min_momentum_pct: float = 0.0,    # require at least X% 12m return to qualify
    require_200d_ema: bool = True,
    # sector constraints
    max_per_sector: int = 3,
    sector_map: dict | None = None,
    # enhancements
    composite_score: bool = False,    # weighted composite of 1m+3m+6m+12m
    inv_vol_weight: bool = False,      # inverse-volatility position sizing
    top_sectors_n: int = 0,            # 0 = off; N = only pick from top N sectors
    # VIX exposure scaling (faster crash exit than 6m momentum filter)
    vix_reduce: float = 25.0,          # VIX > this → invest in top_n//2 only
    vix_exit: float = 35.0,            # VIX > this → 100% cash (override abs filter)
    # Nifty 200d EMA cash filter (earlier bear market exit than 6m return)
    require_nifty_200ema: bool = False, # if True: also cash when Nifty < 200d EMA
    # Defensive allocation when cash signal fires
    defensive_mode: str = "cash",       # "cash" | "gold" | "liquid" | "split"
    gold_close: pd.Series | None = None, # GOLDBEES.NS daily close for gold mode
    liquid_rate: float = 0.065,          # annualized return for liquid ETF simulation
    # Dynamic intra-month management
    dynamic_exits: bool = False,          # if True: check trailing stop + EMA21 between rebalances
    trail_pct: float = 0.15,             # trailing stop: 15% below 30-day high
    ema_break_span: int = 21,            # EMA span for breakdown detection
) -> tuple[pd.Series, list[dict]]:
    """
    Run monthly momentum rotation backtest.

    Returns:
        equity_curve: pd.Series indexed by rebalance date, values = equity
        rebalances: list of dicts, one per rebalance event
    """
    symbols = [s for s in universe_symbols if s in ohlcv_dict]

    # Build wide close DataFrame
    close_wide = pd.DataFrame({s: ohlcv_dict[s]["close"] for s in symbols})
    close_wide = close_wide.sort_index()

    # Get all trading dates from close_wide index
    trading_dates = close_wide.index

    # Rebalance dates: last trading day of each month
    first_date = trading_dates[0]
    last_date  = trading_dates[-1]
    rebal_dates = _month_end_dates(first_date.date(), last_date.date(), trading_dates)

    # ATR calculation (need high/low)
    # Use simplified ATR proxy: rolling std of daily returns × price
    def _atr_pct(sym: str, as_of: pd.Timestamp) -> float:
        df = ohlcv_dict[sym]
        hist = df.loc[:as_of].tail(20)
        if len(hist) < 10:
            return 0.05
        returns = hist["close"].pct_change().dropna()
        return returns.std() * np.sqrt(5)  # 5-day volatility proxy

    def _ema(s: pd.Series, span: int) -> pd.Series:
        return s.ewm(span=span, adjust=False).mean()

    # Pre-compute 200d EMA for each symbol (daily, then ffill)
    ema200_wide: dict[str, pd.Series] = {}
    if require_200d_ema:
        for sym in symbols:
            ema200_wide[sym] = _ema(ohlcv_dict[sym]["close"], 200)

    equity = start_capital
    cash   = start_capital
    holdings: dict[str, int] = {}   # symbol → qty held
    _entry_prices: dict[str, float] = {}
    equity_curve: list[tuple] = []
    rebalances:   list[dict]  = []

    # Defensive allocation state
    use_gold   = defensive_mode in ("gold", "split") and gold_close is not None
    use_liquid = defensive_mode in ("liquid", "split")

    def_gold_val:    float = 0.0
    def_gold_px:     float = np.nan
    def_gold_date:   pd.Timestamp | None = None
    def_liquid_val:  float = 0.0
    def_liquid_date: pd.Timestamp | None = None

    def _gold_at(ts: pd.Timestamp) -> float | None:
        if gold_close is None or gold_close.empty:
            return None
        hist = gold_close.loc[:ts]
        return float(hist.iloc[-1]) if not hist.empty else None

    lookback_bdays  = int(lookback_months  * 21)
    skip_bdays      = int(skip_months      * 21)
    abs_bdays       = int(abs_lookback_months * 21)

    for i, reb_date in enumerate(rebal_dates):
        reb_ts = pd.Timestamp(reb_date)

        # ── Dynamic intra-month exit check (if enabled) ───────────────────────
        if dynamic_exits and holdings and i > 0:
            prev_reb_ts = pd.Timestamp(rebal_dates[i - 1])
            for sym in list(holdings.keys()):
                sym_df = ohlcv_dict.get(sym)
                if sym_df is None:
                    continue
                # Get daily bars between previous rebalance and this one
                window = sym_df[(sym_df.index > prev_reb_ts) & (sym_df.index < reb_ts)]
                if window.empty:
                    continue
                entry_px = _entry_prices.get(sym, 0)
                ema_ser  = sym_df["close"].ewm(span=ema_break_span, adjust=False).mean()
                prev_below = False
                for ts, bar in window.iterrows():
                    close = float(bar["close"])
                    # Hard floor: -15% from entry
                    if entry_px > 0 and close <= entry_px * 0.85:
                        qty = holdings.pop(sym)
                        value = qty * close
                        cash += value - _trade_cost(value, "sell")
                        _entry_prices.pop(sym, None)
                        break
                    # Trailing stop: close < 30-day rolling high × (1 - trail_pct)
                    roll_high = float(sym_df.loc[:ts, "high"].tail(22).max())
                    trail_stop_px = roll_high * (1 - trail_pct)
                    if close <= trail_stop_px:
                        qty = holdings.pop(sym)
                        value = qty * close
                        cash += value - _trade_cost(value, "sell")
                        _entry_prices.pop(sym, None)
                        break
                    # EMA21 breakdown: 2 consecutive days below EMA21
                    ema_val = float(ema_ser.loc[ts]) if ts in ema_ser.index else close
                    today_below = close < ema_val
                    if today_below and prev_below:
                        qty = holdings.pop(sym)
                        value = qty * close
                        cash += value - _trade_cost(value, "sell")
                        _entry_prices.pop(sym, None)
                        break
                    prev_below = today_below

        # Current prices at rebalance
        prices = close_wide.loc[:reb_ts].iloc[-1].dropna()

        # ── Absolute momentum filter ──────────────────────────────────────────
        in_market = True
        if abs_momentum:
            nifty_hist = nifty_close.loc[:reb_ts]
            if len(nifty_hist) > abs_bdays + 5:
                nifty_now  = nifty_hist.iloc[-1]
                nifty_then = nifty_hist.iloc[-(abs_bdays + 1)]
                nifty_ret  = (nifty_now / nifty_then) - 1.0
                in_market  = nifty_ret > abs_threshold

        # ── Nifty 200d EMA filter (optional early bear-market exit) ───────────
        if in_market and require_nifty_200ema:
            nifty_hist = nifty_close.loc[:reb_ts]
            if len(nifty_hist) >= 200:
                nifty_ema200 = float(_ema(nifty_hist, 200).iloc[-1])
                if float(nifty_hist.iloc[-1]) < nifty_ema200:
                    in_market = False

        # ── VIX exposure scaling (override abs-momentum if VIX is extreme) ────────
        # vix_exit: go to 100% cash — faster than waiting for Nifty 6m to turn
        # vix_reduce: invest in half the positions — partial protection
        effective_top_n = top_n
        vix_level = np.nan
        if vix_close is not None:
            vix_hist = vix_close.loc[:reb_ts]
            if not vix_hist.empty:
                vix_level = float(vix_hist.iloc[-1])
                if vix_level > vix_exit:
                    in_market = False          # forced cash regardless of momentum
                elif vix_level > vix_reduce:
                    effective_top_n = max(1, top_n // 2)  # half exposure

        # ── Sector momentum filter (top N sectors by 3m avg return) ─────────────
        allowed_sectors: set[str] | None = None
        if top_sectors_n > 0 and sector_map:
            sector_groups: dict[str, list[str]] = {}
            for sym in prices.index:
                s = sector_map.get(sym, "Other")
                sector_groups.setdefault(s, []).append(sym)
            sector_returns: dict[str, float] = {}
            for sec, sec_syms in sector_groups.items():
                rets = []
                for sym in sec_syms:
                    h = close_wide[sym].dropna().loc[:reb_ts]
                    if len(h) >= 66:
                        rets.append(h.iloc[-1] / h.iloc[-64] - 1.0)
                if rets:
                    sector_returns[sec] = float(np.mean(rets))
            top = sorted(sector_returns.items(), key=lambda x: x[1], reverse=True)[:top_sectors_n]
            allowed_sectors = {s for s, _ in top}

        # ── Momentum scores ───────────────────────────────────────────────────
        # Composite: weighted sum of 12m, 6m, 3m, 1m returns (academically motivated)
        # Single: classic 12-1 month return (skip last month for reversal avoidance)
        COMPOSITE_WINDOWS = [(231, 0.40), (105, 0.30), (42, 0.20), (0, 0.10)]
        # 231=252-21, 105=126-21, 42=63-21, 0=1m (no skip for 1m component)

        target_syms: list[str] = []
        if in_market:
            scores: dict[str, float] = {}
            for sym in prices.index:
                hist = close_wide[sym].dropna().loc[:reb_ts]
                min_needed = lookback_bdays + 10
                if len(hist) < min_needed:
                    continue

                price_now = float(hist.iloc[-1])

                # Quality proxy filters
                if price_now < min_price:
                    continue

                # 200d EMA alignment
                if require_200d_ema and sym in ema200_wide:
                    ema_val = ema200_wide[sym].loc[:reb_ts].iloc[-1] if not ema200_wide[sym].loc[:reb_ts].empty else np.nan
                    if pd.isna(ema_val) or price_now < float(ema_val):
                        continue

                # Sector filter
                if allowed_sectors is not None:
                    sym_sector = sector_map.get(sym, "Other") if sector_map else "Other"
                    if sym_sector not in allowed_sectors:
                        continue

                # Score: composite or single window
                if composite_score:
                    score = 0.0
                    valid = True
                    for lb, w in COMPOSITE_WINDOWS:
                        needed = lb + 22  # need lb + 1-month skip + buffer
                        if len(hist) < needed + 5:
                            valid = False
                            break
                        past = float(hist.iloc[-(lb + 22)])
                        score += w * (price_now / past - 1.0)
                    if not valid:
                        continue
                else:
                    price_past = float(hist.iloc[-(lookback_bdays - skip_bdays + 1)])
                    score = price_now / price_past - 1.0

                if score > max_momentum_cap:
                    continue  # exclude pump-and-dump

                if score < min_momentum_pct:
                    continue  # require minimum momentum — filter coasting stocks

                scores[sym] = score

            if scores:
                # Sector cap: max_per_sector per sector
                ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                sector_count: dict[str, int] = {}
                for sym, sc in ranked:
                    sector = sector_map.get(sym, "Other") if sector_map else "Other"
                    if sector_count.get(sector, 0) >= max_per_sector:
                        continue
                    target_syms.append(sym)
                    sector_count[sector] = sector_count.get(sector, 0) + 1
                    if len(target_syms) >= effective_top_n:
                        break

        # ── Exit defensive assets if re-entering market ───────────────────────
        if in_market and (def_gold_val > 0 or def_liquid_val > 0):
            if def_gold_val > 0:
                gold_now = _gold_at(reb_ts)
                exit_val  = def_gold_val * (gold_now / def_gold_px) if (gold_now and def_gold_px) else def_gold_val
                cash     += exit_val - _trade_cost(exit_val, "sell")
                def_gold_val = 0.0; def_gold_px = np.nan; def_gold_date = None

            if def_liquid_val > 0 and def_liquid_date is not None:
                days_held   = (reb_ts - def_liquid_date).days
                cash       += def_liquid_val * (1.0 + liquid_rate) ** (days_held / 365.0)
                def_liquid_val = 0.0; def_liquid_date = None

        # ── Current portfolio value (equity + defensive MTM) ──────────────────
        port_value = cash
        for sym, qty in holdings.items():
            p = prices.get(sym, np.nan)
            if not np.isnan(p):
                port_value += qty * p
        if def_gold_val > 0:
            gold_now    = _gold_at(reb_ts)
            port_value += def_gold_val * (gold_now / def_gold_px) if (gold_now and def_gold_px) else def_gold_val
        if def_liquid_val > 0 and def_liquid_date is not None:
            days        = (reb_ts - def_liquid_date).days
            port_value += def_liquid_val * (1.0 + liquid_rate) ** (days / 365.0)

        # ── Inverse-volatility weights (or equal weight) ──────────────────────
        if target_syms and inv_vol_weight:
            raw_vols: dict[str, float] = {}
            for sym in target_syms:
                h = close_wide[sym].dropna().loc[:reb_ts].tail(22)
                raw_vols[sym] = max(float(h.pct_change().dropna().std()), 0.001)
            inv_v = {sym: 1.0 / v for sym, v in raw_vols.items()}
            total_inv = sum(inv_v.values())
            target_weights = {sym: inv_v[sym] / total_inv for sym in target_syms}
        else:
            n_tgt = len(target_syms) or 1
            target_weights = {sym: 1.0 / n_tgt for sym in target_syms}

        # ── Determine sells and buys ──────────────────────────────────────────
        to_sell = [s for s in holdings if s not in target_syms]

        # Sell
        sell_log = []
        for sym in to_sell:
            qty = holdings.pop(sym)
            p   = prices.get(sym, np.nan)
            if np.isnan(p) or qty == 0:
                continue
            value  = qty * p
            cost   = _trade_cost(value, "sell")
            cash  += value - cost
            sell_log.append({"sym": sym, "qty": qty, "price": p, "value": value, "cost": cost})

        # Buy new positions and top-up existing (using per-stock target weights)
        buy_log = []
        for sym in target_syms:
            p = prices.get(sym, np.nan)
            if np.isnan(p) or p <= 0:
                continue
            target_val    = target_weights.get(sym, 0) * port_value
            current_qty   = holdings.get(sym, 0)
            current_value = current_qty * p
            diff_value    = target_val - current_value

            if diff_value > 500:  # only buy if meaningful
                qty_add = int(diff_value / p)
                if qty_add > 0 and cash >= qty_add * p:
                    value = qty_add * p
                    cost  = _trade_cost(value, "buy")
                    if cash >= value + cost:
                        cash -= value + cost
                        holdings[sym] = current_qty + qty_add
                        if sym not in _entry_prices:
                            _entry_prices[sym] = p
                        buy_log.append({"sym": sym, "qty": qty_add, "price": p, "value": value, "cost": cost})
            elif diff_value < -500:  # trim if overweight
                qty_trim = int(abs(diff_value) / p)
                if qty_trim > 0 and holdings.get(sym, 0) >= qty_trim:
                    value = qty_trim * p
                    cost  = _trade_cost(value, "sell")
                    cash += value - cost
                    holdings[sym] = holdings[sym] - qty_trim

        # ── Enter defensive assets when not in market ─────────────────────────
        if not in_market and defensive_mode != "cash" and cash > 1000:
            to_invest = cash  # total idle cash to deploy defensively
            if defensive_mode == "liquid":
                if def_liquid_val == 0:
                    def_liquid_val  = to_invest
                    def_liquid_date = reb_ts
                    cash           -= to_invest
            elif defensive_mode == "gold":
                if def_gold_val == 0:
                    gold_px_now = _gold_at(reb_ts)
                    if gold_px_now:
                        entry_cost    = _trade_cost(to_invest, "buy")
                        def_gold_val  = to_invest - entry_cost
                        def_gold_px   = gold_px_now
                        def_gold_date = reb_ts
                        cash         -= to_invest
                    else:
                        # No gold price — park in liquid as fallback
                        def_liquid_val  = to_invest
                        def_liquid_date = reb_ts
                        cash           -= to_invest
            elif defensive_mode == "split":
                half = to_invest / 2.0
                # Gold half
                if def_gold_val == 0 and half > 500:
                    gold_px_now = _gold_at(reb_ts)
                    if gold_px_now:
                        entry_cost    = _trade_cost(half, "buy")
                        def_gold_val  = half - entry_cost
                        def_gold_px   = gold_px_now
                        def_gold_date = reb_ts
                        cash         -= half
                    else:
                        # Fallback: add to liquid portion
                        half_extra = half
                        half = to_invest  # liquid gets all
                        half = to_invest - 0  # effectively no gold, liquid gets all
                # Liquid half (always invest remaining cash)
                if def_liquid_val == 0 and cash > 500:
                    invest_liq      = cash
                    def_liquid_val  = invest_liq
                    def_liquid_date = reb_ts
                    cash           -= invest_liq

        # Compute current equity after rebalance
        equity = cash
        for sym, qty in holdings.items():
            p = prices.get(sym, np.nan)
            if not np.isnan(p):
                equity += qty * p
        # Add defensive value to equity
        if def_gold_val > 0 and use_gold:
            gold_now = _gold_at(reb_ts)
            equity  += def_gold_val * (gold_now / def_gold_px) if (gold_now and def_gold_px) else def_gold_val
        if def_liquid_val > 0 and def_liquid_date is not None:
            days    = (reb_ts - def_liquid_date).days
            equity += def_liquid_val * (1.0 + liquid_rate) ** (days / 365.0)

        equity_curve.append((reb_ts, equity))
        rebalances.append({
            "date": reb_ts,
            "equity": equity,
            "cash": cash,
            "in_market": in_market,
            "portfolio": list(holdings.keys()),
            "sells": len(sell_log),
            "buys": len(buy_log),
            "n_held": len(holdings),
        })

    eq_series = pd.Series(
        [e for _, e in equity_curve],
        index=[d for d, _ in equity_curve],
    )
    return eq_series, rebalances


def _print_report(
    eq_curve: pd.Series,
    rebalances: list[dict],
    start_capital: float,
    is_end_year: int,
    label: str,
) -> None:
    if eq_curve.empty:
        print("No equity curve — no trades executed.")
        return

    def _cagr(s: pd.Series) -> float:
        if len(s) < 2:
            return 0.0
        years = (s.index[-1] - s.index[0]).days / 365.25
        if years <= 0:
            return 0.0
        return (s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1

    def _maxdd(s: pd.Series) -> float:
        roll_max = s.cummax()
        dd = (s - roll_max) / roll_max
        return float(dd.min())

    def _cagr_range(s: pd.Series, start_capital: float) -> float:
        if s.empty:
            return 0.0
        return _cagr(pd.concat([pd.Series([start_capital], index=[s.index[0] - pd.DateOffset(months=1)]), s]))

    is_mask  = eq_curve.index.year <= is_end_year
    oos_mask = eq_curve.index.year >  is_end_year

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    if is_mask.any():
        is_eq = eq_curve[is_mask]
        is_start = start_capital if is_mask.all() else eq_curve[~is_mask].iloc[0]
        is_cagr  = _cagr(pd.concat([pd.Series([start_capital], index=[eq_curve.index[0] - pd.DateOffset(months=1)]), is_eq]))
        is_dd    = _maxdd(pd.concat([pd.Series([start_capital], index=[eq_curve.index[0] - pd.DateOffset(months=1)]), is_eq]))
        print(f"\n  IS  ({eq_curve.index[is_mask][0].year}–{is_end_year}):")
        print(f"    CAGR: {is_cagr*100:+.1f}%  |  MaxDD: {is_dd*100:.1f}%  |  Final: ₹{is_eq.iloc[-1]:,.0f}")

    if oos_mask.any():
        oos_eq = eq_curve[oos_mask]
        oos_start_val = eq_curve[is_mask].iloc[-1] if is_mask.any() else start_capital
        oos_cagr = _cagr(pd.concat([pd.Series([oos_start_val], index=[oos_eq.index[0] - pd.DateOffset(months=1)]), oos_eq]))
        oos_dd   = _maxdd(pd.concat([pd.Series([oos_start_val], index=[oos_eq.index[0] - pd.DateOffset(months=1)]), oos_eq]))

        print(f"\n  OOS ({is_end_year+1}–{eq_curve.index.year.max()}):")
        print(f"    CAGR: {oos_cagr*100:+.1f}%  |  MaxDD: {oos_dd*100:.1f}%  |  Final: ₹{oos_eq.iloc[-1]:,.0f}")

        print(f"\n  OOS year-by-year:")
        # Build a dict: year → last equity of that year
        oos_full = pd.concat([pd.Series([oos_start_val],
                              index=[oos_eq.index[0] - pd.DateOffset(months=1)]), oos_eq])
        yr_end_equity: dict[int, float] = {}
        for yr in [oos_full.index[0].year] + sorted(oos_eq.index.year.unique().tolist()):
            yr_pts = oos_full[oos_full.index.year == yr]
            if not yr_pts.empty:
                yr_end_equity[yr] = float(yr_pts.iloc[-1])
        for yr in sorted(oos_eq.index.year.unique()):
            prev_yr = yr - 1
            if yr in yr_end_equity and prev_yr in yr_end_equity:
                yr_ret = yr_end_equity[yr] / yr_end_equity[prev_yr] - 1
                print(f"    {yr}: {yr_ret*100:+.1f}%")

    total_cagr = _cagr(pd.concat([pd.Series([start_capital], index=[eq_curve.index[0] - pd.DateOffset(months=1)]), eq_curve]))
    total_dd   = _maxdd(pd.concat([pd.Series([start_capital], index=[eq_curve.index[0] - pd.DateOffset(months=1)]), eq_curve]))
    n_months_in_market = sum(1 for r in rebalances if r["in_market"])
    pct_invested = n_months_in_market / len(rebalances) * 100 if rebalances else 0

    print(f"\n  Overall: CAGR {total_cagr*100:+.1f}%  |  MaxDD {total_dd*100:.1f}%  |  "
          f"Final ₹{eq_curve.iloc[-1]:,.0f}")
    print(f"  Rebalances: {len(rebalances)}  |  Invested: {pct_invested:.0f}% of months")
    print(f"{'='*70}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Monthly Momentum Rotation backtest")
    ap.add_argument("--capital",       type=float, default=settings.STARTING_VIRTUAL_CAPITAL)
    ap.add_argument("--is-end",        type=int,   default=2020)
    ap.add_argument("--from-year",     type=int,   default=2015)
    ap.add_argument("--to-year",       type=int,   default=2025)
    ap.add_argument("--top-n",         type=int,   default=10,
                    help="Number of stocks in portfolio (default 10)")
    ap.add_argument("--lookback",      type=int,   default=12,
                    help="Momentum lookback in months (default 12)")
    ap.add_argument("--skip",          type=int,   default=1,
                    help="Skip last N months to avoid reversal (default 1 = 12-1 momentum)")
    ap.add_argument("--no-abs-momentum", action="store_true",
                    help="Disable absolute momentum filter (always stay invested)")
    ap.add_argument("--abs-lookback",  type=int,   default=6,
                    help="Absolute momentum lookback months (default 6)")
    ap.add_argument("--no-ema200",     action="store_true",
                    help="Disable 200d EMA quality filter")
    ap.add_argument("--max-per-sector", type=int,  default=3,
                    help="Max stocks per sector (default 3)")
    # Enhancement flags
    ap.add_argument("--composite",     action="store_true",
                    help="Use composite momentum score (12m×0.4 + 6m×0.3 + 3m×0.2 + 1m×0.1)")
    ap.add_argument("--inv-vol",       action="store_true",
                    help="Inverse-volatility position sizing (risk parity) instead of equal weight")
    ap.add_argument("--top-sectors",   type=int,   default=0,
                    help="Only pick from top N sectors by 3m return (0 = disabled)")
    ap.add_argument("--vix-reduce",    type=float, default=25.0,
                    help="VIX above this → invest in top_n//2 only (default 25)")
    ap.add_argument("--vix-exit",      type=float, default=35.0,
                    help="VIX above this → 100%% cash (default 35)")
    ap.add_argument("--no-vix-scale",  action="store_true",
                    help="Disable VIX-based exposure scaling")
    ap.add_argument("--min-momentum",  type=float, default=0.0,
                    help="Minimum 12m return required to qualify (e.g. 0.10 = 10%%, default 0)")
    ap.add_argument("--nifty-200ema",  action="store_true",
                    help="Also go to cash when Nifty < 200d EMA (earlier bear-market exit)")
    ap.add_argument("--defensive",     type=str, default="cash",
                    choices=["cash", "gold", "liquid", "split"],
                    help="Where to park capital when cash signal fires: "
                         "cash (default), gold (GOLDBEES ETF), liquid (6.5%% p.a.), "
                         "split (50%% gold + 50%% liquid)")
    ap.add_argument("--liquid-rate",   type=float, default=0.065,
                    help="Annual return to model for liquid ETF (default 0.065 = 6.5%%)")
    ap.add_argument("--dynamic", action="store_true",
                    help="Enable dynamic intra-month trailing stop + EMA21 exit simulation")
    args = ap.parse_args()

    from_dt = date(args.from_year, 1, 1)
    to_dt   = date(args.to_year, 12, 31)

    print(f"\nLoading data {args.from_year}–{args.to_year} ...")
    ohlcv_dict  = _load_adjusted_ohlcv(from_dt, to_dt)
    nifty_close = _load_index(settings.NIFTY50_SYMBOL,   from_dt, to_dt)
    vix_close   = _load_index(settings.INDIA_VIX_SYMBOL, from_dt, to_dt)

    gold_close = None
    if args.defensive in ("gold", "split"):
        print("Fetching GOLDBEES.NS from yfinance ...")
        gold_close = _load_goldbees(from_dt, to_dt)
        if gold_close.empty:
            print("WARNING: GOLDBEES data unavailable — falling back to cash")
            args.defensive = "cash"
        else:
            print(f"GOLDBEES: {len(gold_close)} days ({gold_close.index[0].date()} – {gold_close.index[-1].date()})")

    if not ohlcv_dict:
        print("ERROR: No OHLCV data found. Run: python scripts/ingest_data.py")
        sys.exit(1)
    if nifty_close.empty:
        print("ERROR: Nifty 50 data missing. Run: python scripts/ingest_data.py")
        sys.exit(1)

    vix_arg = None if (vix_close.empty or args.no_vix_scale) else vix_close

    # Universe: all symbols in DuckDB (exclude indices)
    universe_symbols = [s for s in ohlcv_dict.keys()
                        if s not in (settings.NIFTY50_SYMBOL, settings.INDIA_VIX_SYMBOL)]

    sector_df  = get_sector_map()
    sector_map = dict(zip(sector_df["symbol"], sector_df["sector"])) if not sector_df.empty else None

    print(f"Universe: {len(universe_symbols)} symbols")
    print(f"Strategy: top-{args.top_n} | lookback {args.lookback}-{args.skip}mo | "
          f"abs={not args.no_abs_momentum} | inv-vol={args.inv_vol} | "
          f"vix-scale={not args.no_vix_scale}")

    eq_curve, rebalances = run_monthly_momentum(
        ohlcv_dict=ohlcv_dict,
        nifty_close=nifty_close,
        universe_symbols=universe_symbols,
        vix_close=vix_arg,
        start_capital=args.capital,
        top_n=args.top_n,
        lookback_months=args.lookback,
        skip_months=args.skip,
        abs_momentum=not args.no_abs_momentum,
        abs_lookback_months=args.abs_lookback,
        require_200d_ema=not args.no_ema200,
        min_momentum_pct=args.min_momentum,
        require_nifty_200ema=args.nifty_200ema,
        defensive_mode=args.defensive,
        gold_close=gold_close,
        liquid_rate=args.liquid_rate,
        max_per_sector=args.max_per_sector,
        sector_map=sector_map,
        composite_score=args.composite,
        inv_vol_weight=args.inv_vol,
        top_sectors_n=args.top_sectors,
        vix_reduce=args.vix_reduce,
        vix_exit=args.vix_exit,
        dynamic_exits=args.dynamic,
    )

    enhancements = []
    if args.composite:    enhancements.append("composite-score")
    if args.inv_vol:      enhancements.append("inv-vol")
    if args.top_sectors:  enhancements.append(f"top{args.top_sectors}sec")
    if not args.no_vix_scale: enhancements.append(f"vix≤{args.vix_reduce:.0f}/{args.vix_exit:.0f}")
    if args.min_momentum:   enhancements.append(f"min{args.min_momentum*100:.0f}%mom")
    if args.nifty_200ema:  enhancements.append("n200ema")
    if args.defensive != "cash": enhancements.append(f"def={args.defensive}")
    enh_str = "+" + "+".join(enhancements) if enhancements else ""
    label = (
        f"[Monthly-Momentum{enh_str}] top-{args.top_n} | {args.lookback}-{args.skip}mo "
        f"| abs={not args.no_abs_momentum}"
    )
    _print_report(eq_curve, rebalances, args.capital, args.is_end, label)

    # Show current portfolio (if run up to recent date)
    if rebalances:
        last = rebalances[-1]
        print(f"Last rebalance ({last['date'].date()}): {last['portfolio']}")


if __name__ == "__main__":
    main()
