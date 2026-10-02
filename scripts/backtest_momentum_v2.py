"""
Momentum v2 — HONEST rebuild
============================
Fixes the issues in backtest_momentum_monthly.py:
  * Survivorship: ranks across the FULL eod2 NSE universe (~3700 stocks) with a
    POINT-IN-TIME liquidity filter (top-N by that month's turnover) — not today's
    500 index constituents. (Residual: fully-delisted names absent from eod2.)
  * Clean 12-1 momentum (no dead composite term).
  * Realistic costs: brokerage + STT + 0.10%/side slippage on traded turnover.
  * A-PRIORI standard parameters (NOT tuned on any test window).
  * One compounding equity curve; reported vs Nifty buy-&-hold, IS vs OOS.

Run:  .venv/bin/python scripts/backtest_momentum_v2.py
"""
from __future__ import annotations
import sys, glob, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import pandas as pd
import duckdb

DAILY_DIR  = "data_store/eod2/src/eod2_data/daily"
CACHE      = "data_store/mom_v2_panels.parquet"
DB_PATH    = "data_store/piedpiper.duckdb"
CAPITAL    = 200_000.0

# ── A-PRIORI standard params (textbook momentum — NOT fitted to any period) ──
TOP_N        = 15       # hold top-15 equal weight
LB, SKIP     = 12, 1    # 12-month momentum, skip most recent 1 month
TURN_TOP     = 500      # point-in-time universe: top-500 by monthly turnover
PRICE_MIN    = 30.0     # avoid penny stocks
SLIP         = 0.0010   # 0.10% per side
STT          = 0.0010   # 0.1% per side (delivery)
RT_COST      = 2 * (SLIP + STT) + 0.0005   # round-trip cost per name changed (~0.45%)
CASH_MO      = (1.065) ** (1/12) - 1        # LIQUIDBEES-like 6.5%/yr in cash
FROM_YEAR    = 2010     # start where breadth (>900 stocks) + Nifty regime data are adequate
IS_END       = 2018     # IS 2010-2018, OOS 2019-2026 (chronological, OOS untouched)


def build_panels():
    if os.path.exists(CACHE):
        print("Loading cached monthly panels ...", flush=True)
        p = pd.read_parquet(CACHE)
        close_m = p.xs("close", axis=1, level=0)
        turn_m  = p.xs("turn",  axis=1, level=0)
        return close_m, turn_m
    print(f"Building monthly panels from {DAILY_DIR} (first run, ~few min) ...", flush=True)
    closes, turns = {}, {}
    files = glob.glob(f"{DAILY_DIR}/*.csv")
    for i, f in enumerate(files):
        sym = Path(f).stem.upper()
        try:
            d = pd.read_csv(f, usecols=["Date","Close","Volume","Series"])
        except Exception:
            continue
        d = d[d["Series"] == "EQ"]
        if len(d) < 260:      # need ~1yr+ of daily data
            continue
        d["Date"] = pd.to_datetime(d["Date"])
        d = d.set_index("Date").sort_index()
        d["turnover"] = d["Close"] * d["Volume"]
        closes[sym] = d["Close"].resample("ME").last()
        turns[sym]  = d["turnover"].resample("ME").mean()
        if (i + 1) % 500 == 0:
            print(f"    [{i+1}/{len(files)}]", flush=True)
    close_m = pd.DataFrame(closes).sort_index()
    turn_m  = pd.DataFrame(turns).reindex(close_m.index)
    out = pd.concat({"close": close_m, "turn": turn_m}, axis=1)
    out.to_parquet(CACHE)
    print(f"  cached {close_m.shape[1]} symbols × {close_m.shape[0]} months", flush=True)
    return close_m, turn_m


def load_nifty():
    c = duckdb.connect(DB_PATH, read_only=True)
    n = c.execute("SELECT dt,close FROM adjusted_ohlcv WHERE symbol='Nifty 50' ORDER BY dt").df()
    c.close()
    n["dt"] = pd.to_datetime(n["dt"])
    s = n.set_index("dt")["close"].sort_index()
    return s.resample("ME").last(), s


def simulate(close_m, turn_m, nifty_m, top_n=TOP_N, rt_cost=RT_COST, turn_top=TURN_TOP, use_regime=True):
    idx = close_m.index
    ema200_ok = nifty_m.reindex(idx).ffill()
    nifty_ema = ema200_ok.ewm(span=10, adjust=False).mean()   # ~200d ≈ 10mo EMA on monthly
    equity, prev_picks, curve, invested = CAPITAL, set(), [], 0
    # momentum panel: price 1 month ago / price 13 months ago  (12m, skip 1)
    mom = close_m.shift(SKIP) / close_m.shift(SKIP + LB) - 1.0
    start_p = max(SKIP + LB + 1, int(np.searchsorted(idx.year.values, FROM_YEAR)))
    for p in range(start_p, len(idx) - 1):
        reb, nxt = idx[p], idx[p + 1]
        in_mkt = True
        if use_regime:
            nret6 = nifty_m.reindex(idx).iloc[p] / nifty_m.reindex(idx).iloc[p-6] - 1 if p >= 6 else 0
            in_mkt = (ema200_ok.iloc[p] > nifty_ema.iloc[p]) and (nret6 > -0.02)
        if in_mkt:
            price = close_m.iloc[p]
            turn  = turn_m.iloc[p]
            hist  = close_m.iloc[:p+1].notna().sum()
            elig = price.index[(price >= PRICE_MIN) & (hist >= LB + SKIP + 1) &
                               mom.iloc[p].notna() & turn.notna()]
            if len(elig) == 0:
                gross, picks = CASH_MO, set()
            else:
                liq = turn[elig].nlargest(min(turn_top, len(elig))).index      # point-in-time liquid set
                ranked = mom.iloc[p][liq].nlargest(top_n)
                picks = set(ranked.index)
                rets = [close_m.iloc[p+1][s] / close_m.iloc[p][s] - 1
                        for s in picks if pd.notna(close_m.iloc[p+1].get(s)) and pd.notna(close_m.iloc[p][s])]
                gross = float(np.mean(rets)) if rets else CASH_MO
                invested += 1
        else:
            gross, picks = CASH_MO, set()
        turnover = len(picks.symmetric_difference(prev_picks)) / max(len(picks | prev_picks), 1)
        net = gross - turnover * rt_cost
        equity *= (1 + net)
        curve.append((nxt, equity))
        prev_picks = picks
    eq = pd.Series(dict(curve)).sort_index()
    return eq, invested, len(curve)


def metrics(eq, start):
    if eq.empty or len(eq) < 2:
        return float("nan"), float("nan")
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.1)
    cagr = (eq.iloc[-1] / start) ** (1 / yrs) - 1
    dd = ((eq - eq.cummax()) / eq.cummax()).min()
    return cagr, dd


def bh(nifty_m, start_ts, end_ts, start_cap):
    s = nifty_m[(nifty_m.index >= start_ts) & (nifty_m.index <= end_ts)].dropna()
    eq = start_cap * (s / s.iloc[0])
    return metrics(eq, start_cap)


def report(eq, nifty_m):
    is_eq  = eq[eq.index.year <= IS_END]
    oos_eq = eq[eq.index.year >  IS_END]
    print(f"\n{'='*74}")
    print(f"  MOMENTUM v2 (honest) vs NIFTY 50 buy-&-hold")
    print(f"{'='*74}")
    print(f"  {'Period':<22}{'Momentum':>22}{'Nifty B&H':>22}")
    for label, e in [("FULL", eq), ("IS (≤2018)", is_eq), ("OOS (≥2019)", oos_eq)]:
        if e.empty:
            continue
        start = CAPITAL if e is eq or e is is_eq else is_eq.iloc[-1]
        mc, md = metrics(e, start)
        nc, nd = bh(nifty_m, e.index[0], e.index[-1], start)
        print(f"  {label:<22}{f'{mc*100:+.1f}% / DD {md*100:.0f}%':>22}{f'{nc*100:+.1f}% / DD {nd*100:.0f}%':>22}")
    print(f"\n  Final equity: ₹{eq.iloc[-1]:,.0f} from ₹{CAPITAL:,.0f} over {eq.index[0].date()}→{eq.index[-1].date()}")


def main():
    close_m, turn_m = build_panels()
    nifty_m, _ = load_nifty()
    print(f"  universe: {close_m.shape[1]} stocks | months: {close_m.index[0].date()}→{close_m.index[-1].date()}", flush=True)
    print(f"  params (a-priori): top-{TOP_N}, {LB}-{SKIP}mo, liquid top-{TURN_TOP}, price≥₹{PRICE_MIN:.0f}, "
          f"slippage {SLIP*100:.2f}%/side", flush=True)
    eq, inv, n = simulate(close_m, turn_m, nifty_m, use_regime=True)
    report(eq, nifty_m)
    print(f"  Invested {inv}/{n} months ({inv/n*100:.0f}%)")

    # ── Robustness / stress tests (does the edge survive harsher assumptions?) ──
    print(f"\n{'='*74}\n  ROBUSTNESS — OOS (2019-2026) CAGR under harsher / varied assumptions\n{'='*74}")
    print(f"  {'variant':<40}{'OOS CAGR':>12}{'OOS MaxDD':>12}")
    def oos_stat(**kw):
        e, _, _ = simulate(close_m, turn_m, nifty_m, **kw)
        oe = e[e.index.year > IS_END]
        c, d = metrics(oe, e[e.index.year <= IS_END].iloc[-1])
        return c, d
    for label, kw in [
        ("base (slip 0.10%/side, top-15, liq500)", dict()),
        ("slip 0.25%/side (illiquid + delist proxy)", dict(rt_cost=2*(0.0025+STT)+0.0005)),
        ("slip 0.50%/side (harsh)",                  dict(rt_cost=2*(0.0050+STT)+0.0005)),
        ("top-10 (more concentrated)",               dict(top_n=10)),
        ("top-20 (more diversified)",                dict(top_n=20)),
        ("liquid top-300 (more liquid only)",        dict(turn_top=300)),
        ("liquid top-750 (deeper/less liquid)",      dict(turn_top=750)),
        ("NO regime filter (always invested)",       dict(use_regime=False)),
    ]:
        c, d = oos_stat(**kw)
        print(f"  {label:<40}{c*100:>+11.1f}%{d*100:>11.0f}%")


if __name__ == "__main__":
    main()
