"""
ORB improvement experiments — HONEST single-account, IS/OOS disciplined.
Tests structural changes against the honest baseline (which loses money):
  REF : honest market entry (the losing baseline)
  L   : patient LIMIT entry (removes entry slippage)
  L5  : limit entry + stricter 5x vol surge (fewer, higher-conviction trades)
  L5R : L5 + wider 2.5x target (let winners run to offset cost drag)
All: 0.10%/side exit slippage + gap-through stops, LONG-only, ₹2L, 2020-2026.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd
import strategies.orb_intraday as orb
from strategies.orb_intraday import backtest_orb_on_candles
from scripts.backtest_orb_realistic import load, simulate, _cagr, _maxdd, CAPITAL, IS_END, SLIPPAGE


def gen(df, nc, ne, entry_limit, vol_mult=None, target_mult=None):
    v0, t0 = orb.VOL_SURGE_MULT, orb.TARGET_MULT
    if vol_mult is not None:    orb.VOL_SURGE_MULT = vol_mult
    if target_mult is not None: orb.TARGET_MULT = target_mult
    out = []
    try:
        syms = df["symbol"].unique().tolist()
        for i, sym in enumerate(syms):
            sd = df[df["symbol"] == sym].set_index("dt").sort_index()[["open","high","low","close","volume"]]
            if len(sd) < 100:
                continue
            t = backtest_orb_on_candles(sym, sd, capital=CAPITAL, nifty_close=nc, nifty_ema20=ne,
                                        direction="LONG", stock_ema_filter=True,
                                        slippage_pct=SLIPPAGE, gap_through=True, entry_limit=entry_limit)
            if not t.empty:
                out.append(t)
            if (i + 1) % 50 == 0:
                print(f"    [{i+1}/{len(syms)}]", flush=True)
    finally:
        orb.VOL_SURGE_MULT, orb.TARGET_MULT = v0, t0
    tr = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    if not tr.empty:
        tr = tr[tr["direction"] == "LONG"].copy()
        tr["trade_date"] = pd.to_datetime(tr["trade_date"])
        tr["risk_share"] = tr["entry_price"] - tr["stop_loss"]
        tr = tr[tr["risk_share"] > 0]
    return tr


def rpt(label, tr, risk=0.015):
    if tr.empty:
        print(f"  {label:<34} no trades"); return
    eq, ntr, wins, ruin = simulate(tr, risk)
    is_eq  = eq[eq.index.year <= IS_END]
    oos_eq = eq[eq.index.year > IS_END]
    oos0 = is_eq.iloc[-1] if not is_eq.empty else CAPITAL
    full_c = _cagr(eq, CAPITAL) * 100
    is_c   = _cagr(is_eq, CAPITAL) * 100 if not is_eq.empty else float("nan")
    oos_c  = _cagr(oos_eq, oos0) * 100 if not oos_eq.empty else float("nan")
    flag = "  ⚠️RUIN" if ruin else ""
    print(f"  {label:<34} trades {ntr:>4} | win {wins/ntr*100:4.1f}% | "
          f"Full {full_c:+6.1f}% | IS {is_c:+6.1f}% | OOS {oos_c:+6.1f}% | "
          f"MaxDD {_maxdd(eq)*100:5.1f}% | Final ₹{eq.iloc[-1]:>10,.0f}{flag}")


def main():
    df, nc, ne = load()
    print("gen REF (market entry) ...", flush=True);      ref = gen(df, nc, ne, entry_limit=False)
    print("gen L (limit entry) ...", flush=True);          l   = gen(df, nc, ne, entry_limit=True)
    print("gen L5 (limit + 5x vol) ...", flush=True);      l5  = gen(df, nc, ne, entry_limit=True, vol_mult=5.0)
    print("gen L5R (limit + 5x vol + 2.5x tgt) ...", flush=True); l5r = gen(df, nc, ne, entry_limit=True, vol_mult=5.0, target_mult=2.5)
    print(f"\n{'='*112}\n  ORB IMPROVEMENT TESTS — honest single account, 1.5% risk  (OOS = untouched 2024-2026)\n{'='*112}")
    rpt("REF  market entry (baseline)", ref)
    rpt("L    limit entry", l)
    rpt("L5   limit + 5x vol", l5)
    rpt("L5R  limit + 5x vol + 2.5x target", l5r)


if __name__ == "__main__":
    main()
