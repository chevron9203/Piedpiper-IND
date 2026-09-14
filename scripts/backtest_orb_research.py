"""
ORB improvement research — walk-forward, honest single account.
Tune ONLY on IS (2020-2023); report OOS (2024-2026) as the realistic estimate.

Structural improvements baked in (validated earlier): LIMIT entry (no entry
slippage), single compounding account, MIS-margin cap, 0.10%/side exit slippage
+ gap-through stops, LONG-only.

Grid (tuned on IS): target_mult × vol_surge × min_range × max_per_day × risk.
Selection metric: IS Calmar (CAGR / |MaxDD|). We then read that config's OOS —
and show the median OOS of the top-20 IS configs as an overfitting check.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import pandas as pd
import strategies.orb_intraday as orb
from strategies.orb_intraday import backtest_orb_on_candles, _intraday_charges
from scripts.backtest_orb_realistic import load, CAPITAL

IS_END, LEVERAGE, SLIP = 2023, 5.0, 0.0010


def gen(df, nc, ne, target_mult):
    t0 = orb.TARGET_MULT
    orb.TARGET_MULT = target_mult
    out = []
    try:
        syms = df["symbol"].unique().tolist()
        for i, sym in enumerate(syms):
            sd = df[df["symbol"] == sym].set_index("dt").sort_index()[["open","high","low","close","volume"]]
            if len(sd) < 100:
                continue
            t = backtest_orb_on_candles(sym, sd, capital=CAPITAL, nifty_close=nc, nifty_ema20=ne,
                                        direction="LONG", stock_ema_filter=True,
                                        slippage_pct=SLIP, gap_through=True, entry_limit=True)
            if not t.empty:
                out.append(t)
    finally:
        orb.TARGET_MULT = t0
    tr = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    if not tr.empty:
        tr = tr[tr["direction"] == "LONG"].copy()
        tr["trade_date"] = pd.to_datetime(tr["trade_date"])
        tr["risk_share"] = tr["entry_price"] - tr["stop_loss"]
        tr = tr[tr["risk_share"] > 0]
    print(f"    target {target_mult}: {len(tr)} trades", flush=True)
    return tr


def simulate(trades, risk, max_per_day):
    equity, curve, ntr, wins = CAPITAL, {}, 0, 0
    for d in sorted(trades["trade_date"].unique()):
        day = trades[trades["trade_date"] == d].sort_values("vol_ratio", ascending=False).head(max_per_day)
        used, day_pnl = 0.0, 0.0
        for _, r in day.iterrows():
            qty = int((equity * risk) / r["risk_share"])
            free = equity * LEVERAGE - used
            if qty * r["entry_price"] > free:
                qty = int(free / r["entry_price"]) if r["entry_price"] > 0 else 0
            if qty < 1:
                continue
            net = (r["exit_price"] - r["entry_price"]) * qty - _intraday_charges(r["entry_price"]*qty, r["exit_price"]*qty)
            day_pnl += net; used += qty * r["entry_price"]; ntr += 1; wins += 1 if net > 0 else 0
        equity += day_pnl
        curve[d] = equity
        if equity <= 0:
            break
    return pd.Series(curve).sort_index(), ntr, wins


def _metrics(eq, start):
    if eq.empty or len(eq) < 2:
        return float("nan"), float("nan"), float("nan")
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.1)
    cagr = (eq.iloc[-1] / start) ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1.0
    dd = ((eq - eq.cummax()) / eq.cummax()).min()
    rets = eq.pct_change().dropna()
    sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else float("nan")
    return cagr, dd, sharpe


def evaluate(trades, risk, max_per_day):
    eq, ntr, wins = simulate(trades, risk, max_per_day)
    if eq.empty:
        return None
    is_eq, oos_eq = eq[eq.index.year <= IS_END], eq[eq.index.year > IS_END]
    if is_eq.empty or oos_eq.empty:
        return None
    is_c, is_dd, is_sh = _metrics(is_eq, CAPITAL)
    oos_c, oos_dd, oos_sh = _metrics(oos_eq, is_eq.iloc[-1])
    return dict(ntr=ntr, win=wins/ntr if ntr else 0, is_cagr=is_c, is_dd=is_dd, is_sharpe=is_sh,
                oos_cagr=oos_c, oos_dd=oos_dd, oos_sharpe=oos_sh,
                calmar_is=(is_c / abs(is_dd)) if is_dd < 0 else float("nan"))


def main():
    df, nc, ne = load()
    print("Generating trade sets (entry_limit, target 1.5/2.0/2.5) ...", flush=True)
    gens = {tm: gen(df, nc, ne, tm) for tm in (1.5, 2.0, 2.5)}

    rows = []
    for tm, base in gens.items():
        if base.empty:
            continue
        for vol in (3, 4, 5, 6):
            for rng in (0.3, 0.5, 0.7):
                f = base[(base["vol_ratio"] >= vol) & (base["range_pct"] >= rng)]
                if len(f) < 40:
                    continue
                for maxday in (2, 3):
                    for risk in (0.010, 0.015):
                        r = evaluate(f, risk, maxday)
                        if r:
                            r.update(target=tm, vol=vol, rng=rng, maxday=maxday, risk=risk)
                            rows.append(r)
    res = pd.DataFrame(rows)
    res = res[res["is_cagr"].notna() & np.isfinite(res["calmar_is"])]

    # Select on IS only
    res = res.sort_values("calmar_is", ascending=False).reset_index(drop=True)

    print(f"\n{'='*118}")
    print("  TOP 15 CONFIGS BY IN-SAMPLE CALMAR  (selected on IS 2020-2023; OOS 2024-2026 is untouched)")
    print(f"{'='*118}")
    print(f"  {'tgt':>3} {'vol':>3} {'rng':>4} {'day':>3} {'risk':>5} | {'trades':>6} {'win':>5} | "
          f"{'IS_CAGR':>8} {'IS_DD':>7} {'IS_Sh':>6} | {'OOS_CAGR':>8} {'OOS_DD':>7} {'OOS_Sh':>6}")
    print(f"  {'-'*112}")
    for _, r in res.head(15).iterrows():
        print(f"  {r['target']:>3.1f} {int(r['vol']):>3} {r['rng']:>4.1f} {int(r['maxday']):>3} {r['risk']*100:>4.1f}% | "
              f"{int(r['ntr']):>6} {r['win']*100:>4.1f}% | "
              f"{r['is_cagr']*100:>+7.1f}% {r['is_dd']*100:>6.1f}% {r['is_sharpe']:>6.2f} | "
              f"{r['oos_cagr']*100:>+7.1f}% {r['oos_dd']*100:>6.1f}% {r['oos_sharpe']:>6.2f}")

    best = res.iloc[0]
    top20 = res.head(20)
    print(f"\n{'='*118}")
    print("  VERDICT")
    print(f"{'='*118}")
    print(f"  Best-on-IS config: target={best['target']}, vol>={int(best['vol'])}x, "
          f"range>={best['rng']}%, max/day={int(best['maxday'])}, risk={best['risk']*100:.1f}%")
    print(f"    IS  (2020-2023, TRAINED): CAGR {best['is_cagr']*100:+.1f}%  MaxDD {best['is_dd']*100:.1f}%  Sharpe {best['is_sharpe']:.2f}")
    print(f"    OOS (2024-2026, REAL)   : CAGR {best['oos_cagr']*100:+.1f}%  MaxDD {best['oos_dd']*100:.1f}%  Sharpe {best['oos_sharpe']:.2f}")
    print(f"    Trades: {int(best['ntr'])} over 6.6y (~{int(best['ntr']/6.6)}/yr) | Win {best['win']*100:.1f}%")
    print(f"\n  Overfitting check — across the TOP-20 IS configs:")
    print(f"    Median OOS CAGR: {top20['oos_cagr'].median()*100:+.1f}%   "
          f"| OOS positive in {int((top20['oos_cagr']>0).sum())}/20 configs   "
          f"| Median OOS MaxDD: {top20['oos_dd'].median()*100:.1f}%")


if __name__ == "__main__":
    main()
