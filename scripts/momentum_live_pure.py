"""
Signal generator for System 4 (pure MID-cap momentum + regime cash defense) and
System 5 (pure MID-cap momentum, always invested). PAPER/SIGNAL ONLY.

Both systems trade only Indian equities — no gold, no US. Exact same MID-cap
momentum logic validated in scripts/validate_pure_momentum.py.

Run:  python scripts/momentum_live_pure.py            (emit S4 + S5 signals)
      python scripts/momentum_live_pure.py --dry-run  (same; no files written)
"""
from __future__ import annotations
import sys, glob, os, argparse, json
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np, pandas as pd, duckdb

DAILY = "data_store/eod2/src/eod2_data/daily"
DB    = "data_store/piedpiper.duckdb"
SIGNAL_S4 = "data_store/live_signal_s4.json"
SIGNAL_S5 = "data_store/live_signal_s5.json"
TOP_N=15; TURN_TOP=500; PRICE_MIN=30.0; VOL_FLOOR=0.02

def log(msg):
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} | {msg}", flush=True)

def build_monthly():
    """Also returns the newest RAW daily date. resample("ME") labels the last
    available bar as the month-end, so a late EOD feed silently shifts the whole
    signal onto the wrong close -- the caller must validate this."""
    closes, turns = {}, {}; last_daily = None
    for f in glob.glob(f"{DAILY}/*.csv"):
        sym = Path(f).stem.upper()
        try: d = pd.read_csv(f, usecols=["Date","Close","Volume","Series"])
        except Exception: continue
        d = d[d["Series"]=="EQ"]
        if len(d) < 260: continue
        d["Date"] = pd.to_datetime(d["Date"]); d = d.set_index("Date").sort_index()
        if last_daily is None or d.index[-1] > last_daily: last_daily = d.index[-1]
        closes[sym] = d["Close"].resample("ME").last()
        turns[sym]  = (d["Close"]*d["Volume"]).resample("ME").mean()
    close = pd.DataFrame(closes).sort_index()
    turn  = pd.DataFrame(turns).reindex(close.index)
    return close, turn, last_daily

def _last_trading_day(asof):
    """The true final TRADING day of asof's month (month-end may be a weekend/holiday)."""
    try:
        hol = set(json.load(open("data_store/nse_holidays.json")))
    except Exception:
        hol = set()
    d = pd.Timestamp(asof)
    for _ in range(10):
        if d.weekday() < 5 and d.strftime("%Y-%m-%d") not in hol:
            return d
        d -= pd.Timedelta(days=1)
    return d

def assert_month_complete(asof, last_daily):
    """Abort rather than emit a signal built on an incomplete final month.

    resample("ME") labels whatever bar it last saw as the month-end, so a stalled
    EOD feed silently ranks the universe on the 29th's closes and stamps it the 30th
    -- different picks, no warning. Comparing against the true last TRADING day (not
    a raw business-day gap) is what distinguishes "month ended on a Saturday" from
    "the feed missed Wednesday". A missed month is recoverable; a wrong month is not."""
    if last_daily is None:
        raise SystemExit("ABORT: no daily data available to validate month completeness")
    ltd = _last_trading_day(asof)
    if pd.Timestamp(last_daily).normalize() < ltd.normalize():
        raise SystemExit(
            f"ABORT: month-end data incomplete -- newest bar {pd.Timestamp(last_daily).date()} "
            f"is before the month's last trading day {ltd.date()} (month-end {pd.Timestamp(asof).date()}). "
            f"Re-run once the EOD feed catches up.")
    log(f"Month-end data check OK (newest bar {pd.Timestamp(last_daily).date()} >= "
        f"last trading day {ltd.date()})")

def load_nifty():
    c = duckdb.connect(DB, read_only=True)
    n = c.execute("SELECT dt,close FROM adjusted_ohlcv WHERE symbol='Nifty 50' ORDER BY dt").df(); c.close()
    n["dt"] = pd.to_datetime(n["dt"])
    return n.set_index("dt")["close"].resample("ME").last()

def is_stock(sym):
    s = str(sym).upper()
    return not (s.endswith("BEES") or s.endswith("ETF") or s.endswith("IETF")
                or any(p in s for p in ("LIQUID","GILT","GSEC","BHARATBOND","CASHIETF")))

def pick_stocks(close, turn, p, prev, m3, m6, m12, vol, ma10):
    """MID-cap tier (60th-90th pct turnover), multi-TF risk-adj momentum,
    trend confirm (price > 10mo MA), top-15 with rank buffer.
    Excludes cash/liquid/bond ETFs via vol floor + name filter."""
    pr = close.iloc[p]; tn = turn.iloc[p]
    hist = close.iloc[:p+1].notna().sum(); v = vol.iloc[p]
    base = pr.index[(pr>=PRICE_MIN)&(hist>=14)&tn.notna()&(tn>0)&(v>=VOL_FLOOR)]
    base = [s for s in base if is_stock(s)]
    if len(base) < 100: return []
    rank = tn[base].rank(pct=True)
    mid  = list(rank.index[(rank>0.60)&(rank<=0.90)])              # MID-cap tier
    elig = [s for s in mid if pr[s] > ma10.iloc[p].get(s, 1e9)]   # trend confirm
    score = pd.Series(0.0, index=elig); nc = 0
    for sg in (m3, m6, m12):
        x = (sg.iloc[p][elig] / vol.iloc[p][elig].replace(0,np.nan)).dropna()
        if len(x): score = score.add(x.rank(pct=True), fill_value=0); nc += 1
    if nc == 0: return []
    ranked = (score/nc).sort_values(ascending=False)
    rk = {s:i for i,s in enumerate(ranked.index)}
    picks = [s for s in prev if rk.get(s, 10**9) < TOP_N*1.67]    # keep incumbents in top N*1.67
    for s in ranked.index:
        if len(picks) >= TOP_N: break
        if s not in picks: picks.append(s)
    return picks[:TOP_N]

def regime_check(close, turn, nifty, p):
    """Nifty > 10mo-EMA AND 6mo return > -2% AND breadth > 45%."""
    idx  = close.index; nser = nifty.reindex(idx).ffill()
    nema = nser.ewm(span=10, adjust=False).mean()
    nifty_ok = (nser.iloc[p] > nema.iloc[p]) and (nser.iloc[p]/nser.iloc[p-6]-1 > -0.02)
    ma10 = close.rolling(10).mean(); pr = close.iloc[p]; tn = turn.iloc[p]
    base = pr.index[(pr>=PRICE_MIN)&tn.notna()]
    liq  = tn[base].nlargest(min(TURN_TOP, len(base))).index
    breadth = float((pr[liq] > ma10.iloc[p][liq]).mean())
    return nifty_ok, breadth, (nifty_ok and breadth > 0.45)

def eod2_close(sym):
    f = Path(DAILY)/f"{sym.lower()}.csv"
    if not f.exists(): return None
    try:
        d = pd.read_csv(f, usecols=["Close","Series"]); d = d[d["Series"]=="EQ"]
        return round(float(d["Close"].iloc[-1]), 2) if len(d) else None
    except Exception: return None

def _prev_picks(path, asof_str):
    try:
        if os.path.exists(path):
            prev = json.load(open(path))
            if prev.get("as_of") != asof_str:
                return prev.get("momentum_picks") or []
    except Exception: pass
    return []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    log("="*60)
    log("Pure Momentum signals — S4 (regime-gated) + S5 (always invested) [PAPER]")
    log("Building monthly panel from eod2 ...")
    close, turn, last_daily = build_monthly()
    cur  = pd.Timestamp.now().to_period("M")
    keep = close.index.to_period("M") < cur
    close, turn = close[keep], turn[keep]
    nifty = load_nifty()
    p     = len(close.index) - 1
    asof  = close.index[p]
    assert_month_complete(asof, last_daily)
    log(f"As-of month: {asof.date()} | universe {close.shape[1]} stocks")

    ret  = close.pct_change(); vol = ret.rolling(6).std()
    m3   = close.shift(1)/close.shift(4)-1
    m6   = close.shift(1)/close.shift(7)-1
    m12  = close.shift(1)/close.shift(13)-1
    ma10 = close.rolling(10).mean()

    nifty_ok, breadth, risk_on = regime_check(close, turn, nifty, p)
    log(f"Regime: Nifty_uptrend={nifty_ok} breadth={breadth:.0%} → {'RISK-ON' if risk_on else 'RISK-OFF'}")

    asof_str = str(asof.date())
    # rank buffer uses S5's prior picks (S5 is always invested, so its pick history
    # is the most stable reference; S4 uses the same picks when it's risk-ON)
    prev_picks = _prev_picks(SIGNAL_S5, asof_str)

    # both systems share the same stock universe and pick logic
    picks = pick_stocks(close, turn, p, prev_picks, m3, m6, m12, vol, ma10)
    log(f"Top-{TOP_N} pure MID-cap: {picks[:5]}{'...' if len(picks)>5 else ''}" if picks else "No picks found")

    entry = {s: eod2_close(s) for s in picks}
    entry = {k:v for k,v in entry.items() if v}

    sig4 = {"as_of": asof_str, "generated": str(datetime.now()),
            "risk_on": bool(risk_on), "breadth": round(breadth,3),
            "allocation": {"momentum":1.0} if (risk_on and picks) else {},
            "cash": 0.0 if (risk_on and picks) else 1.0,
            "momentum_picks": picks if risk_on else [],
            "entry_prices": {s:v for s,v in entry.items() if s in picks} if risk_on else {},
            "mode": "PAPER", "system": "S4-pure-momentum-regime"}

    sig5 = {"as_of": asof_str, "generated": str(datetime.now()),
            "risk_on": bool(risk_on), "breadth": round(breadth,3),
            "allocation": {"momentum":1.0} if picks else {},
            "cash": 0.0 if picks else 1.0,
            "momentum_picks": picks, "entry_prices": entry,
            "mode": "PAPER", "system": "S5-pure-momentum-always"}

    log("── TARGET PORTFOLIOS ──")
    log(f"  S4 (regime-gated): {'INVESTED in ' + str(len(sig4['momentum_picks'])) + ' stocks' if risk_on else 'CASH (regime OFF)'}")
    log(f"  S5 (always-on):    {len(picks)} stocks → {picks[:5]}{'...' if len(picks)>5 else ''}")

    if not args.dry_run:
        os.makedirs("data_store", exist_ok=True)
        json.dump(sig4, open(SIGNAL_S4,"w"), indent=2)
        json.dump(sig5, open(SIGNAL_S5,"w"), indent=2)
        log(f"Written: {SIGNAL_S4}  {SIGNAL_S5}")
    else:
        log("DRY-RUN: files NOT written")
    log("Done. PAPER mode — no orders placed.")

if __name__ == "__main__": main()
