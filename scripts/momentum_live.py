"""
LIVE monthly signal generator — the validated multi-asset momentum system.
PAPER/SIGNAL ONLY: prints & logs this month's target portfolio; places NO orders.

Implements the validated logic:
  - momentum-equity: multi-TF risk-adjusted momentum + trend confirm + buffer,
    broad point-in-time liquid universe, top-15
  - regime: Nifty AND breadth healthy -> risk-ON (momentum); else risk-OFF (gold/cash)
  - multi-asset: risk-parity (alpha-tilt=2) across momentum-equity / gold / US-Nasdaq / cash

Run:  python scripts/momentum_live.py            (build panel from eod2, emit signal)
      python scripts/momentum_live.py --dry-run  (same; explicit no-op safety)
"""
from __future__ import annotations
import sys, glob, os, argparse, json
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np, pandas as pd

DAILY="data_store/eod2/src/eod2_data/daily"; DB="data_store/piedpiper.duckdb"
LOGDIR="logs"; SIGNAL_JSON="data_store/live_signal.json"
TOP_N=15; TURN_TOP=500; PRICE_MIN=30.0; ALPHA_TILT=2.0

def log(msg):
    # print only; the cron ">> logs/momentum_live.log" redirect handles file capture
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} | {msg}", flush=True)

def build_monthly():
    """Build monthly close + turnover panels from eod2 daily CSVs (latest data).

    Also returns the latest RAW daily date seen. resample("ME") silently labels the
    last available bar as the month-end, so if the EOD feed is a day late the whole
    signal is built off the wrong close without any visible symptom — the caller
    must check this against the true month-end before trusting the panel."""
    closes, turns = {}, {}; last_daily=None
    for f in glob.glob(f"{DAILY}/*.csv"):
        sym=Path(f).stem.upper()
        try: d=pd.read_csv(f, usecols=["Date","Close","Volume","Series"])
        except Exception: continue
        d=d[d["Series"]=="EQ"]
        if len(d)<260: continue
        d["Date"]=pd.to_datetime(d["Date"]); d=d.set_index("Date").sort_index()
        if last_daily is None or d.index[-1]>last_daily: last_daily=d.index[-1]
        closes[sym]=d["Close"].resample("ME").last()
        turns[sym]=(d["Close"]*d["Volume"]).resample("ME").mean()
    close=pd.DataFrame(closes).sort_index(); turn=pd.DataFrame(turns).reindex(close.index)
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
    import duckdb
    c=duckdb.connect(DB,read_only=True)
    n=c.execute("SELECT dt,close FROM adjusted_ohlcv WHERE symbol='Nifty 50' ORDER BY dt").df();c.close()
    n["dt"]=pd.to_datetime(n["dt"]); return n.set_index("dt")["close"].resample("ME").last()

def px_monthly(ticker):
    import yfinance as yf
    d=yf.download(ticker,start="2010-06-01",auto_adjust=True,progress=False)
    s=d["Close"].squeeze().dropna(); s.index=pd.to_datetime(s.index); return s.resample("ME").last()

VOL_FLOOR=0.02   # exclude near-zero-vol instruments (cash/liquid/gilt/bond ETFs — NOT stocks)
def is_stock(sym):  # exclude equity ETFs / funds by name (belt-and-suspenders)
    s=str(sym).upper()
    return not (s.endswith("BEES") or s.endswith("ETF") or s.endswith("IETF")
                or any(p in s for p in ("LIQUID","GILT","GSEC","BHARATBOND","CASHIETF")))

def momentum_picks(close, turn, p, prev_picks=None):
    """Multi-TF risk-adjusted momentum + trend confirm + rank buffer (matches backtest).
    Excludes cash/liquid/bond ETFs (vol floor + name filter) so it picks real STOCKS."""
    ret=close.pct_change(); vol=ret.rolling(6).std()
    m3=close.shift(1)/close.shift(4)-1; m6=close.shift(1)/close.shift(7)-1; m12=close.shift(1)/close.shift(13)-1
    ma10=close.rolling(10).mean()
    price=close.iloc[p]; tn=turn.iloc[p]; hist=close.iloc[:p+1].notna().sum(); v=vol.iloc[p]
    base=price.index[(price>=PRICE_MIN)&(hist>=14)&tn.notna()&(v>=VOL_FLOOR)]   # vol floor removes cash-like
    base=[s for s in base if is_stock(s)]                                        # name filter removes ETFs
    elig=[s for s in tn[base].nlargest(min(TURN_TOP,len(base))).index
          if pd.notna(ma10.iloc[p].get(s)) and price[s]>ma10.iloc[p][s]]
    score=pd.Series(0.0,index=elig); nc=0
    for sg in (m3,m6,m12):
        s=(sg.iloc[p][elig]/vol.iloc[p][elig].replace(0,np.nan)).dropna()
        if len(s): score=score.add(s.rank(pct=True),fill_value=0); nc+=1
    if nc==0: return []
    ranked=(score/nc).sort_values(ascending=False)
    if prev_picks:                                    # buffer: keep incumbents still in top N*1.67
        rk={s:i for i,s in enumerate(ranked.index)}
        picks=[s for s in prev_picks if rk.get(s,10**9)<TOP_N*1.67]
        for s in ranked.index:
            if len(picks)>=TOP_N: break
            if s not in picks: picks.append(s)
        return picks[:TOP_N]
    return list(ranked.head(TOP_N).index)

def regime_ok(close, turn, nifty, p):
    idx=close.index; nser=nifty.reindex(idx).ffill(); nema=nser.ewm(span=10,adjust=False).mean()
    nifty_ok=(nser.iloc[p]>nema.iloc[p]) and (nser.iloc[p]/nser.iloc[p-6]-1 > -0.02)
    ma10=close.rolling(10).mean(); pr=close.iloc[p]; tn=turn.iloc[p]
    base=pr.index[(pr>=PRICE_MIN)&tn.notna()]
    liq=tn[base].nlargest(min(TURN_TOP,len(base))).index
    breadth=float((pr[liq]>ma10.iloc[p][liq]).mean())
    return nifty_ok, breadth, (nifty_ok and breadth>0.45)

def alloc_weights(risk_on, gold_px, us_px, wmom=0.70):
    """REGIME-CONDITIONAL (validated best config, scripts/backtest_regime_conditional.py):
    risk-ON  -> heavy momentum (70 / gold 15 / US 15), gold & US trend-gated (off->cash);
    risk-OFF -> defensive (gold 45 / US 35 / cash 20), trend-gated."""
    def on(px): px=px.dropna(); return bool(float(px.iloc[-1])>float(px.ewm(span=10,adjust=False).mean().iloc[-1]))
    g_on, u_on = on(gold_px), on(us_px)
    w={}; cash=0.0
    if risk_on:
        w["momentum"]=wmom; rest=(1-wmom)/2
        if g_on: w["gold"]=rest
        else:    cash+=rest
        if u_on: w["us_nasdaq"]=rest
        else:    cash+=rest
    else:
        if g_on: w["gold"]=0.45
        else:    cash+=0.45
        if u_on: w["us_nasdaq"]=0.35
        else:    cash+=0.35
        cash+=0.20
    return w, cash

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dry-run",action="store_true")
    args=ap.parse_args()
    log("="*60); log("Momentum LIVE signal (PAPER — no orders placed)")
    log("Building monthly panel from eod2 ...")
    close, turn, last_daily = build_monthly()
    # use only FULLY-COMPLETE months (drop the current, in-progress month)
    cur = pd.Timestamp.now().to_period("M")
    keep = close.index.to_period("M") < cur
    close, turn = close[keep], turn[keep]
    nifty = load_nifty()
    p = len(close.index)-1                      # last completed month
    asof = close.index[p]
    assert_month_complete(asof, last_daily)
    log(f"As-of month: {asof.date()} | universe {close.shape[1]} stocks")

    nifty_ok, breadth, risk_on = regime_ok(close, turn, nifty, p)
    log(f"Regime: Nifty_uptrend={nifty_ok} breadth={breadth:.0%} -> {'RISK-ON' if risk_on else 'RISK-OFF'}")

    # last month's picks (for the turnover buffer) — read BEFORE overwriting the signal
    prev_picks=None
    try:
        if os.path.exists(SIGNAL_JSON):
            prev=json.load(open(SIGNAL_JSON))
            if prev.get("as_of")!=str(asof.date()):     # only use if it's a genuinely prior month
                prev_picks=prev.get("momentum_picks") or None
    except Exception: pass
    picks = momentum_picks(close, turn, p, prev_picks) if risk_on else []
    # multi-asset weights
    try:
        gold=px_monthly("GOLDBEES.NS"); us=px_monthly("MON100.NS")
        gold=gold[gold.index.to_period("M")<cur]; us=us[us.index.to_period("M")<cur]  # align to same as-of month (drop incomplete)
        if gold.empty or us.empty: raise ValueError("gold/us empty after aligning to as-of month")
        w, cash = alloc_weights(risk_on, gold, us)
    except Exception as e:
        log(f"WARN multi-asset weights failed ({e}); defaulting to momentum/cash only")
        w, cash = ({"momentum":1.0} if risk_on else {}), (0.0 if risk_on else 1.0)

    # record entry prices (for the monitor's mark-to-market P&L)
    entry={}
    for s in picks:
        try: entry[s]=round(float(close.iloc[p][s]),2)
        except Exception: pass
    try: entry["gold"]=round(float(gold.dropna().iloc[-1]),2)
    except Exception: pass
    try: entry["us_nasdaq"]=round(float(us.dropna().iloc[-1]),2)
    except Exception: pass
    signal={"as_of":str(asof.date()), "generated":str(datetime.now()), "risk_on":bool(risk_on),
            "breadth":round(breadth,3), "allocation":{k:round(v,3) for k,v in w.items()},
            "cash":round(cash,3), "momentum_picks":picks, "entry_prices":entry, "mode":"PAPER"}
    log("── TARGET PORTFOLIO (paper) ──")
    for k,v in w.items(): log(f"   {k:12s} {v*100:5.1f}%")
    log(f"   {'cash':12s} {cash*100:5.1f}%")
    if picks: log(f"   momentum top-{TOP_N}: {', '.join(picks)}")
    else: log("   momentum: RISK-OFF (no stock picks this month)")
    if not args.dry_run:
        os.makedirs(os.path.dirname(SIGNAL_JSON), exist_ok=True)
        with open(SIGNAL_JSON,"w") as f: json.dump(signal,f,indent=2)
        log(f"Signal written to {SIGNAL_JSON}")
    else:
        log("DRY-RUN: signal NOT written")
    log("Done. PAPER mode — execute manually. No orders placed.")

if __name__=="__main__": main()
