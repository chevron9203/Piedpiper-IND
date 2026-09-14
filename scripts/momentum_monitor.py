"""
Daily position/P&L monitor — VISIBILITY ONLY (paper, no trading).
Reads the current target (live_signal.json), marks each holding to the latest
price, shows per-position + total P&L since the signal date, and watches for a
regime flip (Nifty trend) so you're never surprised. Logs a daily snapshot.
"""
from __future__ import annotations
import sys, json, glob, os
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd, duckdb

SIGNAL="data_store/live_signal.json"; DAILY="data_store/eod2/src/eod2_data/daily"
DB="data_store/piedpiper.duckdb"; LOGDIR="logs"; OUT="data_store/monitor_status.json"

def log(m):
    # print only; the cron ">> logs/momentum_monitor.log" redirect handles file capture
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} | {m}", flush=True)

def eod2_last(sym):
    f=f"{DAILY}/{sym.lower()}.csv"
    if not os.path.exists(f): return None
    try:
        d=pd.read_csv(f,usecols=["Close","Series"]); d=d[d["Series"]=="EQ"]
        return float(d["Close"].iloc[-1]) if len(d) else None
    except Exception: return None

def etf_last(ticker):
    try:
        import yfinance as yf
        d=yf.download(ticker,period="10d",auto_adjust=True,progress=False)
        return float(d["Close"].squeeze().dropna().iloc[-1])
    except Exception: return None

def nifty_regime_now():
    c=duckdb.connect(DB,read_only=True)
    n=c.execute("SELECT dt,close FROM adjusted_ohlcv WHERE symbol='Nifty 50' ORDER BY dt").df();c.close()
    n["dt"]=pd.to_datetime(n["dt"]); m=n.set_index("dt")["close"].resample("ME").last()
    ema=m.ewm(span=10,adjust=False).mean()
    return bool(m.iloc[-1]>ema.iloc[-1]), float(m.iloc[-1]), float(ema.iloc[-1])

def main():
    if not os.path.exists(SIGNAL):
        log("No live_signal.json yet — run momentum_live.py first."); return
    sig=json.load(open(SIGNAL))
    # use PAPER execution entries (stamped at inception/rebalance) so monitor P&L
    # matches the dashboard exactly — NOT the month-end signal price
    POS="data_store/paper_positions.json"
    _pos=json.load(open(POS)) if os.path.exists(POS) else {}
    entry=_pos.get("entries",{}) or sig.get("entry_prices",{})
    inception=_pos.get("inception", sig["as_of"])
    log("="*60); log(f"MONITOR (paper) | signal as-of {sig['as_of']} | since {inception} | risk_on={sig['risk_on']}")
    alloc=sig.get("allocation",{}); picks=sig.get("momentum_picks",[])
    rows=[]; port_pl=0.0
    # ETF sleeves
    for k,tk in [("gold","GOLDBEES.NS"),("us_nasdaq","MON100.NS")]:
        if k in alloc:
            now=etf_last(tk); e=entry.get(k)
            if now and e:
                r=now/e-1; port_pl+=alloc[k]*r
                rows.append((k, alloc[k]*100, e, now, r*100))
    # momentum stock sleeve (equal weight within the momentum allocation)
    mom_w=alloc.get("momentum",0.0)
    if picks and mom_w>0:
        per=mom_w/len(picks)
        for s in picks:
            now=eod2_last(s); e=entry.get(s)
            if now and e:
                r=now/e-1; port_pl+=per*r; rows.append((s, per*100, e, now, r*100))
    log("── positions (paper) ──")
    for name,wt,e,now,pl in rows:
        log(f"   {name:<12} {wt:4.1f}%  entry {e:>9.2f}  now {now:>9.2f}  {pl:+6.1f}%")
    log(f"   cash {sig.get('cash',0)*100:.1f}%")
    log(f"   PORTFOLIO P&L since {inception}: {port_pl*100:+.2f}% (paper)")
    # regime watch
    nok,nv,ne=nifty_regime_now()
    if nok!=sig["risk_on"]:
        log(f"⚠️ REGIME MAY BE SHIFTING: Nifty now {'ABOVE' if nok else 'BELOW'} trend ({nv:.0f} vs EMA {ne:.0f}) "
            f"but signal is risk_{'ON' if sig['risk_on'] else 'OFF'} — full re-check at month-end.")
    else:
        log(f"regime stable (Nifty {'above' if nok else 'below'} trend; {nv:.0f} vs {ne:.0f})")
    json.dump({"checked":str(datetime.now()),"as_of":sig["as_of"],"portfolio_pl_pct":round(port_pl*100,2),
               "positions":[{"name":n,"wt_pct":round(w,1),"pl_pct":round(p,1)} for n,w,e,no,p in rows],
               "regime_shift_watch":bool(nok!=sig["risk_on"])}, open(OUT,"w"), indent=2)
    log(f"snapshot -> {OUT}")

if __name__=="__main__": main()
