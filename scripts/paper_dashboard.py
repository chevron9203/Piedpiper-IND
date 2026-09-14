"""
Paper-trading dashboard for the regime-conditional momentum system.
Localhost-only (reach via SSH tunnel). Serves:
  1. Invested / Day P&L / Overall P&L
  2. Daily activities (recent monitor + signal log lines)
  3. Comparison: System vs Nifty 50 vs Nifty 500 (equity chart + stats)
  4. Portfolio (current target allocation + holdings)
Data sources: data_store/live_signal.json, monitor_status.json, dashboard_data.json,
              logs/*.log. All read-only, PAPER.
Run:  python scripts/paper_dashboard.py   (http://127.0.0.1:5002)
"""
from __future__ import annotations
import sys, json, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from datetime import datetime
from flask import Flask, jsonify, render_template, request
import yfinance as yf, pandas as pd

BASE=Path(__file__).parent.parent
SIGNAL=BASE/"data_store/live_signal.json"; MON=BASE/"data_store/monitor_status.json"
SIGNAL_S4=BASE/"data_store/live_signal_s4.json"; SIGNAL_S5=BASE/"data_store/live_signal_s5.json"
HIST=BASE/"data_store/perf_history.csv"; POS=BASE/"data_store/paper_positions.json"
EOD=BASE/"data_store/eod_snapshot.json"; LOGDIR=BASE/"logs"
CAP=200_000.0
app=Flask(__name__, template_folder=str(BASE/"reporting/dashboard/templates"))

def _jload(p, default=None):
    try: return json.load(open(p))
    except Exception: return default if default is not None else {}

def _tail(path, n=15):
    try: return [l.rstrip() for l in open(path).read().splitlines()][-n:]
    except Exception: return []

DAILY=BASE/"data_store/eod2/src/eod2_data/daily"
def _eod_two(sym):
    """(prev_close, last_close, last_date) from the official NSE eod2 CSV — the
    reliable EOD source (yfinance .NS lags). Used for both entry & current pricing."""
    f=DAILY/f"{sym.lower()}.csv"
    if not f.exists(): return None,None,None
    try:
        d=pd.read_csv(f, usecols=["Date","Close","Series"]); d=d[d["Series"]=="EQ"]
        c=d["Close"].dropna()
        if len(c)>=2: return float(c.iloc[-2]), float(c.iloc[-1]), str(d["Date"].iloc[-1])
        if len(c)==1: return float(c.iloc[-1]), float(c.iloc[-1]), str(d["Date"].iloc[-1])
    except Exception: pass
    return None,None,None

def _is_trading_day():
    now=datetime.now()
    if now.weekday()>=5: return False, "weekend"
    try:
        hol=set(json.load(open(BASE/"data_store/nse_holidays.json")))
        if now.strftime("%Y-%m-%d") in hol: return False, "holiday"
    except Exception: pass
    return (9<=now.hour<16), ("open" if 9<=now.hour<16 else "closed")

def _perf(period="all"):
    """Live forward track of the 5 systems + benchmarks (from inception) with period filter."""
    if not HIST.exists(): return {}
    h=pd.read_csv(HIST); h["date"]=pd.to_datetime(h["date"])
    if "s1_nav" not in h.columns: return {}
    # fill new systems at CAP if they weren't in earlier rows
    for c in ["s4_nav","s5_nav"]:
        if c not in h.columns: h[c]=CAP
    if period!="all" and len(h):
        days={"1w":7,"1m":30,"3m":90,"6m":182,"12m":365}.get(period)
        if days: h=h[h["date"]>=h["date"].max()-pd.Timedelta(days=days)]
    if len(h)==0: return {}
    def col(c): return [round(float(v)) for v in h[c]]
    def stat(c):
        base=h[c].iloc[0]; cur=h[c].iloc[-1]; dd=((h[c]-h[c].cummax())/h[c].cummax()).min()
        return dict(ret=round((cur/base-1)*100,2), value=round(float(cur)), maxdd=round(dd*100,1))
    keys=["s1_nav","s2_nav","s3_nav","s4_nav","s5_nav","nifty50_nav","nifty500_nav"]
    return dict(dates=[d.strftime("%Y-%m-%d") for d in h["date"]],
               series={k:col(k) for k in keys}, stats={k:stat(k) for k in keys},
               days=len(h), inception=h["date"].iloc[0].strftime("%Y-%m-%d"))

@app.route("/api/performance")
def performance():
    return jsonify(_perf(request.args.get("period","all")))

@app.route("/api/status")
def status():
    sig=_jload(SIGNAL, {}); mon=_jload(MON, {}); pos=_jload(POS, {})
    sig4=_jload(SIGNAL_S4, {}); sig5=_jload(SIGNAL_S5, {})
    alloc=sig.get("allocation",{}); cash=sig.get("cash",0)
    # entry = PAPER execution price (stamped at inception/rebalance), NOT the month-end signal price
    entry=pos.get("entries", {}) or sig.get("entry_prices",{})
    picks=sig.get("momentum_picks",[])
    picks4=sig4.get("momentum_picks",[]); picks5=sig5.get("momentum_picks",[])

    # portfolio rows with live prices + day/overall P&L
    cash_day=(1.065**(1/252)-1)              # arbitrage/cash daily accrual
    inception=pos.get("inception")           # no full trading day has elapsed until price date > inception
    src={"gold":"goldbees","us_nasdaq":"mon100"}   # eod2 CSV names (official NSE closes)
    rows=[]; overall=0.0; day=0.0; last_date=None
    for k,wt in alloc.items():
        row={"name":k,"label":{"gold":"Gold (GOLDBEES)","us_nasdaq":"US Nasdaq-100 (MON100)","momentum":"India Momentum (15 stocks)"}.get(k,k),
             "weight":round(wt*100,1)}
        if k in src:
            prev,now,ld=_eod_two(src[k]); e=entry.get(k); last_date=ld or last_date
            if now and e:
                pl=(now/e-1); overall+=wt*pl; row.update(entry=round(e,2), now=round(now,2), pl_pct=round(pl*100,2))
            # day move only counts once a trading day has CLOSED after entry
            if now and prev and inception and last_date and last_date>inception:
                dch=now/prev-1; day+=wt*dch; row["day_pct"]=round(dch*100,2)
            else:
                row["day_pct"]=0.0
        rows.append(row)
    if inception and last_date and last_date>inception: day+=cash*cash_day
    if cash>0: rows.append({"name":"cash","label":"Arbitrage fund (~6-7% · equity-taxed)",
                            "weight":round(cash*100,1),"note":"~0.5%/mo","day_pct":round(cash_day*100,3)})

    invested_pct=round((1-cash)*100,1)
    trading, mstatus = _is_trading_day()

    # ---- per-system POSITIONS ----
    LBL={"gold":"Gold (GOLDBEES)","us_nasdaq":"US Nasdaq (MON100)","momentum":"Momentum stocks",
         "cash":"Cash","arbitrage":"Arbitrage fund (safe)"}
    def fmt(d): return [{"label":LBL.get(k,k),"weight":round(v*100,1)} for k,v in d.items() if v>0.001]
    s1a=dict(alloc)
    if cash>0: s1a["cash"]=cash
    s2a={"arbitrage":0.70+0.60*cash}
    for k,v in alloc.items(): s2a[k]=0.60*v          # 70% safe + 30%@2x = 60% of S1's book
    s3a={k:2*v for k,v in alloc.items()}
    if cash>0: s3a["cash"]=2*cash                     # 2x the whole book
    s4a=dict(sig4.get("allocation",{}))
    s4c=sig4.get("cash",1.0)
    if s4c>0: s4a["cash"]=s4c
    s5a={"momentum":1.0}
    POS5={"s1":{"rows":fmt(s1a),"gross":round(sum(s1a.values())*100),"borrow":0,"picks":picks},
          "s2":{"rows":fmt(s2a),"gross":round(sum(s2a.values())*100),"borrow":30,"picks":picks},
          "s3":{"rows":fmt(s3a),"gross":round(sum(s3a.values())*100),"borrow":100,"picks":picks},
          "s4":{"rows":fmt(s4a),"gross":100,"borrow":0,"picks":picks4},
          "s5":{"rows":fmt(s5a),"gross":100,"borrow":0,"picks":picks5}}

    # ---- 5-SYSTEM summary (NAV + overall/day P&L from perf_history + eod snapshot) ----
    eod=_jload(EOD, {})
    systems=[]
    if HIST.exists():
        h=pd.read_csv(HIST)
        for c in ["s4_nav","s5_nav"]:
            if c not in h.columns: h[c]=CAP
        if "s1_nav" in h.columns and len(h):
            cur=h.iloc[-1]
            meta=[("s1","System 1 · Momentum","Medium risk · balanced multi-asset","s1_day_pct"),
                  ("s2","System 2 · Survivorship","Low risk · barbell, sleeps well","s2_day_pct"),
                  ("s3","System 3 · Aggressive","High risk · 2× leverage","s3_day_pct"),
                  ("s4","System 4 · Pure (Regime)","Regime-gated · pure MID-cap momentum","s4_day_pct"),
                  ("s5","System 5 · Pure (Always)","Always invested · pure MID-cap momentum","s5_day_pct")]
            for key,name,desc,dk in meta:
                nav=float(cur[f"{key}_nav"])
                mdd=((h[f"{key}_nav"]-h[f"{key}_nav"].cummax())/h[f"{key}_nav"].cummax()).min()
                systems.append(dict(id=key,name=name,desc=desc,nav=round(nav),
                                    overall_pct=round((nav/CAP-1)*100,2),
                                    day_pct=eod.get(dk,0.0), maxdd=round(mdd*100,1),
                                    positions=POS5[key]["rows"], gross=POS5[key]["gross"],
                                    borrow=POS5[key]["borrow"],
                                    momentum_picks=POS5[key]["picks"]))
    return jsonify(dict(
        as_of=sig.get("as_of"), risk_on=sig.get("risk_on"), breadth=sig.get("breadth"),
        mode=sig.get("mode","PAPER"), systems=systems,
        invested_pct=invested_pct, day_pnl_pct=round(day*100,2), overall_pnl_pct=round(overall*100,2),
        overall_pnl_rs=round(overall*CAP), portfolio=rows, momentum_picks=picks,
        market_status=mstatus, prices_asof=last_date, regime_shift=mon.get("regime_shift_watch",False),
        updated=str(datetime.now())[:19]))

@app.route("/api/activity")
def activity():
    return jsonify(dict(monitor=_tail(LOGDIR/"momentum_monitor.log",20),
                        signal=_tail(LOGDIR/"momentum_live.log",15)))

@app.route("/")
def index():
    resp=app.make_response(render_template("paper_dashboard.html"))
    resp.headers["Cache-Control"]="no-store, no-cache, must-revalidate, max-age=0"
    return resp

if __name__=="__main__":
    app.run(host="127.0.0.1", port=5002, debug=False)
