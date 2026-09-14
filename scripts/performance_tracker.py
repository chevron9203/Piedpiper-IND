"""
LIVE performance tracker — records the paper book vs Nifty 50 / Nifty 500 DAILY,
starting from inception (first run = today). Builds the real forward track record.
Each run appends one row (idempotent per date) to data_store/perf_history.csv:
  date, system_nav, nifty50_nav, nifty500_nav   (all rebased to 200000 at inception)
System daily return = sum(weight_i * asset_i daily return) from the live signal.
Run daily (cron). PAPER — no orders.
"""
from __future__ import annotations
import sys, json, os
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd, yfinance as yf, duckdb

BASE=Path(__file__).parent.parent
SIGNAL=BASE/"data_store/live_signal.json"; HIST=BASE/"data_store/perf_history.csv"
SIGNAL_S4=BASE/"data_store/live_signal_s4.json"; SIGNAL_S5=BASE/"data_store/live_signal_s5.json"
POS=BASE/"data_store/paper_positions.json"   # paper entry prices stamped at EXECUTION (inception / each rebalance)
CAP=200_000.0
# tradeable proxies for each sleeve (what you'd actually hold)
TICK={"gold":"GOLDBEES.NS","us_nasdaq":"MON100.NS","momentum":"^CNX500","nifty50":"^NSEI","nifty500":"^CRSLDX"}

def last_two(tk):
    try:
        d=yf.download(tk,period="7d",auto_adjust=True,progress=False)["Close"].squeeze().dropna()
        return (float(d.iloc[-2]), float(d.iloc[-1])) if len(d)>=2 else (float(d.iloc[-1]),float(d.iloc[-1]))
    except Exception: return None

def daily_ret(tk):
    lt=last_two(tk);
    return (lt[1]/lt[0]-1) if lt else 0.0

def nifty_daily(sym):
    """Daily return for an index from the eod2 CSV (most reliable Indian EOD source,
    fresh once the 19:00 data job runs). Falls back to yfinance."""
    fn={"Nifty 50":"nifty 50.csv","Nifty 500":"nifty 500.csv"}.get(sym)
    if fn:
        f=BASE/"data_store/eod2/src/eod2_data/daily"/fn
        if f.exists():
            try:
                d=pd.read_csv(f, usecols=["Close"])["Close"].dropna()
                if len(d)>=2: return float(d.iloc[-1]/d.iloc[-2]-1)
            except Exception: pass
    return daily_ret({"Nifty 50":"^NSEI","Nifty 500":"^CRSLDX"}.get(sym,""))

def live_price(tk):
    lt=last_two(tk); return lt[1] if lt else None

def eod2_close(sym):
    """Latest EOD close for a stock symbol from eod2 CSV."""
    f=BASE/"data_store/eod2/src/eod2_data/daily"/f"{sym.lower()}.csv"
    if not f.exists(): return None
    try:
        d=pd.read_csv(f, usecols=["Close","Series"]); d=d[d["Series"]=="EQ"]
        return round(float(d["Close"].iloc[-1]),2) if len(d) else None
    except Exception: return None

def eod2_two(sym):
    """(prev_close, last_close) from eod2 CSV — for real daily returns."""
    f=BASE/"data_store/eod2/src/eod2_data/daily"/f"{sym.lower()}.csv"
    if not f.exists(): return None
    try:
        d=pd.read_csv(f, usecols=["Close","Series"]); c=d[d["Series"]=="EQ"]["Close"].dropna()
        return (float(c.iloc[-2]), float(c.iloc[-1])) if len(c)>=2 else None
    except Exception: return None

def momentum_sleeve_daily(picks):
    """REAL daily return of the held momentum stocks (equal-weight), from eod2 closes.
    Falls back to Nifty-500 proxy only if no stock data available."""
    rs=[]
    for s in picks:
        t=eod2_two(s)
        if t and t[0]>0: rs.append(t[1]/t[0]-1)
    return (sum(rs)/len(rs)) if rs else None

def stamp_paper_entries(sig):
    """Stamp paper ENTRY prices at execution time (=now). Re-stamp on a new
    rebalance (signal as_of changed). Makes day-1 P&L start at 0 — you entered
    TODAY, not at the month-end signal date. Covers gold, US, AND the 15 stocks."""
    alloc=sig.get("allocation",{}); as_of=sig.get("as_of"); picks=sig.get("momentum_picks",[])
    cur=json.load(open(POS)) if os.path.exists(POS) else {}
    if cur.get("as_of")==as_of and cur.get("entries"):
        return cur                          # already stamped for this rebalance
    src={"gold":"goldbees","us_nasdaq":"mon100"}   # stamp from SAME eod2 source used to price now
    entries={}
    for k in alloc:
        if k in src:
            p=eod2_close(src[k]);  entries[k]=p if p else None
        elif k=="momentum":                 # stamp each held stock at its EOD close
            for s in picks:
                p=eod2_close(s)
                if p: entries[s]=p
    entries={k:v for k,v in entries.items() if v}
    out={"as_of":as_of, "executed":datetime.now().strftime("%Y-%m-%d %H:%M"),
         "inception":cur.get("inception", datetime.now().strftime("%Y-%m-%d")),
         "momentum_picks":picks, "entries":entries}
    json.dump(out, open(POS,"w"), indent=2)
    return out

def is_trading_day(dt):
    """Skip weekends and NSE holidays — never record a row on a non-trading day."""
    if dt.weekday()>=5: return False          # Sat/Sun
    try:
        hol=set(json.load(open(BASE/"data_store/nse_holidays.json")))
        if dt.strftime("%Y-%m-%d") in hol: return False
    except Exception: pass
    return True

def eod2_last_date():
    """Latest trading date present in the eod2 data (the day this NAV move represents)."""
    dates=[]
    for f in ["goldbees.csv","mon100.csv","nifty 50.csv"]:
        p=BASE/"data_store/eod2/src/eod2_data/daily"/f
        if p.exists():
            try:
                d=pd.read_csv(p, usecols=["Date"])["Date"].dropna()
                if len(d): dates.append(str(d.iloc[-1]))
            except Exception: pass
    return max(dates) if dates else None

def main():
    sig=json.load(open(SIGNAL)) if os.path.exists(SIGNAL) else {}
    sig4=json.load(open(SIGNAL_S4)) if SIGNAL_S4.exists() else {}
    sig5=json.load(open(SIGNAL_S5)) if SIGNAL_S5.exists() else {}
    alloc=sig.get("allocation",{}); cash=sig.get("cash",0.0)
    picks4=sig4.get("momentum_picks",[]); picks5=sig5.get("momentum_picks",[])
    now=datetime.now()

    # ---- data date = the trading day this move represents (NOT wall-clock "today") ----
    data_date=eod2_last_date()
    if not data_date:
        print(f"{now:%Y-%m-%d %H:%M} | no eod2 data available — skipping"); return
    _cols=["date","s1_nav","s2_nav","s3_nav","s4_nav","s5_nav","nifty50_nav","nifty500_nav"]
    _h=pd.read_csv(HIST) if os.path.exists(HIST) else pd.DataFrame(columns=_cols)
    if "s1_nav" not in _h.columns: _h=pd.DataFrame(columns=_cols)
    # migration: seed new columns at CAP for existing rows (S4/S5 start tracking from now)
    for c in ["s4_nav","s5_nav"]:
        if c not in _h.columns: _h[c]=CAP
    hist_exists=len(_h)>0
    today=data_date                          # row is keyed on the DATA date

    # ---- GUARD: advance at most ONCE per new trading day of data ----
    # Fixes (A) idempotency: re-run same day -> data_date already recorded -> skip
    #       (B) staleness:   NSE data late -> data_date not newer -> skip (don't record stale as new)
    if hist_exists:
        recorded=set(_h["date"].astype(str))
        if data_date in recorded or data_date <= str(_h["date"].astype(str).max()):
            print(f"{now:%Y-%m-%d %H:%M} | data date {data_date} already processed / not newer — skipping")
            return
    stamp_paper_entries(sig)                 # ensure paper entries reflect actual execution price

    # ---- EOD: how did each asset do today? (fresh closes) ----
    cash_day=(1.06)**(1/252)-1
    sys_ret=cash*cash_day; breakdown={}
    if cash>0: breakdown["cash"]=round(cash_day*100,3)
    picks=sig.get("momentum_picks",[])
    # gold/US: prefer reliable eod2 NSE closes; yfinance only as fallback (avoids
    # silently recording 0% if yfinance API is down on a given evening)
    EOD2_SRC={"gold":"goldbees","us_nasdaq":"mon100"}
    def asset_daily(k):
        if k in EOD2_SRC:
            t=eod2_two(EOD2_SRC[k])
            if t and t[0]>0: return t[1]/t[0]-1
        return daily_ret(TICK[k]) if k in TICK else 0.0
    for k,w in alloc.items():
        if k=="momentum":
            r=momentum_sleeve_daily(picks)       # REAL held-stock avg daily return
            if r is None: r=nifty_daily("Nifty 500") or 0.0   # fallback only if no stock data
        elif k in TICK:
            r=asset_daily(k)
        else:
            r=0.0
        sys_ret+=w*r; breakdown[k]=round(r*100,2)
    n50=nifty_daily("Nifty 50") or 0.0
    n500=nifty_daily("Nifty 500") or 0.0
    # inception = no prior history (guard above already ensured data_date is NEW)
    is_inception = not hist_exists
    if is_inception:
        sys_ret=n50=n500=0.0; breakdown={k:0.0 for k in breakdown}

    # ---- THREE SYSTEMS with REALISTIC sleeve tracking + MONTHLY rebalance ----
    # S1: momentum book (deployed).  S2 (barbell): 70% safe + 30% [momentum @2x].
    # S3 (aggressive): 2x the book.  S2/S3 DRIFT within a month, REBALANCE monthly
    # (matching S1's monthly cadence) — not silently rebalanced daily.
    s1=sys_ret
    safe_day=(1.065)**(1/252)-1; borrow_day=(1.10)**(1/252)-1
    REB_COST=0.0015                              # cost on the traded (drift) amount at monthly rebalance
    STATE=BASE/"data_store/leverage_state.json"
    cur_month=now.strftime("%Y-%m")
    st=json.load(open(STATE)) if os.path.exists(STATE) else {}
    if is_inception:
        # seed sleeves at target weights; day-1 return is 0
        st={"s2_safe":0.70*CAP,"s2_risky":0.30*CAP,"s3_gross":2.0*CAP,"s3_debt":1.0*CAP,"reb_month":cur_month}
        s2=s3=0.0
    else:
        if not st:
            # RECOVERY: state file missing but history exists — reconstruct from last NAV
            # (assumes it was at target weights; only triggers if state was lost)
            pv=_h.sort_values("date").iloc[-1]
            st={"s2_safe":0.70*pv["s2_nav"],"s2_risky":0.30*pv["s2_nav"],
                "s3_gross":2.0*pv["s3_nav"],"s3_debt":1.0*pv["s3_nav"],"reb_month":cur_month}
        s2_nav0=st["s2_safe"]+st["s2_risky"]; s3_nav0=st["s3_gross"]-st["s3_debt"]
        # MONTHLY REBALANCE (first tracked day of a new month) — restore target weights/leverage
        if st.get("reb_month")!=cur_month:
            traded2=abs(0.70*s2_nav0-st["s2_safe"])+abs(0.30*s2_nav0-st["s2_risky"])  # amount reshuffled
            st["s2_safe"]=0.70*s2_nav0 - traded2*REB_COST/2; st["s2_risky"]=0.30*s2_nav0 - traded2*REB_COST/2
            traded3=abs(2.0*s3_nav0-st["s3_gross"])
            st["s3_gross"]=2.0*s3_nav0 - traded3*REB_COST; st["s3_debt"]=1.0*(st["s3_gross"]/2.0)  # ~restore 2x
            st["reb_month"]=cur_month
            s2_nav0=st["s2_safe"]+st["s2_risky"]; s3_nav0=st["s3_gross"]-st["s3_debt"]
        # DAILY DRIFT — sleeves move independently (barbell) / position & debt (leverage)
        st["s2_safe"]*=(1+safe_day); st["s2_risky"]*=(1 + 2.0*s1 - borrow_day)
        st["s3_gross"]*=(1+s1);      st["s3_debt"] *=(1+borrow_day)
        s2=(st["s2_safe"]+st["s2_risky"])/s2_nav0 - 1     # realized daily return (for display)
        s3=(st["s3_gross"]-st["s3_debt"])/s3_nav0 - 1
    json.dump(st, open(STATE,"w"), indent=2)

    # ---- S4 / S5: pure MID-cap momentum systems (no leverage, no gold/US) ----
    def _pure_sys_ret(sig_j, picks_j):
        """Daily return for a pure-momentum signal: invested or cash."""
        if sig_j.get("allocation",{}).get("momentum",0)>0 and picks_j:
            r=momentum_sleeve_daily(picks_j)
            return r if r is not None else (nifty_daily("Nifty 500") or 0.0)
        return cash_day
    if is_inception:
        s4_ret=s5_ret=0.0
    else:
        s4_ret=_pure_sys_ret(sig4,picks4); s5_ret=_pure_sys_ret(sig5,picks5)

    # write EOD snapshot (per-asset + per-system day performance) for the dashboard
    json.dump({"date":today,"asof":datetime.now().strftime("%Y-%m-%d %H:%M"),
               "s1_day_pct":round(s1*100,2),"s2_day_pct":round(s2*100,2),"s3_day_pct":round(s3*100,2),
               "s4_day_pct":round(s4_ret*100,2),"s5_day_pct":round(s5_ret*100,2),
               "nifty50_day_pct":round(n50*100,2),"nifty500_day_pct":round(n500*100,2),"assets":breakdown},
              open(BASE/"data_store/eod_snapshot.json","w"), indent=2)

    # ---- append to history (rebased to CAP at inception); 5 systems + benchmarks ----
    cols=["date","s1_nav","s2_nav","s3_nav","s4_nav","s5_nav","nifty50_nav","nifty500_nav"]
    h=pd.read_csv(HIST) if os.path.exists(HIST) else pd.DataFrame(columns=cols)
    if "s1_nav" not in h.columns: h=pd.DataFrame(columns=cols)   # migrate old single-system file
    for c in ["s4_nav","s5_nav"]:
        if c not in h.columns: h[c]=CAP                          # seed new systems at CAP
    h=h[h["date"]!=today]                      # idempotent: drop any existing row for today
    if len(h)==0:
        row={"date":today,"s1_nav":CAP,"s2_nav":CAP,"s3_nav":CAP,
             "s4_nav":CAP,"s5_nav":CAP,"nifty50_nav":CAP,"nifty500_nav":CAP}
    else:
        p=h.iloc[-1]
        row={"date":today,
             "s1_nav":round(p["s1_nav"]*(1+s1)),
             "s2_nav":round(st["s2_safe"]+st["s2_risky"]),        # from sleeve state (drift+rebalance)
             "s3_nav":round(st["s3_gross"]-st["s3_debt"]),        # from leveraged position state
             "s4_nav":round(p["s4_nav"]*(1+s4_ret)),
             "s5_nav":round(p["s5_nav"]*(1+s5_ret)),
             "nifty50_nav":round(p["nifty50_nav"]*(1+n50)),"nifty500_nav":round(p["nifty500_nav"]*(1+n500))}
    h=pd.concat([h,pd.DataFrame([row])],ignore_index=True).sort_values("date").reset_index(drop=True)
    h.to_csv(HIST,index=False)
    r=h.iloc[-1]
    print(f"{datetime.now():%Y-%m-%d %H:%M} | tracked {today}: S1 {r['s1_nav']:.0f} S2 {r['s2_nav']:.0f} "
          f"S3 {r['s3_nav']:.0f} S4 {r['s4_nav']:.0f} S5 {r['s5_nav']:.0f} | "
          f"N50 {r['nifty50_nav']:.0f} N500 {r['nifty500_nav']:.0f} ({len(h)} days)")

if __name__=="__main__": main()
