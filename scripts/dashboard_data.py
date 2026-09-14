"""
Pre-compute the dashboard's benchmark comparison (expensive -> cache to JSON):
regime-conditional SYSTEM vs NIFTY 50 vs NIFTY 500, monthly equity from 2012.
Run monthly (or on demand): python scripts/dashboard_data.py
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np, pandas as pd
from scripts.backtest_regime_conditional import load, components, regime_conditional, CAP

OUT="data_store/dashboard_data.json"
N500_CSV="data_store/eod2/src/eod2_data/daily/nifty 500.csv"

def curve_stats(eq):
    yrs=(eq.index[-1]-eq.index[0]).days/365.25
    return dict(cagr=round(((eq.iloc[-1]/eq.iloc[0])**(1/yrs)-1)*100,1),
                maxdd=round(((eq-eq.cummax())/eq.cummax()).min()*100,1),
                final=round(float(eq.iloc[-1])))

def main():
    close,turn,nifty,gold,us=load()
    D=components(close,turn,nifty,gold,us)
    sysr=regime_conditional(D,0.70); syseq=CAP*(1+sysr).cumprod()
    # align all to 2012+ (Nifty 500 history) and re-base to CAP
    syseq=syseq[syseq.index.year>=2012]; syseq=syseq/syseq.iloc[0]*CAP
    idx=syseq.index
    n50=CAP*(1+nifty.reindex(idx).pct_change().fillna(0)).cumprod()
    # nifty 500 from eod2
    d=pd.read_csv(N500_CSV, usecols=["Date","Close"]); d["Date"]=pd.to_datetime(d["Date"])
    n500m=d.set_index("Date")["Close"].resample("ME").last().reindex(idx).ffill()
    n500=CAP*(1+n500m.pct_change().fillna(0)).cumprod()
    data=dict(
        dates=[d.strftime("%Y-%m") for d in idx],
        system=[round(float(v)) for v in syseq],
        nifty50=[round(float(v)) for v in n50],
        nifty500=[round(float(v)) for v in n500],
        stats=dict(system=curve_stats(syseq), nifty50=curve_stats(n50), nifty500=curve_stats(n500)),
        generated=str(pd.Timestamp.now()))
    json.dump(data, open(OUT,"w"))
    print(f"saved {OUT}: system {data['stats']['system']} | nifty50 {data['stats']['nifty50']} | nifty500 {data['stats']['nifty500']}")

if __name__=="__main__": main()
