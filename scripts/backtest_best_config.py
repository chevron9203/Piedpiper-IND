"""
BEST config = adaptive (inverse-vol, NON-hardcoded) risk-parity across
momentum-equity (Nifty+breadth regime) / gold / US-Nasdaq.  2011-2026.
Shows the difference vs NIFTY across labelled BEAR and BULL periods, plus a
systematic up-month/down-month breakdown.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np, pandas as pd, yfinance as yf, duckdb

CAP=200_000.0; TOP_N=15; TURN_TOP=500; PRICE_MIN=30.0; RT=2*(0.0010+0.0010)+0.0005; CASH=(1.065)**(1/12)-1

def load():
    pk=pd.read_parquet("data_store/mom_v2_panels.parquet"); close=pk.xs("close",1,0); turn=pk.xs("turn",1,0)
    c=duckdb.connect("data_store/piedpiper.duckdb",read_only=True)
    n=c.execute("SELECT dt,close FROM adjusted_ohlcv WHERE symbol='Nifty 50' ORDER BY dt").df();c.close()
    n["dt"]=pd.to_datetime(n["dt"]); nifty=n.set_index("dt")["close"].resample("ME").last()
    def mpx(t): d=yf.download(t,start="2010-06-01",auto_adjust=True,progress=False); s=d["Close"].squeeze().dropna(); s.index=pd.to_datetime(s.index); return s.resample("ME").last()
    return close,turn,nifty,mpx("GOLDBEES.NS"),mpx("MON100.NS")

def components(close,turn,nifty,gold,us):
    idx=close.index; nser=nifty.reindex(idx).ffill(); nema=nser.ewm(span=10,adjust=False).mean()
    ret=close.pct_change(); vol=ret.rolling(6).std()
    m3=close.shift(1)/close.shift(4)-1; m6=close.shift(1)/close.shift(7)-1; m12=close.shift(1)/close.shift(13)-1
    ma10=close.rolling(10).mean(); above=(close>ma10)
    gret=gold.reindex(idx).ffill().pct_change(); uret=us.reindex(idx).ffill().pct_change()
    prev=set(); rows=[]; start=int(np.searchsorted(idx.year.values,2011))
    for p in range(start,len(idx)-1):
        pr=close.iloc[p]; tn=turn.iloc[p]; base=pr.index[(pr>=PRICE_MIN)&tn.notna()]
        liq=tn[base].nlargest(min(TURN_TOP,len(base))).index; breadth=above.iloc[p][liq].mean() if len(liq) else .5
        risk_on=(nser.iloc[p]>nema.iloc[p]) and (nser.iloc[p]/nser.iloc[p-6]-1>-0.02) and breadth>0.45
        mom=CASH
        if risk_on:
            elig=[s for s in liq if pr[s]>ma10.iloc[p].get(s,1e9)]; sc=pd.Series(0.0,index=elig); nc=0
            for sg in (m3,m6,m12):
                x=(sg.iloc[p][elig]/vol.iloc[p][elig].replace(0,np.nan)).dropna()
                if len(x): sc=sc.add(x.rank(pct=True),fill_value=0); nc+=1
            if nc:
                rk=(sc/nc).sort_values(ascending=False); rank={s:i for i,s in enumerate(rk.index)}
                picks=[s for s in prev if rank.get(s,1e9)<TOP_N*1.67]
                for s in rk.index:
                    if len(picks)>=TOP_N: break
                    if s not in picks: picks.append(s)
                picks=set(picks[:TOP_N])
                rr=[close.iloc[p+1][s]/close.iloc[p][s]-1 for s in picks if pd.notna(close.iloc[p+1].get(s)) and pd.notna(close.iloc[p][s])]
                to=len(picks.symmetric_difference(prev))/max(len(picks|prev),1); mom=(float(np.mean(rr)) if rr else CASH)-to*RT; prev=picks
        else: prev=set()
        rows.append((idx[p+1], mom, gret.iloc[p+1] if pd.notna(gret.iloc[p+1]) else CASH, uret.iloc[p+1] if pd.notna(uret.iloc[p+1]) else CASH))
    return pd.DataFrame(rows,columns=["dt","mom","gold","us"]).set_index("dt")

def adaptive_riskparity(D):
    vm=D["mom"].rolling(6).std().shift(1); vg=D["gold"].rolling(6).std().shift(1); vu=D["us"].rolling(6).std().shift(1)
    tot=1/vm+1/vg+1/vu
    return (((1/vm)/tot)*D["mom"]+((1/vg)/tot)*D["gold"]+((1/vu)/tot)*D["us"]).fillna(D["mom"])

def cagr_dd_sh(r):
    eq=CAP*(1+r).cumprod(); yrs=(eq.index[-1]-eq.index[0]).days/365.25
    return (eq.iloc[-1]/CAP)**(1/yrs)-1, ((eq-eq.cummax())/eq.cummax()).min(), r.mean()/r.std()*np.sqrt(12)

def main():
    close,turn,nifty,gold,us=load()
    D=components(close,turn,nifty,gold,us)
    sysr=adaptive_riskparity(D)
    syseq=CAP*(1+sysr).cumprod()
    nret=nifty.reindex(syseq.index).pct_change().fillna(0); nifeq=CAP*(1+nret).cumprod()
    def win(eq,a,b): x=eq[(eq.index>=a)&(eq.index<=b)]; return (x.iloc[-1]/x.iloc[0]-1) if len(x)>1 else float('nan')
    print("="*82)
    print("  BEST CONFIG (adaptive risk-parity) vs NIFTY — bear & bull periods (2011-2026)")
    print("="*82)
    cg,dd,sh=cagr_dd_sh(sysr); print(f"  SYSTEM full: CAGR {cg*100:+.1f}% | MaxDD {dd*100:.0f}% | Sharpe {sh:.2f}")
    ncg,ndd,nsh=cagr_dd_sh(nret); print(f"  NIFTY  full: CAGR {ncg*100:+.1f}% | MaxDD {ndd*100:.0f}% | Sharpe {nsh:.2f}\n")
    print(f"  {'PERIOD':<26}{'type':>6}{'SYSTEM':>10}{'NIFTY':>9}{'edge':>8}")
    print("  "+"-"*72)
    events=[("2011 bear","BEAR","2011-01-31","2011-12-31"),
            ("2013-14 Modi bull","BULL","2013-08-31","2014-12-31"),
            ("2015-16 bear","BEAR","2015-03-31","2016-02-29"),
            ("2017 bull","BULL","2017-01-31","2017-12-31"),
            ("2018-19 midcap bear","BEAR","2018-01-31","2019-12-31"),
            ("2020 COVID crash","BEAR","2020-01-31","2020-03-31"),
            ("2020-21 recovery","BULL","2020-04-30","2021-12-31"),
            ("2022 correction","BEAR","2022-01-31","2022-06-30"),
            ("2023-24 bull","BULL","2023-03-31","2024-09-30")]
    for nm,ty,a,b in events:
        s=win(syseq,a,b); n=win(nifeq,a,b); print(f"  {nm:<26}{ty:>6}{s*100:>+9.0f}%{n*100:>+8.0f}%{(s-n)*100:>+7.0f}%")
    # systematic up/down month split
    common=sysr.index.intersection(nret.index)
    sm=sysr.loc[common]; nm2=nret.loc[common]
    down=nm2<0; up=nm2>=0
    print("\n  Systematic split (all months):")
    print(f"    Nifty DOWN months ({down.sum()}): Nifty avg {nm2[down].mean()*100:+.1f}%/mo  |  SYSTEM avg {sm[down].mean()*100:+.1f}%/mo")
    print(f"    Nifty UP   months ({up.sum()}): Nifty avg {nm2[up].mean()*100:+.1f}%/mo  |  SYSTEM avg {sm[up].mean()*100:+.1f}%/mo")
    print(f"    -> in down months system captures {sm[down].mean()/nm2[down].mean()*100:.0f}% of Nifty's fall; in up months {sm[up].mean()/nm2[up].mean()*100:.0f}% of its rise")

if __name__=="__main__": main()
