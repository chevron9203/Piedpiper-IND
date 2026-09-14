"""
STRATEGY A — momentum by SIZE TIER (where does momentum alpha really live?)
Tests the SAME validated momentum engine (multi-TF risk-adj + trend + buffer +
regime) on different point-in-time turnover tiers of the NSE universe.
Answers: does concentrating in smaller/less-liquid names raise real OOS return?

Honest: point-in-time universe, 0.10%/side slippage + costs, IS 2010-18 / OOS 2019-26.
NOTE: smaller tiers = higher slippage in reality, so we ALSO test each at 0.25%/side.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np, pandas as pd, duckdb

CACHE="data_store/mom_v2_panels.parquet"; DB="data_store/piedpiper.duckdb"
CAP=200_000.0; PRICE_MIN=30.0; FROM_YEAR=2010; IS_END=2018

def load():
    pk=pd.read_parquet(CACHE); close=pk.xs("close",1,0); turn=pk.xs("turn",1,0)
    c=duckdb.connect(DB,read_only=True)
    n=c.execute("SELECT dt,close FROM adjusted_ohlcv WHERE symbol='Nifty 50' ORDER BY dt").df();c.close()
    n["dt"]=pd.to_datetime(n["dt"]); nifty=n.set_index("dt")["close"].resample("ME").last()
    return close,turn,nifty

def simulate(close,turn,nifty, tier, top_n, rt_cost, min_liq_rank=0.0):
    """tier=(lo,hi) turnover-percentile band among liquid names. top_n held, equal wt."""
    idx=close.index; nser=nifty.reindex(idx).ffill(); nema=nser.ewm(span=10,adjust=False).mean()
    ret=close.pct_change(); vol=ret.rolling(6).std()
    m3=close.shift(1)/close.shift(4)-1; m6=close.shift(1)/close.shift(7)-1; m12=close.shift(1)/close.shift(13)-1
    ma10=close.rolling(10).mean(); above=(close>ma10)
    prev=set(); rets=[]; cash_mo=(1.065)**(1/12)-1
    start=max(14,int(np.searchsorted(idx.year.values,FROM_YEAR)))
    for p in range(start,len(idx)-1):
        pr=close.iloc[p]; tn=turn.iloc[p]; hist=close.iloc[:p+1].notna().sum()
        base=pr.index[(pr>=PRICE_MIN)&(hist>=14)&tn.notna()&(tn>0)]
        if len(base)<50: rets.append((idx[p+1],cash_mo)); continue
        # size tier by turnover percentile within the liquid base
        rank=tn[base].rank(pct=True)
        tierset=rank.index[(rank>tier[0])&(rank<=tier[1])]
        # regime (Nifty + breadth on the FULL liquid base)
        breadth=above.iloc[p][base].mean()
        risk_on=(nser.iloc[p]>nema.iloc[p]) and (nser.iloc[p]/nser.iloc[p-6]-1>-0.02) and breadth>0.45
        if not risk_on: rets.append((idx[p+1],cash_mo)); prev=set(); continue
        elig=[s for s in tierset if pr[s]>ma10.iloc[p].get(s,1e9)]
        score=pd.Series(0.0,index=elig); nc=0
        for sg in (m3,m6,m12):
            x=(sg.iloc[p][elig]/vol.iloc[p][elig].replace(0,np.nan)).dropna()
            if len(x): score=score.add(x.rank(pct=True),fill_value=0); nc+=1
        if nc==0 or len(elig)<top_n: rets.append((idx[p+1],cash_mo)); prev=set(); continue
        ranked=(score/nc).sort_values(ascending=False); rk={s:i for i,s in enumerate(ranked.index)}
        picks=[s for s in prev if rk.get(s,10**9)<top_n*1.67]
        for s in ranked.index:
            if len(picks)>=top_n: break
            if s not in picks: picks.append(s)
        picks=set(picks[:top_n])
        r=[close.iloc[p+1][s]/close.iloc[p][s]-1 for s in picks
           if pd.notna(close.iloc[p+1].get(s)) and pd.notna(close.iloc[p][s])]
        gross=float(np.mean(r)) if r else cash_mo
        to=len(picks.symmetric_difference(prev))/max(len(picks|prev),1)
        rets.append((idx[p+1],gross-to*rt_cost)); prev=picks
    return pd.Series(dict(rets)).sort_index()

def stats(r):
    eq=CAP*(1+r).cumprod(); ie=eq[eq.index.year<=IS_END]; oe=eq[eq.index.year>IS_END]
    def cd(e,s):
        yrs=max((e.index[-1]-e.index[0]).days/365.25,.1)
        return (e.iloc[-1]/s)**(1/yrs)-1, ((e-e.cummax())/e.cummax()).min(), (e.pct_change().dropna().mean()/e.pct_change().dropna().std()*np.sqrt(12) if e.pct_change().dropna().std()>0 else float('nan'))
    ic,idd,ish=cd(ie,CAP); oc,odd,osh=cd(oe,ie.iloc[-1]); fc,fdd,fsh=cd(eq,CAP)
    return ic,oc,odd,osh,fc

def main():
    close,turn,nifty=load()
    RT_LOW=2*(0.0010+0.0010)+0.0005   # 0.10%/side
    RT_HI =2*(0.0025+0.0010)+0.0005   # 0.25%/side (realistic for smaller caps)
    print("="*100)
    print("  MOMENTUM BY SIZE TIER  (top-15, IS 2010-18 / OOS 2019-26)  — where does the alpha live?")
    print("="*100)
    print(f"  {'tier (turnover pct)':<26}{'slip':<7}{'IS_CAGR':>9}{'OOS_CAGR':>10}{'OOS_DD':>8}{'OOS_Sh':>8}{'Full':>8}")
    tiers=[("LARGE  top10% (.90-1.0)",(0.90,1.00)),
           ("MID    (.60-.90)",(0.60,0.90)),
           ("SMALL  (.30-.60)",(0.30,0.60)),
           ("MICRO  (.10-.30)",(0.10,0.30)),
           ("ALL liquid (.0-1.0) [base]",(0.00,1.00))]
    for label,tier in tiers:
        for slabel,rt in [("0.10%",RT_LOW),("0.25%",RT_HI)]:
            ic,oc,odd,osh,fc=stats(simulate(close,turn,nifty,tier,15,rt))
            print(f"  {label:<26}{slabel:<7}{ic*100:>+8.1f}%{oc*100:>+9.1f}%{odd*100:>7.0f}%{osh:>8.2f}{fc*100:>+7.1f}%")
        print()

if __name__=="__main__": main()
