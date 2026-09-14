"""
Intelligent regime allocator — momentum / gold / cash.
Risk-ON (Nifty uptrend)  -> momentum MAX sleeve (multi-TF + trend + buffer)
Risk-OFF (Nifty breaks)  -> GOLD if gold itself is trending up, else cash.
Compares vs current (momentum+cash) and Nifty. IS 2010-18 / OOS 2019-26.
Gold = split-adjusted GOLDBEES monthly (data_store/gold_monthly.parquet).
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np, pandas as pd, duckdb

CACHE="data_store/mom_v2_panels.parquet"; GOLD="data_store/gold_monthly.parquet"; DB="data_store/piedpiper.duckdb"
CAP=200_000.0; TOP_N=15; TURN_TOP=500; PRICE_MIN=30.0
RT=2*(0.0010+0.0010)+0.0005; CASH_MO=(1.065)**(1/12)-1; GOLD_COST=0.0015
FROM_YEAR=2010; IS_END=2018

def load():
    p=pd.read_parquet(CACHE); close=p.xs("close",axis=1,level=0); turn=p.xs("turn",axis=1,level=0)
    c=duckdb.connect(DB,read_only=True)
    n=c.execute("SELECT dt,close FROM adjusted_ohlcv WHERE symbol='Nifty 50' ORDER BY dt").df();c.close()
    n["dt"]=pd.to_datetime(n["dt"]); nifty=n.set_index("dt")["close"].resample("ME").last()
    gold=pd.read_parquet(GOLD)["gold"]
    return close,turn,nifty,gold

def run(close,turn,nifty,gold, riskoff="cash", regime_mode="nifty"):
    """riskoff in {'cash','gold','goldtrend'}; regime_mode in {'nifty','breadth','both'}"""
    idx=close.index; nser=nifty.reindex(idx).ffill(); nema=nser.ewm(span=10,adjust=False).mean()
    g=gold.reindex(idx).ffill(); gma=g.ewm(span=10,adjust=False).mean(); gret=g.pct_change()
    ret=close.pct_change(); vol=ret.rolling(6).std()
    m3=close.shift(1)/close.shift(4)-1; m6=close.shift(1)/close.shift(7)-1; m12=close.shift(1)/close.shift(13)-1
    ma10=close.rolling(10).mean()
    above=(close>ma10)                       # breadth: is each stock above its own 10mo trend
    prev=set(); rets=[]; alloc={"mom":0,"gold":0,"cash":0}
    start=int(np.searchsorted(idx.year.values,FROM_YEAR))
    for p in range(start,len(idx)-1):
        nifty_ok=(nser.iloc[p]>nema.iloc[p]) and (nser.iloc[p]/nser.iloc[p-6]-1>-0.02)
        # breadth = fraction of liquid universe above its own trend
        pr=close.iloc[p]; tnb=turn.iloc[p]
        libase=pr.index[(pr>=PRICE_MIN)&tnb.notna()]
        liq_b=tnb[libase].nlargest(min(TURN_TOP,len(libase))).index
        breadth=above.iloc[p][liq_b].mean() if len(liq_b) else 0.5
        breadth_ok=breadth>0.45
        in_mkt = nifty_ok if regime_mode=="nifty" else (breadth_ok if regime_mode=="breadth" else (nifty_ok and breadth_ok))
        if in_mkt:
            price=close.iloc[p]; tn=turn.iloc[p]; hist=close.iloc[:p+1].notna().sum()
            base=price.index[(price>=PRICE_MIN)&(hist>=14)&tn.notna()]
            elig=[s for s in tn[base].nlargest(min(TURN_TOP,len(base))).index
                  if pd.notna(ma10.iloc[p].get(s)) and price[s]>ma10.iloc[p][s]]
            score=pd.Series(0.0,index=elig); nc=0
            for sg in (m3,m6,m12):
                s=(sg.iloc[p][elig]/vol.iloc[p][elig].replace(0,np.nan)).dropna()
                if len(s): score=score.add(s.rank(pct=True),fill_value=0); nc+=1
            if nc==0: gross,picks=CASH_MO,set()
            else:
                ranked=(score/nc).sort_values(ascending=False); rk={s:i for i,s in enumerate(ranked.index)}
                picks=[s for s in prev if rk.get(s,10**9)<TOP_N*1.67]
                for s in ranked.index:
                    if len(picks)>=TOP_N: break
                    if s not in picks: picks.append(s)
                picks=set(picks[:TOP_N])
                r=[close.iloc[p+1][s]/close.iloc[p][s]-1 for s in picks
                   if pd.notna(close.iloc[p+1].get(s)) and pd.notna(close.iloc[p][s])]
                gross=float(np.mean(r)) if r else CASH_MO
            to=len(picks.symmetric_difference(prev))/max(len(picks|prev),1)
            rets.append((idx[p+1],gross-to*RT)); prev=picks; alloc["mom"]+=1
        else:
            # risk-off: choose defensive
            use_gold = (riskoff=="gold") or (riskoff=="goldtrend" and g.iloc[p]>gma.iloc[p])
            if use_gold and pd.notna(gret.iloc[p+1]):
                rets.append((idx[p+1], gret.iloc[p+1]-GOLD_COST)); alloc["gold"]+=1
            else:
                rets.append((idx[p+1], CASH_MO)); alloc["cash"]+=1
            prev=set()
    return pd.Series(dict(rets)).sort_index(), alloc

def metrics(eq,start):
    yrs=max((eq.index[-1]-eq.index[0]).days/365.25,0.1)
    cagr=(eq.iloc[-1]/start)**(1/yrs)-1; dd=((eq-eq.cummax())/eq.cummax()).min()
    r=eq.pct_change().dropna(); sh=(r.mean()/r.std()*np.sqrt(12)) if r.std()>0 else float("nan")
    return cagr,dd,sh

def show(rets,label,alloc=None):
    eq=CAP*(1+rets).cumprod()
    ie=eq[eq.index.year<=IS_END]; oe=eq[eq.index.year>IS_END]
    ic,idd,ish=metrics(ie,CAP); oc,odd,osh=metrics(oe,ie.iloc[-1]); fc,fdd,fsh=metrics(eq,CAP)
    extra=f" | mom/gold/cash {alloc['mom']}/{alloc['gold']}/{alloc['cash']}" if alloc else ""
    print(f"  {label:<34} IS {ic*100:>+5.1f}%/DD{idd*100:.0f}% | OOS {oc*100:>+5.1f}%/DD{odd*100:.0f}%/Sh{osh:.2f} | Full {fc*100:>+5.1f}%/DD{fdd*100:.0f}%{extra}")

def main():
    close,turn,nifty,gold=load()
    print(f"{'='*112}\n  INTELLIGENT REGIME ALLOCATOR — momentum / gold / cash  (IS 2010-18 / OOS 2019-26)\n{'='*112}")
    show(run(close,turn,nifty,gold,"cash","nifty")[0],      "CURRENT: mom+cash, Nifty regime")
    show(run(close,turn,nifty,gold,"goldtrend","nifty")[0], "A: mom+gold, Nifty regime")
    print("  "+"-"*108)
    print("  breadth-aware regime (catches midcap crashes the Nifty-50 filter misses):")
    show(run(close,turn,nifty,gold,"cash","breadth")[0],    "B: mom+cash, BREADTH regime")
    show(run(close,turn,nifty,gold,"goldtrend","breadth")[0],"C: mom+gold, BREADTH regime")
    show(run(close,turn,nifty,gold,"goldtrend","both")[0],  "D: mom+gold, Nifty AND breadth")
    ne=CAP*(nifty[nifty.index.year>=FROM_YEAR].pct_change().fillna(0)+1).cumprod()
    nc,ndd,nsh=metrics(ne[ne.index.year>IS_END],ne[ne.index.year<=IS_END].iloc[-1])
    print(f"\n  Nifty benchmark:                   OOS {nc*100:+.1f}% / DD {ndd*100:.0f}% / Sharpe {nsh:.2f}")

if __name__=="__main__": main()
