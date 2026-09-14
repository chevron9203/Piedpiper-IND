"""
STRATEGY C — LONG-SHORT market-neutral momentum.
Long top momentum stocks, SHORT bottom momentum stocks (MID-cap tier). Market-neutral
(equal $ long/short) removes market risk -> low drawdown -> leverageable.
Also tests: mean-reversion (does buying LOSERS work short-term in India?).
Honest costs incl. short borrow. IS 2010-18 / OOS 2019-26.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np, pandas as pd, duckdb

CACHE="data_store/mom_v2_panels.parquet"; DB="data_store/piedpiper.duckdb"
CAP=200_000.0; PRICE_MIN=30.0; FROM_YEAR=2010; IS_END=2018
RT=2*(0.0010+0.0010)+0.0005; BORROW=0.06/12   # short borrow ~6%/yr

def load():
    pk=pd.read_parquet(CACHE); close=pk.xs("close",1,0); turn=pk.xs("turn",1,0)
    c=duckdb.connect(DB,read_only=True)
    n=c.execute("SELECT dt,close FROM adjusted_ohlcv WHERE symbol='Nifty 50' ORDER BY dt").df();c.close()
    n["dt"]=pd.to_datetime(n["dt"]); nifty=n.set_index("dt")["close"].resample("ME").last()
    return close,turn,nifty

def run(close,turn,nifty, mode="longshort", top_n=15, reversion=False):
    """mode: longshort (mkt-neutral) | longonly. reversion: rank ASC (buy losers)."""
    idx=close.index; nser=nifty.reindex(idx).ffill(); nema=nser.ewm(span=10,adjust=False).mean()
    ret=close.pct_change(); vol=ret.rolling(6).std()
    if reversion:
        sigp=close.shift(1)/close.shift(2)-1     # last 1-month return (short-term reversal)
    m3=close.shift(1)/close.shift(4)-1; m6=close.shift(1)/close.shift(7)-1; m12=close.shift(1)/close.shift(13)-1
    ma10=close.rolling(10).mean()
    prevL=set(); prevS=set(); rets=[]; cash_mo=(1.065)**(1/12)-1
    start=max(14,int(np.searchsorted(idx.year.values,FROM_YEAR)))
    for p in range(start,len(idx)-1):
        pr=close.iloc[p]; tn=turn.iloc[p]; hist=close.iloc[:p+1].notna().sum()
        base=pr.index[(pr>=PRICE_MIN)&(hist>=14)&tn.notna()&(tn>0)]
        if len(base)<100: rets.append((idx[p+1],0.0)); continue
        rank=tn[base].rank(pct=True); mid=rank.index[(rank>0.60)&(rank<=0.90)]  # MID tier (best from Strat A)
        if reversion:
            sc=sigp.iloc[p][mid].dropna()
            longs=list(sc.nsmallest(top_n).index); shorts=list(sc.nlargest(top_n).index)  # buy losers/sell winners
        else:
            score=pd.Series(0.0,index=list(mid)); nc=0
            for sg in (m3,m6,m12):
                x=(sg.iloc[p][mid]/vol.iloc[p][mid].replace(0,np.nan)).dropna()
                if len(x): score=score.add(x.rank(pct=True),fill_value=0); nc+=1
            if nc==0: rets.append((idx[p+1],0.0)); continue
            r_=(score/nc).sort_values(ascending=False)
            longs=list(r_.head(top_n).index); shorts=list(r_.tail(top_n).index)
        def fwd(names):
            xs=[close.iloc[p+1][s]/close.iloc[p][s]-1 for s in names if pd.notna(close.iloc[p+1].get(s)) and pd.notna(close.iloc[p][s])]
            return float(np.mean(xs)) if xs else 0.0
        lr=fwd(longs)
        if mode=="longonly":
            to=len(set(longs).symmetric_difference(prevL))/max(len(set(longs)|prevL),1)
            gross=lr - to*RT; prevL=set(longs)
        else:
            sr=fwd(shorts)
            toL=len(set(longs).symmetric_difference(prevL))/max(len(set(longs)|prevL),1)
            toS=len(set(shorts).symmetric_difference(prevS))/max(len(set(shorts)|prevS),1)
            gross=(lr - sr)/2 - (toL+toS)*RT - BORROW*0.5 + cash_mo*0.5  # neutral: half long half short, cash earns on short proceeds
            prevL=set(longs); prevS=set(shorts)
        rets.append((idx[p+1],gross))
    return pd.Series(dict(rets)).sort_index()

def stats(r,label,lev=1.0):
    r=(r*lev).dropna(); eq=CAP*(1+r).cumprod()
    ie=eq[eq.index.year<=IS_END]; oe=eq[eq.index.year>IS_END]
    def cd(e,s):
        if len(e)<2: return float('nan'),float('nan'),float('nan')
        yrs=max((e.index[-1]-e.index[0]).days/365.25,.1)
        return (e.iloc[-1]/s)**(1/yrs)-1,((e-e.cummax())/e.cummax()).min(),(e.pct_change().dropna().mean()/e.pct_change().dropna().std()*np.sqrt(12) if e.pct_change().dropna().std()>0 else float('nan'))
    ic,idd,ish=cd(ie,CAP); oc,odd,osh=cd(oe,ie.iloc[-1] if len(ie) else CAP); fc,fdd,fsh=cd(eq,CAP)
    print(f"  {label:<44} IS {ic*100:>+6.1f}%/Sh{ish:.2f} | OOS {oc*100:>+6.1f}%/DD{odd*100:.0f}%/Sh{osh:.2f} | Full {fc*100:>+6.1f}%")

def main():
    close,turn,nifty=load()
    print(f"{'='*104}\n  STRATEGY C — LONG-SHORT & MEAN-REVERSION (MID-cap, IS 2010-18/OOS 2019-26)\n{'='*104}")
    ls=run(close,turn,nifty,"longshort")
    stats(ls, "Long-short momentum (market-neutral)")
    stats(ls, "  same, 2x leverage (neutral=low risk)", lev=2.0)
    stats(ls, "  same, 3x leverage", lev=3.0)
    stats(run(close,turn,nifty,"longonly"), "Long-only momentum (MID) [reference]")
    print()
    stats(run(close,turn,nifty,"longshort",reversion=True), "Mean-reversion long-short (buy losers)")

if __name__=="__main__": main()
