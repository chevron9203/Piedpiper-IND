"""
UNIVERSE SIZE research — does a BIGGER candidate pool surface better momentum trades?
Screen top-N by turnover, rank by momentum, hold top-15. KEY HONESTY: slippage
scales with each PICKED stock's liquidity (illiquid names cost more) — so a bigger
pool that surfaces illiquid momentum names PAYS for it. IS 2010-18 / OOS 2019-26.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np, pandas as pd, duckdb

CACHE="data_store/mom_v2_panels.parquet"; DB="data_store/piedpiper.duckdb"
CAP=200_000.0; PRICE_MIN=30.0; FROM_YEAR=2010; IS_END=2018; TOP_N=15

def load():
    pk=pd.read_parquet(CACHE); close=pk.xs("close",1,0); turn=pk.xs("turn",1,0)
    c=duckdb.connect(DB,read_only=True)
    n=c.execute("SELECT dt,close FROM adjusted_ohlcv WHERE symbol='Nifty 50' ORDER BY dt").df();c.close()
    n["dt"]=pd.to_datetime(n["dt"]); nifty=n.set_index("dt")["close"].resample("ME").last()
    return close,turn,nifty

def liq_slippage(turnover_rank_in_full):
    """Realistic per-side slippage by how liquid the stock is (rank among ALL stocks)."""
    r=turnover_rank_in_full
    if r<=300:   return 0.0010    # very liquid
    if r<=750:   return 0.0020
    if r<=1500:  return 0.0035
    return 0.0060                  # illiquid — brutal

def run(close,turn,nifty, univ_size, liquidity_aware=True):
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
        full_rank={s:i+1 for i,s in enumerate(tn[base].sort_values(ascending=False).index)}  # 1=most liquid
        univ=[s for s in tn[base].nlargest(min(univ_size,len(base))).index]                  # candidate pool
        breadth=above.iloc[p][base].mean()
        risk_on=(nser.iloc[p]>nema.iloc[p]) and (nser.iloc[p]/nser.iloc[p-6]-1>-0.02) and breadth>0.45
        if not risk_on: rets.append((idx[p+1],cash_mo)); prev=set(); continue
        elig=[s for s in univ if pr[s]>ma10.iloc[p].get(s,1e9)]
        score=pd.Series(0.0,index=elig); nc=0
        for sg in (m3,m6,m12):
            x=(sg.iloc[p][elig]/vol.iloc[p][elig].replace(0,np.nan)).dropna()
            if len(x): score=score.add(x.rank(pct=True),fill_value=0); nc+=1
        if nc==0 or len(elig)<TOP_N: rets.append((idx[p+1],cash_mo)); prev=set(); continue
        ranked=(score/nc).sort_values(ascending=False); rk={s:i for i,s in enumerate(ranked.index)}
        picks=[s for s in prev if rk.get(s,10**9)<TOP_N*1.67]
        for s in ranked.index:
            if len(picks)>=TOP_N: break
            if s not in picks: picks.append(s)
        picks=picks[:TOP_N]
        # gross return + per-stock liquidity-aware slippage on turnover
        rr=[]; cost=0.0
        for s in picks:
            if pd.notna(close.iloc[p+1].get(s)) and pd.notna(close.iloc[p].get(s)):
                rr.append(close.iloc[p+1][s]/close.iloc[p][s]-1)
            if s not in prev:  # newly bought this month -> pay entry slippage (round trip amortized)
                slp=liq_slippage(full_rank.get(s,9999)) if liquidity_aware else 0.0010
                cost+=2*slp/TOP_N   # round-trip, spread across the book
        gross=float(np.mean(rr)) if rr else cash_mo
        rets.append((idx[p+1],gross-cost)); prev=set(picks)
    return pd.Series(dict(rets)).sort_index()

def stats(r,label):
    r=r.dropna(); eq=CAP*(1+r).cumprod(); ie=eq[eq.index.year<=IS_END]; oe=eq[eq.index.year>IS_END]
    def cd(e,s):
        if len(e)<2: return float('nan'),float('nan'),float('nan')
        y=max((e.index[-1]-e.index[0]).days/365.25,.1)
        return (e.iloc[-1]/s)**(1/y)-1,((e-e.cummax())/e.cummax()).min(),(e.pct_change().dropna().mean()/e.pct_change().dropna().std()*np.sqrt(12) if e.pct_change().dropna().std()>0 else float('nan'))
    ic,idd,ish=cd(ie,CAP); oc,odd,osh=cd(oe,ie.iloc[-1] if len(ie) else CAP); fc,fdd,fsh=cd(eq,CAP)
    print(f"  {label:<44} IS {ic*100:>+6.1f}% | OOS {oc*100:>+6.1f}%/DD{odd*100:.0f}%/Sh{osh:.2f} | Full {fc*100:>+6.1f}%")

def main():
    close,turn,nifty=load()
    print(f"{'='*100}\n  DOES A BIGGER UNIVERSE HELP?  (screen top-N by turnover, hold top-15)\n{'='*100}")
    print("  -- LIQUIDITY-AWARE slippage (honest: illiquid picks cost more) --")
    for u in (300,500,750,1000,1500,2500):
        stats(run(close,turn,nifty,u,liquidity_aware=True), f"universe top-{u} by turnover")
    print("\n  -- FLAT 0.10% slippage (optimistic — ignores illiquidity cost) --")
    for u in (500,1000,2500):
        stats(run(close,turn,nifty,u,liquidity_aware=False), f"universe top-{u} (flat slippage)")

if __name__=="__main__": main()
