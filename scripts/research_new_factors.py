"""
NEW FACTOR RESEARCH — genuinely unexplored ideas that could add RETURN.
On MID-cap tier (best from Strategy A). Honest costs, IS 2010-18 / OOS 2019-26.
  1. Concentration: top-5/8/10/15/20 (does fewer picks raise return?)
  2. Acceleration momentum (2nd derivative: rank by change-in-momentum)
  3. Value proxy = LONG-TERM reversal (5yr losers = cheap; the classic value factor)
  4. Value + Momentum COMBINATION (Asness — the most robust combo in quant)
  5. Momentum + low-vol tilt (defensive momentum)
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np, pandas as pd, duckdb

CACHE="data_store/mom_v2_panels.parquet"; DB="data_store/piedpiper.duckdb"
CAP=200_000.0; PRICE_MIN=30.0; FROM_YEAR=2010; IS_END=2018
RT=2*(0.0010+0.0010)+0.0005

def load():
    pk=pd.read_parquet(CACHE); close=pk.xs("close",1,0); turn=pk.xs("turn",1,0)
    c=duckdb.connect(DB,read_only=True)
    n=c.execute("SELECT dt,close FROM adjusted_ohlcv WHERE symbol='Nifty 50' ORDER BY dt").df();c.close()
    n["dt"]=pd.to_datetime(n["dt"]); nifty=n.set_index("dt")["close"].resample("ME").last()
    return close,turn,nifty

def run(close,turn,nifty, signal="mom", top_n=15, lowvol_tilt=False):
    idx=close.index; nser=nifty.reindex(idx).ffill(); nema=nser.ewm(span=10,adjust=False).mean()
    ret=close.pct_change(); vol=ret.rolling(6).std()
    m3=close.shift(1)/close.shift(4)-1; m6=close.shift(1)/close.shift(7)-1; m12=close.shift(1)/close.shift(13)-1
    accel=(close.shift(1)/close.shift(4)-1)-(close.shift(4)/close.shift(7)-1)  # recent 3mo minus prior 3mo (acceleration)
    rev5=-(close.shift(13)/close.shift(61)-1)  # long-term reversal: NEGATIVE of 4yr-ago-to-1yr-ago return (buy long-term losers=value)
    ma10=close.rolling(10).mean(); above=(close>ma10)
    prev=set(); rets=[]; cash_mo=(1.065)**(1/12)-1
    start=max(62,int(np.searchsorted(idx.year.values,FROM_YEAR)))
    for p in range(start,len(idx)-1):
        pr=close.iloc[p]; tn=turn.iloc[p]; hist=close.iloc[:p+1].notna().sum()
        base=pr.index[(pr>=PRICE_MIN)&(hist>=62)&tn.notna()&(tn>0)]
        if len(base)<100: rets.append((idx[p+1],cash_mo)); continue
        rank=tn[base].rank(pct=True); mid=list(rank.index[(rank>0.60)&(rank<=0.90)])  # MID tier
        breadth=above.iloc[p][base].mean()
        risk_on=(nser.iloc[p]>nema.iloc[p]) and (nser.iloc[p]/nser.iloc[p-6]-1>-0.02) and breadth>0.45
        if not risk_on: rets.append((idx[p+1],cash_mo)); prev=set(); continue
        elig=[s for s in mid if pr[s]>ma10.iloc[p].get(s,1e9)]
        def zrank(sig):  # cross-sectional rank of a signal among elig
            s=sig.iloc[p][elig].dropna(); return s.rank(pct=True)
        if signal=="mom":
            sc=pd.Series(0.0,index=elig); nc=0
            for sg in (m3,m6,m12):
                x=(sg.iloc[p][elig]/vol.iloc[p][elig].replace(0,np.nan)).dropna()
                if len(x): sc=sc.add(x.rank(pct=True),fill_value=0); nc+=1
            score=sc/max(nc,1)
        elif signal=="accel":
            score=zrank(accel)
        elif signal=="value":
            score=zrank(rev5)
        elif signal=="valmom":            # 50/50 blend of value + momentum ranks
            sc=pd.Series(0.0,index=elig); nc=0
            for sg in (m3,m6,m12):
                x=(sg.iloc[p][elig]/vol.iloc[p][elig].replace(0,np.nan)).dropna()
                if len(x): sc=sc.add(x.rank(pct=True),fill_value=0); nc+=1
            momr=sc/max(nc,1); valr=zrank(rev5)
            score=(momr.add(valr,fill_value=0))/2
        if lowvol_tilt:
            lv=(1/vol.iloc[p][elig].replace(0,np.nan)).dropna().rank(pct=True)  # low vol = high rank
            score=(score.add(lv*0.5,fill_value=0))
        score=score.dropna()
        if len(score)<top_n: rets.append((idx[p+1],cash_mo)); prev=set(); continue
        ranked=score.sort_values(ascending=False); rk={s:i for i,s in enumerate(ranked.index)}
        picks=[s for s in prev if rk.get(s,10**9)<top_n*1.67]
        for s in ranked.index:
            if len(picks)>=top_n: break
            if s not in picks: picks.append(s)
        picks=set(picks[:top_n])
        r=[close.iloc[p+1][s]/close.iloc[p][s]-1 for s in picks if pd.notna(close.iloc[p+1].get(s)) and pd.notna(close.iloc[p][s])]
        gross=float(np.mean(r)) if r else cash_mo
        to=len(picks.symmetric_difference(prev))/max(len(picks|prev),1)
        rets.append((idx[p+1],gross-to*RT)); prev=picks
    return pd.Series(dict(rets)).sort_index()

def stats(r,label):
    r=r.dropna(); eq=CAP*(1+r).cumprod(); ie=eq[eq.index.year<=IS_END]; oe=eq[eq.index.year>IS_END]
    def cd(e,s):
        if len(e)<2: return float('nan'),float('nan'),float('nan')
        y=max((e.index[-1]-e.index[0]).days/365.25,.1)
        return (e.iloc[-1]/s)**(1/y)-1,((e-e.cummax())/e.cummax()).min(),(e.pct_change().dropna().mean()/e.pct_change().dropna().std()*np.sqrt(12) if e.pct_change().dropna().std()>0 else float('nan'))
    ic,idd,ish=cd(ie,CAP); oc,odd,osh=cd(oe,ie.iloc[-1] if len(ie) else CAP); fc,fdd,fsh=cd(eq,CAP)
    print(f"  {label:<40} IS {ic*100:>+6.1f}%/Sh{ish:.2f} | OOS {oc*100:>+6.1f}%/DD{odd*100:.0f}%/Sh{osh:.2f} | Full {fc*100:>+6.1f}%")

def main():
    close,turn,nifty=load()
    print(f"{'='*104}\n  NEW FACTOR RESEARCH (MID-cap, IS 2010-18/OOS 2019-26, honest costs)\n{'='*104}")
    print("  -- CONCENTRATION (does holding fewer raise return?) --")
    for n in (5,8,10,15,20):
        stats(run(close,turn,nifty,"mom",top_n=n), f"momentum, top-{n}")
    print("\n  -- NEW SIGNALS --")
    stats(run(close,turn,nifty,"mom",15),   "momentum [baseline]")
    stats(run(close,turn,nifty,"accel",15), "acceleration momentum")
    stats(run(close,turn,nifty,"value",15), "value (5yr long-term reversal)")
    stats(run(close,turn,nifty,"valmom",15),"VALUE + MOMENTUM combo (Asness)")
    stats(run(close,turn,nifty,"mom",15,lowvol_tilt=True), "momentum + low-vol tilt (defensive)")

if __name__=="__main__": main()
