"""
Momentum backtest with IS 2000-2012 / OOS 2013-2026.
DATA CAVEAT: no real Nifty pre-2007 -> build a SYNTHETIC market index from the
universe (equal-weight top-100 by turnover, rebalanced monthly), VERIFIED against
real Nifty on the 2007+ overlap. Pre-2008 universe is thin & survivorship-heavy
(272 stocks in 2000) -> early results inflated; treat IS as a rough stress test.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np, pandas as pd, duckdb

CACHE="data_store/mom_v2_panels.parquet"; DB="data_store/piedpiper.duckdb"
CAP=200_000.0; TOP_N=15; TURN_TOP=500; PRICE_MIN=30.0
RT=2*(0.0010+0.0010)+0.0005; CASH_MO=(1.065)**(1/12)-1; IS_END=2012

def synth_market(close, turn):
    """Equal-weight top-100-by-turnover monthly return -> chained index (proxy for market)."""
    ret=close.pct_change(); idx=close.index; mkt=[]
    for p in range(1,len(idx)):
        sel=turn.iloc[p-1].nlargest(100).index
        r=ret.iloc[p][sel].dropna()
        mkt.append((idx[p], float(r.mean()) if len(r) else 0.0))
    s=pd.Series(dict(mkt)); return (1+s).cumprod()

def verify_proxy(mkt):
    c=duckdb.connect(DB,read_only=True)
    n=c.execute("SELECT dt,close FROM adjusted_ohlcv WHERE symbol='Nifty 50' ORDER BY dt").df();c.close()
    n["dt"]=pd.to_datetime(n["dt"]); nm=n.set_index("dt")["close"].resample("ME").last()
    a=mkt.pct_change(); b=nm.pct_change(); j=a.index.intersection(b.index)
    j=j[j.year>=2007]
    corr=a.loc[j].corr(b.loc[j])
    print(f"  proxy verify vs real Nifty (2007+): monthly-return corr = {corr:+.2f}  ({'OK' if corr>0.85 else 'WEAK'})")
    return corr

def sim(close, turn, mkt, from_year):
    idx=close.index; m=mkt.reindex(idx).ffill(); mema=m.ewm(span=10,adjust=False).mean()
    ret=close.pct_change(); vol=ret.rolling(6).std()
    s3=close.shift(1)/close.shift(4)-1; s6=close.shift(1)/close.shift(7)-1; s12=close.shift(1)/close.shift(13)-1
    ma10=close.rolling(10).mean(); above=(close>ma10)
    equity=CAP; prev=set(); rets=[]
    start=max(14,int(np.searchsorted(idx.year.values,from_year)))
    for p in range(start,len(idx)-1):
        mkt_ok=(m.iloc[p]>mema.iloc[p]) and (m.iloc[p]/m.iloc[p-6]-1>-0.02 if p>=6 else True)
        pr=close.iloc[p]; tn=turn.iloc[p]
        libase=pr.index[(pr>=PRICE_MIN)&tn.notna()]
        liq=tn[libase].nlargest(min(TURN_TOP,len(libase))).index
        breadth=above.iloc[p][liq].mean() if len(liq) else 0.5
        in_mkt=mkt_ok and breadth>0.45
        if in_mkt:
            hist=close.iloc[:p+1].notna().sum()
            elig=[s for s in liq if hist.get(s,0)>=14 and pr[s]>ma10.iloc[p].get(s,1e9)]
            score=pd.Series(0.0,index=elig); nc=0
            for sg in (s3,s6,s12):
                x=(sg.iloc[p][elig]/vol.iloc[p][elig].replace(0,np.nan)).dropna()
                if len(x): score=score.add(x.rank(pct=True),fill_value=0); nc+=1
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
        else: gross,picks=CASH_MO,set()
        to=len(picks.symmetric_difference(prev))/max(len(picks|prev),1)
        equity*=(1+gross-to*RT); rets.append((idx[p+1],equity)); prev=picks
    return pd.Series(dict(rets)).sort_index()

def metrics(eq,start):
    yrs=max((eq.index[-1]-eq.index[0]).days/365.25,0.1)
    cagr=(eq.iloc[-1]/start)**(1/yrs)-1; dd=((eq-eq.cummax())/eq.cummax()).min()
    r=eq.pct_change().dropna(); sh=(r.mean()/r.std()*np.sqrt(12)) if r.std()>0 else float("nan")
    return cagr,dd,sh

def main():
    pk=pd.read_parquet(CACHE); close=pk.xs("close",1,0); turn=pk.xs("turn",1,0)
    close=close[close.index.year>=1998]; turn=turn.reindex(close.index)
    print("Building synthetic market index (2000+) ...", flush=True)
    mkt=synth_market(close,turn)
    verify_proxy(mkt)
    eq=sim(close,turn,mkt, from_year=2000)
    is_e=eq[eq.index.year<=IS_END]; oos=eq[eq.index.year>IS_END]
    print(f"\n{'='*74}\n  MOMENTUM  IS 2000-2012 / OOS 2013-2026  (synthetic-market regime)\n{'='*74}")
    ic,idd,ish=metrics(is_e,CAP); oc,odd,osh=metrics(oos,is_e.iloc[-1]); fc,fdd,fsh=metrics(eq,CAP)
    print(f"  IS  2000-2012 : CAGR {ic*100:+.1f}%  MaxDD {idd*100:.0f}%  Sharpe {ish:.2f}   ⚠️ survivorship-inflated (thin pre-2008 universe)")
    print(f"  OOS 2013-2026 : CAGR {oc*100:+.1f}%  MaxDD {odd*100:.0f}%  Sharpe {osh:.2f}   (reliable data)")
    print(f"  FULL 2000-2026: CAGR {fc*100:+.1f}%  MaxDD {fdd*100:.0f}%  Sharpe {fsh:.2f}")

if __name__=="__main__": main()
