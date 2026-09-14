"""
Rebalance frequency test — MONTHLY vs WEEKLY momentum, honest costs.
Same signal (multi-TF risk-adj momentum + trend + buffer, broad universe);
only the rebalance cadence differs. Reports NET CAGR, turnover/yr, cost drag, DD.
Note: weekly = all STCG (tax-poison); monthly already mostly STCG too — this is
PRE-tax, so weekly's real disadvantage is even larger after tax.
"""
from __future__ import annotations
import sys, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np, pandas as pd, duckdb

DAILY="data_store/eod2/src/eod2_data/daily"; WCACHE="data_store/mom_weekly_panels.parquet"
MCACHE="data_store/mom_v2_panels.parquet"; DB="data_store/piedpiper.duckdb"
CAP=200_000.0; TOP_N=15; TURN_TOP=500; PRICE_MIN=30.0
SLIP=0.0010; STT=0.0010; RT=2*(SLIP+STT)+0.0005; CASH_YR=0.06; FROM_YEAR=2010; IS_END=2018
import os

def build_weekly():
    if os.path.exists(WCACHE):
        p=pd.read_parquet(WCACHE); return p.xs("close",1,0), p.xs("turn",1,0)
    print("Building WEEKLY panels from eod2 (first run) ...", flush=True)
    closes,turns={},{}
    for i,f in enumerate(glob.glob(f"{DAILY}/*.csv")):
        sym=Path(f).stem.upper()
        try: d=pd.read_csv(f,usecols=["Date","Close","Volume","Series"])
        except Exception: continue
        d=d[d["Series"]=="EQ"]
        if len(d)<260: continue
        d["Date"]=pd.to_datetime(d["Date"]); d=d.set_index("Date").sort_index()
        closes[sym]=d["Close"].resample("W-FRI").last()
        turns[sym]=(d["Close"]*d["Volume"]).resample("W-FRI").mean()
        if (i+1)%800==0: print(f"  [{i+1}]",flush=True)
    close=pd.DataFrame(closes).sort_index(); turn=pd.DataFrame(turns).reindex(close.index)
    pd.concat({"close":close,"turn":turn},axis=1).to_parquet(WCACHE)
    return close,turn

def load_nifty(freq):
    c=duckdb.connect(DB,read_only=True)
    n=c.execute("SELECT dt,close FROM adjusted_ohlcv WHERE symbol='Nifty 50' ORDER BY dt").df();c.close()
    n["dt"]=pd.to_datetime(n["dt"]); return n.set_index("dt")["close"].resample(freq).last()

def sim(close,turn,nifty, lb, vol_w, ma_w, per_year):
    """lb=(short,med,long,skip) in periods; per_year for cost/annualisation."""
    idx=close.index; nser=nifty.reindex(idx).ffill(); nema=nser.ewm(span=ma_w,adjust=False).mean()
    ret=close.pct_change(); vol=ret.rolling(vol_w).std()
    sh,me,lo,sk=lb
    m_s=close.shift(sk)/close.shift(sk+sh)-1; m_m=close.shift(sk)/close.shift(sk+me)-1; m_l=close.shift(sk)/close.shift(sk+lo)-1
    ma=close.rolling(ma_w).mean(); above=(close>ma)
    cash_p=(1+CASH_YR)**(1/per_year)-1
    prev=set(); rets=[]; turns=[]; start=lo+sk+2
    n6=int(per_year/2)
    for p in range(start,len(idx)-1):
        nifty_ok=(nser.iloc[p]>nema.iloc[p]) and (nser.iloc[p]/nser.iloc[p-n6]-1>-0.02)
        pr=close.iloc[p]; tn=turn.iloc[p]
        libase=pr.index[(pr>=PRICE_MIN)&tn.notna()]
        liq=tn[libase].nlargest(min(TURN_TOP,len(libase))).index
        breadth=above.iloc[p][liq].mean() if len(liq) else 0.5
        in_mkt=nifty_ok and breadth>0.45
        if in_mkt:
            hist=close.iloc[:p+1].notna().sum()
            elig=[s for s in liq if hist.get(s,0)>=lo+sk+2 and pr[s]>ma.iloc[p].get(s,1e9)]
            score=pd.Series(0.0,index=elig); nc=0
            for sg in (m_s,m_m,m_l):
                s=(sg.iloc[p][elig]/vol.iloc[p][elig].replace(0,np.nan)).dropna()
                if len(s): score=score.add(s.rank(pct=True),fill_value=0); nc+=1
            if nc==0: gross,picks=cash_p,set()
            else:
                ranked=(score/nc).sort_values(ascending=False); rk={s:i for i,s in enumerate(ranked.index)}
                picks=[s for s in prev if rk.get(s,10**9)<TOP_N*1.67]
                for s in ranked.index:
                    if len(picks)>=TOP_N: break
                    if s not in picks: picks.append(s)
                picks=set(picks[:TOP_N])
                r=[close.iloc[p+1][s]/close.iloc[p][s]-1 for s in picks
                   if pd.notna(close.iloc[p+1].get(s)) and pd.notna(close.iloc[p][s])]
                gross=float(np.mean(r)) if r else cash_p
        else: gross,picks=cash_p,set()
        to=len(picks.symmetric_difference(prev))/max(len(picks|prev),1)
        turns.append(to); rets.append((idx[p+1],gross-to*RT)); prev=picks
    eq=CAP*(1+pd.Series(dict(rets)).sort_index()).cumprod()
    yrs=(eq.index[-1]-eq.index[0]).days/365.25
    cagr=(eq.iloc[-1]/CAP)**(1/yrs)-1; dd=((eq-eq.cummax())/eq.cummax()).min()
    oe=eq[eq.index.year>IS_END]; oc=(oe.iloc[-1]/eq[eq.index.year<=IS_END].iloc[-1])**(1/((oe.index[-1]-oe.index[0]).days/365.25))-1
    return cagr, dd, oc, np.mean(turns)*per_year   # annualized turnover

def main():
    mc=pd.read_parquet(MCACHE); mclose,mturn=mc.xs("close",1,0),mc.xs("turn",1,0)
    mclose=mclose[mclose.index.year>=FROM_YEAR-2]; mturn=mturn.reindex(mclose.index)
    wclose,wturn=build_weekly()
    wclose=wclose[wclose.index.year>=FROM_YEAR-2]; wturn=wturn.reindex(wclose.index)
    print(f"\n{'='*80}\n  REBALANCE FREQUENCY — MONTHLY vs WEEKLY (same signal, honest cost, PRE-tax)\n{'='*80}")
    print(f"  {'cadence':<10}{'Full CAGR':>11}{'OOS CAGR':>10}{'MaxDD':>8}{'turnover/yr':>13}")
    mc_c,mc_dd,mc_oc,mc_to=sim(mclose,mturn,load_nifty("ME"), (3,6,12,1),6,10,12)
    print(f"  {'MONTHLY':<10}{mc_c*100:>+10.1f}%{mc_oc*100:>+9.1f}%{mc_dd*100:>7.0f}%{mc_to*100:>11.0f}%")
    wc_c,wc_dd,wc_oc,wc_to=sim(wclose,wturn,load_nifty("W-FRI"), (13,26,52,4),26,40,52)
    print(f"  {'WEEKLY':<10}{wc_c*100:>+10.1f}%{wc_oc*100:>+9.1f}%{wc_dd*100:>7.0f}%{wc_to*100:>11.0f}%")
    print(f"\n  (weekly is PRE-tax here; its ~{wc_to*100:.0f}% turnover = all STCG @20% => subtract another few % net)")

if __name__=="__main__": main()
