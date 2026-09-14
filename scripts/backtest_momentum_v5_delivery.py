"""
Phase 2 — delivery-% quality filter on the momentum MAX config.
CAVEAT: delivery data only exists from 2020, so this is tested ONLY on 2020-2026
with a pseudo-split (IS' 2020-22 / OOS' 2023-26). This is NOT a proper OOS test
(one short regime) — treat any gain as a forward-test candidate, not a deploy.
Momentum signals still use full price history; only the delivery FILTER is 2020+.
"""
from __future__ import annotations
import sys, glob, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np, pandas as pd, duckdb

CACHE="data_store/mom_v2_panels.parquet"; DCACHE="data_store/mom_deliv_panel.parquet"
DAILY="data_store/eod2/src/eod2_data/daily"; DB="data_store/piedpiper.duckdb"
CAP=200_000.0; TOP_N=15; TURN_TOP=500; PRICE_MIN=30.0
RT=2*(0.0010+0.0010)+0.0005; CASH_MO=(1.065)**(1/12)-1
FROM_YEAR=2020; SPLIT=2022   # IS' <=2022, OOS' >=2023

def build_deliv():
    if os.path.exists(DCACHE):
        return pd.read_parquet(DCACHE)
    print("Building monthly delivery-% panel (reads CSVs, ~few min) ...", flush=True)
    out={}
    files=glob.glob(f"{DAILY}/*.csv")
    for i,f in enumerate(files):
        sym=Path(f).stem.upper()
        try: d=pd.read_csv(f, usecols=["Date","Volume","Series","DLV_QTY"])
        except Exception: continue
        d=d[(d["Series"]=="EQ") & d["DLV_QTY"].notna() & (d["Volume"]>0)]
        if len(d)<20: continue
        d["Date"]=pd.to_datetime(d["Date"])
        dp=(d.set_index("Date")["DLV_QTY"]/d.set_index("Date")["Volume"]).clip(0,1)
        out[sym]=dp.resample("ME").mean()
        if (i+1)%800==0: print(f"    [{i+1}/{len(files)}]", flush=True)
    dm=pd.DataFrame(out).sort_index()
    dm.to_parquet(DCACHE); print(f"  delivery panel: {dm.shape[1]} stocks × {dm.shape[0]} months", flush=True)
    return dm

def load():
    p=pd.read_parquet(CACHE); close=p.xs("close",axis=1,level=0); turn=p.xs("turn",axis=1,level=0)
    c=duckdb.connect(DB,read_only=True)
    n=c.execute("SELECT dt,close FROM adjusted_ohlcv WHERE symbol='Nifty 50' ORDER BY dt").df();c.close()
    n["dt"]=pd.to_datetime(n["dt"]); nifty=n.set_index("dt")["close"].resample("ME").last()
    return close,turn,nifty

def simulate(close,turn,nifty,deliv, deliv_drop=0.0, regime=True):
    idx=close.index; nser=nifty.reindex(idx).ffill(); nema=nser.ewm(span=10,adjust=False).mean()
    ret=close.pct_change(); vol=ret.rolling(6).std()
    m3=close.shift(1)/close.shift(4)-1; m6=close.shift(1)/close.shift(7)-1; m12=close.shift(1)/close.shift(13)-1
    ma10=close.rolling(10).mean(); deliv=deliv.reindex(idx)
    equity=CAP; prev=set(); rets=[]
    start=int(np.searchsorted(idx.year.values,FROM_YEAR))
    for p in range(start,len(idx)-1):
        in_mkt=(nser.iloc[p]>nema.iloc[p]) and (nser.iloc[p]/nser.iloc[p-6]-1>-0.02) if regime else True
        if in_mkt:
            price=close.iloc[p]; tn=turn.iloc[p]; hist=close.iloc[:p+1].notna().sum()
            base=price.index[(price>=PRICE_MIN)&(hist>=14)&tn.notna()]
            elig=list(tn[base].nlargest(min(TURN_TOP,len(base))).index)
            elig=[s for s in elig if pd.notna(ma10.iloc[p].get(s)) and price[s]>ma10.iloc[p][s]]  # trend confirm
            # delivery filter — drop bottom quantile by delivery% (only where data exists)
            if deliv_drop>0 and elig:
                dv=deliv.iloc[p][elig].dropna()
                if len(dv)>10:
                    keep=set(dv.nlargest(int(len(dv)*(1-deliv_drop))).index)
                    elig=[s for s in elig if s in keep or s not in dv.index]  # missing = neutral (keep)
            # multi-timeframe risk-adjusted composite
            score=pd.Series(0.0,index=elig); nc=0
            for sg in (m3,m6,m12):
                s=(sg.iloc[p][elig]/vol.iloc[p][elig].replace(0,np.nan)).dropna()
                if len(s): score=score.add(s.rank(pct=True),fill_value=0); nc+=1
            if nc==0: gross,picks=CASH_MO,set()
            else:
                ranked=(score/nc).sort_values(ascending=False)
                rk={s:i for i,s in enumerate(ranked.index)}
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
        rets.append((idx[p+1],gross-to*RT)); prev=picks
    return pd.Series(dict(rets)).sort_index()

def metrics(eq,start):
    yrs=max((eq.index[-1]-eq.index[0]).days/365.25,0.1)
    cagr=(eq.iloc[-1]/start)**(1/yrs)-1; dd=((eq-eq.cummax())/eq.cummax()).min()
    r=eq.pct_change().dropna(); sh=(r.mean()/r.std()*np.sqrt(12)) if r.std()>0 else float("nan")
    return cagr,dd,sh

def show(rets,label):
    eq=CAP*(1+rets).cumprod()
    a=eq[eq.index.year<=SPLIT]; b=eq[eq.index.year>SPLIT]
    ac,_,ash=metrics(a,CAP); bc,bdd,bsh=metrics(b,a.iloc[-1]); fc,fdd,fsh=metrics(eq,CAP)
    print(f"  {label:<40} IS'20-22 {ac*100:>+5.1f}%/Sh{ash:.2f} | OOS'23-26 {bc*100:>+5.1f}%/DD{bdd*100:.0f}%/Sh{bsh:.2f} | Full {fc*100:+.1f}%")

def main():
    deliv=build_deliv(); close,turn,nifty=load()
    print(f"\n{'='*104}\n  PHASE 2 — delivery filter on Momentum MAX  (2020-2026 ONLY; pseudo-split, NOT proper OOS)\n{'='*104}")
    show(simulate(close,turn,nifty,deliv, deliv_drop=0.0), "MAX (multi-TF+trend), no delivery filter")
    show(simulate(close,turn,nifty,deliv, deliv_drop=0.3), "+ drop bottom 30% delivery%")
    show(simulate(close,turn,nifty,deliv, deliv_drop=0.5), "+ drop bottom 50% delivery%")

if __name__=="__main__": main()
