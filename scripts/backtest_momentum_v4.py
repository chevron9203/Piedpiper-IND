"""
Momentum v4 — maximize the momentum sleeve. Ablation ladder, IS/OOS validated.
Baseline = v3 best (risk-adj 12-1 + buffer + regime). Each row adds ONE a-priori,
research-backed improvement; keep only what helps OOS robustly (no OOS grid-search).
Signals tested: multi-timeframe momentum, 52-week-high proximity, stock trend
confirmation, volatility cap.  All price-only (cached panel).
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np, pandas as pd, duckdb

CACHE="data_store/mom_v2_panels.parquet"; DB_PATH="data_store/piedpiper.duckdb"
CAPITAL=200_000.0; TOP_N=15; TURN_TOP=500; PRICE_MIN=30.0
RT_COST=2*(0.0010+0.0010)+0.0005; CASH_MO=(1.065)**(1/12)-1
FROM_YEAR=2010; IS_END=2018

def load():
    p=pd.read_parquet(CACHE); close=p.xs("close",axis=1,level=0); turn=p.xs("turn",axis=1,level=0)
    c=duckdb.connect(DB_PATH,read_only=True)
    n=c.execute("SELECT dt,close FROM adjusted_ohlcv WHERE symbol='Nifty 50' ORDER BY dt").df();c.close()
    n["dt"]=pd.to_datetime(n["dt"]); nifty=n.set_index("dt")["close"].resample("ME").last()
    return close,turn,nifty

def precompute(close):
    ret=close.pct_change(); vol=ret.rolling(6).std()
    sig={
      "m3":  close.shift(1)/close.shift(4)-1,
      "m6":  close.shift(1)/close.shift(7)-1,
      "m12": close.shift(1)/close.shift(13)-1,
      "prox":close/close.rolling(12).max(),      # 52wk-high proximity (monthly proxy)
    }
    ma10=close.rolling(10).mean()
    return sig, vol, ma10

def simulate(close,turn,nifty,sig,vol,ma10, signals=("m12",), risk_adj=True,
             trend_confirm=False, vol_cap=0.0, buffer=1.67, regime=True):
    idx=close.index; nser=nifty.reindex(idx).ffill(); nema=nser.ewm(span=10,adjust=False).mean()
    equity=CAPITAL; prev=set(); rets=[]; start=14
    start=max(14,int(np.searchsorted(idx.year.values,FROM_YEAR)))
    for p in range(start,len(idx)-1):
        in_mkt=True
        if regime:
            n6=nser.iloc[p]/nser.iloc[p-6]-1 if p>=6 else 0
            in_mkt=(nser.iloc[p]>nema.iloc[p]) and (n6>-0.02)
        if in_mkt:
            price=close.iloc[p]; tn=turn.iloc[p]; hist=close.iloc[:p+1].notna().sum()
            base=price.index[(price>=PRICE_MIN)&(hist>=14)&tn.notna()]
            liq=tn[base].nlargest(min(TURN_TOP,len(base))).index
            elig=list(liq)
            if trend_confirm:
                elig=[s for s in elig if pd.notna(ma10.iloc[p].get(s)) and price[s]>ma10.iloc[p][s]]
            if vol_cap>0 and elig:
                v=vol.iloc[p][elig].dropna()
                keep=v.nsmallest(int(len(v)*(1-vol_cap))).index
                elig=[s for s in elig if s in keep]
            # composite score = mean cross-sectional rank across chosen signals
            score=pd.Series(0.0,index=elig); ncomp=0
            for name in signals:
                s=sig[name].iloc[p][elig]
                if risk_adj and name!="prox":
                    s=s/vol.iloc[p][elig].replace(0,np.nan)
                s=s.dropna()
                if len(s)==0: continue
                score=score.add(s.rank(pct=True),fill_value=0); ncomp+=1
            if ncomp==0: gross,picks=CASH_MO,set()
            else:
                ranked=(score/ncomp).sort_values(ascending=False)
                if buffer>1.0 and prev:
                    rk={s:i for i,s in enumerate(ranked.index)}
                    picks=[s for s in prev if rk.get(s,10**9)<TOP_N*buffer]
                    for s in ranked.index:
                        if len(picks)>=TOP_N: break
                        if s not in picks: picks.append(s)
                    picks=set(picks[:TOP_N])
                else: picks=set(ranked.head(TOP_N).index)
                r=[close.iloc[p+1][s]/close.iloc[p][s]-1 for s in picks
                   if pd.notna(close.iloc[p+1].get(s)) and pd.notna(close.iloc[p][s])]
                gross=float(np.mean(r)) if r else CASH_MO
        else: gross,picks=CASH_MO,set()
        turnover=len(picks.symmetric_difference(prev))/max(len(picks|prev),1)
        rets.append((idx[p+1], gross-turnover*RT_COST)); prev=picks
    return pd.Series(dict(rets)).sort_index()

def metrics(eq,start):
    yrs=max((eq.index[-1]-eq.index[0]).days/365.25,0.1)
    cagr=(eq.iloc[-1]/start)**(1/yrs)-1; dd=((eq-eq.cummax())/eq.cummax()).min()
    r=eq.pct_change().dropna(); sh=(r.mean()/r.std()*np.sqrt(12)) if r.std()>0 else float("nan")
    return cagr,dd,sh

def show(rets,label):
    eq=CAPITAL*(1+rets).cumprod()
    is_e=eq[eq.index.year<=IS_END]; oos=eq[eq.index.year>IS_END]
    ic,_,ish=metrics(is_e,CAPITAL); oc,odd,osh=metrics(oos,is_e.iloc[-1]); fc,fdd,_=metrics(eq,CAPITAL)
    print(f"  {label:<46} IS {ic*100:>+5.1f}%/Sh{ish:.2f} | OOS {oc*100:>+5.1f}%/DD{odd*100:.0f}%/Sh{osh:.2f} | Full {fc*100:+.1f}%")

def main():
    close,turn,nifty=load(); sig,vol,ma10=precompute(close)
    print(f"universe {close.shape[1]} | 2010-2026 | IS≤2018 / OOS≥2019\n")
    print("="*112)
    print("  MOMENTUM MAX — ablation ladder (keep only OOS-robust improvements)")
    print("="*112)
    def run(**kw): return simulate(close,turn,nifty,sig,vol,ma10,**kw)
    show(run(signals=("m12",)),                                   "baseline: risk-adj 12-1 + buffer")
    show(run(signals=("m3","m6","m12")),                          "A. + multi-timeframe (3/6/12)")
    show(run(signals=("m3","m6","m12","prox")),                   "B. + 52wk-high proximity")
    show(run(signals=("m3","m6","m12"), trend_confirm=True),      "C. + stock trend confirm")
    show(run(signals=("m3","m6","m12"), vol_cap=0.2),             "D. + vol cap (drop top 20% vol)")
    print("  "+"-"*108)
    print("  clean combos (drop proximity, which hurt):")
    show(run(signals=("m3","m6","m12"), trend_confirm=True),               "E1. multi + trend")
    show(run(signals=("m3","m6","m12"), vol_cap=0.2),                      "E2. multi + volcap")
    show(run(signals=("m3","m6","m12"), trend_confirm=True, vol_cap=0.2),  "E3. multi + trend + volcap  ← candidate MAX")
    show(run(signals=("m3","m6","m12"), trend_confirm=True, vol_cap=0.3),  "E4. multi + trend + volcap(30%)")
    # nifty bench OOS
    ne=CAPITAL*(nifty[nifty.index.year>=FROM_YEAR].pct_change().fillna(0)+1).cumprod()
    nc,ndd,nsh=metrics(ne[ne.index.year>IS_END],ne[ne.index.year<=IS_END].iloc[-1])
    print(f"\n  Nifty benchmark: OOS {nc*100:+.1f}% / DD {ndd*100:.0f}% / Sharpe {nsh:.2f}")

if __name__=="__main__": main()
