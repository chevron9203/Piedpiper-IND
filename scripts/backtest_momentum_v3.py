"""
Momentum v3 — research-backed improvements over v2, IS/OOS validated.
Builds on the honest v2 (broad point-in-time universe, realistic costs) and adds:
  1. Risk-adjusted momentum ranking (return / trailing vol)   [residual-momentum spirit]
  2. Rank buffering (hold until rank falls out of top N*buffer) [cuts turnover]
  3. Volatility targeting (Barroso-Santa-Clara): scale exposure by
     target_vol / trailing realized vol -> tames momentum crashes; optional leverage.
Each improvement is a-priori (from literature), tested IS(2010-18)/OOS(2019-26).
"""
from __future__ import annotations
import sys, glob, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np, pandas as pd, duckdb

DAILY_DIR="data_store/eod2/src/eod2_data/daily"; CACHE="data_store/mom_v2_panels.parquet"
DB_PATH="data_store/piedpiper.duckdb"; CAPITAL=200_000.0
TOP_N=15; LB,SKIP=12,1; TURN_TOP=500; PRICE_MIN=30.0
SLIP=0.0010; STT=0.0010; RT_COST=2*(SLIP+STT)+0.0005
CASH_MO=(1.065)**(1/12)-1; BORROW_MO=(1.09)**(1/12)-1   # 9%/yr margin funding for leverage
FROM_YEAR=2010; IS_END=2018


def build_panels():
    p=pd.read_parquet(CACHE)
    return p.xs("close",axis=1,level=0), p.xs("turn",axis=1,level=0)

def load_nifty():
    c=duckdb.connect(DB_PATH,read_only=True)
    n=c.execute("SELECT dt,close FROM adjusted_ohlcv WHERE symbol='Nifty 50' ORDER BY dt").df();c.close()
    n["dt"]=pd.to_datetime(n["dt"]); return n.set_index("dt")["close"].resample("ME").last()

def simulate(close_m, turn_m, nifty_m, rank_mode="raw", buffer=1.0, use_regime=True):
    idx=close_m.index
    nser=nifty_m.reindex(idx).ffill(); nema=nser.ewm(span=10,adjust=False).mean()
    ret_m=close_m.pct_change(); vol_m=ret_m.rolling(6).std()
    mom=close_m.shift(SKIP)/close_m.shift(SKIP+LB)-1.0
    equity=CAPITAL; prev=set(); curve=[]; rets=[]; inv=0
    start=max(SKIP+LB+1,int(np.searchsorted(idx.year.values,FROM_YEAR)))
    for p in range(start,len(idx)-1):
        reb,nxt=idx[p],idx[p+1]; in_mkt=True
        if use_regime:
            n6=nser.iloc[p]/nser.iloc[p-6]-1 if p>=6 else 0
            in_mkt=(nser.iloc[p]>nema.iloc[p]) and (n6>-0.02)
        if in_mkt:
            price=close_m.iloc[p]; turn=turn_m.iloc[p]; hist=close_m.iloc[:p+1].notna().sum()
            elig=price.index[(price>=PRICE_MIN)&(hist>=LB+SKIP+1)&mom.iloc[p].notna()&turn.notna()]
            if len(elig)==0: gross,picks=CASH_MO,set()
            else:
                liq=turn[elig].nlargest(min(TURN_TOP,len(elig))).index
                score=mom.iloc[p][liq]
                if rank_mode=="riskadj":
                    v=vol_m.iloc[p][liq].replace(0,np.nan)
                    score=(score/v).dropna()
                ranked=score.sort_values(ascending=False)
                if buffer>1.0 and prev:
                    rank={s:i for i,s in enumerate(ranked.index)}
                    kept=[s for s in prev if rank.get(s,10**9) < TOP_N*buffer]
                    picks=list(kept)
                    for s in ranked.index:
                        if len(picks)>=TOP_N: break
                        if s not in picks: picks.append(s)
                    picks=set(picks[:TOP_N])
                else:
                    picks=set(ranked.head(TOP_N).index)
                r=[close_m.iloc[p+1][s]/close_m.iloc[p][s]-1 for s in picks
                   if pd.notna(close_m.iloc[p+1].get(s)) and pd.notna(close_m.iloc[p][s])]
                gross=float(np.mean(r)) if r else CASH_MO; inv+=1
        else: gross,picks=CASH_MO,set()
        turnover=len(picks.symmetric_difference(prev))/max(len(picks|prev),1)
        net=gross-turnover*RT_COST
        equity*=(1+net); curve.append((nxt,equity)); rets.append((nxt,net)); prev=picks
    return pd.Series(dict(curve)).sort_index(), pd.Series(dict(rets)).sort_index(), inv, len(curve)

def vol_target(monthly_ret, target_vol=0.15, lev_cap=1.0):
    """Barroso-Santa-Clara: scale exposure by target_vol / trailing realized vol (lagged, no lookahead)."""
    rv=(monthly_ret.rolling(6).std()*np.sqrt(12)).shift(1)   # trailing realized ann. vol, lagged
    w=(target_vol/rv).clip(0,lev_cap).fillna(1.0)
    out=np.where(w>1, w*monthly_ret-(w-1)*BORROW_MO, w*monthly_ret+(1-w)*CASH_MO)
    return pd.Series(out,index=monthly_ret.index)

def eq_from_rets(rets): return CAPITAL*(1+rets).cumprod()

def metrics(eq,start):
    if eq.empty or len(eq)<2: return float("nan"),float("nan"),float("nan")
    yrs=max((eq.index[-1]-eq.index[0]).days/365.25,0.1)
    cagr=(eq.iloc[-1]/start)**(1/yrs)-1
    dd=((eq-eq.cummax())/eq.cummax()).min()
    r=eq.pct_change().dropna(); sh=(r.mean()/r.std()*np.sqrt(12)) if r.std()>0 else float("nan")
    return cagr,dd,sh

def is_oos(rets, label):
    eq=eq_from_rets(rets)
    is_e=eq[eq.index.year<=IS_END]; oos_e=eq[eq.index.year>IS_END]
    ic,idd,ish=metrics(is_e,CAPITAL)
    oc,odd,osh=metrics(oos_e, is_e.iloc[-1] if not is_e.empty else CAPITAL)
    fc,fdd,fsh=metrics(eq,CAPITAL)
    print(f"  {label:<42} IS {ic*100:>+5.1f}%/{ish:.2f}  | OOS {oc*100:>+5.1f}%/DD{odd*100:.0f}%/Sh{osh:.2f} | Full {fc*100:>+5.1f}%")
    return dict(label=label, oos_cagr=oc, oos_dd=odd, oos_sharpe=osh, full_cagr=fc)

def main():
    close_m,turn_m=build_panels(); nifty_m=load_nifty()
    print(f"universe {close_m.shape[1]} stocks | 2010-2026 | IS≤2018 / OOS≥2019\n")
    print(f"{'='*104}\n  IMPROVEMENT LADDER  (each row adds one research-backed change; IS tune-free, OOS untouched)\n{'='*104}")
    _,r_base,_,_ = simulate(close_m,turn_m,nifty_m)
    _,r_ra,_,_   = simulate(close_m,turn_m,nifty_m, rank_mode="riskadj")
    _,r_rab,_,_  = simulate(close_m,turn_m,nifty_m, rank_mode="riskadj", buffer=1.67)
    is_oos(r_base,"v2 base (raw mom, no buffer)")
    is_oos(r_ra,  "+ risk-adjusted ranking")
    is_oos(r_rab, "+ risk-adj + buffer (turnover cut)")
    print(f"  {'-'*100}")
    is_oos(vol_target(r_rab, 0.15, 1.0), "+ vol-target 15% (NO leverage) -> crash control")
    is_oos(vol_target(r_rab, 0.15, 1.5), "+ vol-target 15%, lev<=1.5x -> more return")
    is_oos(vol_target(r_rab, 0.15, 2.0), "+ vol-target 15%, lev<=2.0x -> aggressive")
    # Nifty benchmark
    ne=CAPITAL*(nifty_m[(nifty_m.index.year>=FROM_YEAR)].pct_change().fillna(0)+1).cumprod()
    nc,ndd,nsh=metrics(ne[ne.index.year>IS_END], ne[ne.index.year<=IS_END].iloc[-1])
    print(f"\n  Nifty 50 buy-&-hold (benchmark)            OOS {nc*100:+.1f}% / DD {ndd*100:.0f}% / Sharpe {nsh:.2f}")

if __name__=="__main__":
    main()
