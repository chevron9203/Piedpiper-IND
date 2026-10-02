"""
Tax-aware momentum — position-level sim with real Indian capital-gains tax.
STCG 20% (held <12mo, from rupee 1); LTCG 12.5% (held >=12mo, >Rs1.25L/yr exempt).
Compares: standard momentum (pre-tax vs post-tax) vs TAX-AWARE (defer winner sales
to cross 12mo for LTCG) post-tax. Momentum-equity sleeve, always invested.
Also tests RESIDUAL (beta-neutral) momentum as an alternate signal.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np, pandas as pd
from scripts.backtest_momentum_v4 import load, precompute

CAP=200_000.0; TOP_N=15; TURN_TOP=500; PRICE_MIN=30.0
COST=0.0020            # per side (slippage+STT)
STCG=0.20; LTCG=0.125; LTCG_EXEMPT=125_000.0
FROM_YEAR=2010; IS_END=2018

def sim(close,turn,nifty, tax_aware=False, apply_tax=True, residual=False, defer_from=0):
    idx=close.index
    ret=close.pct_change(); vol=ret.rolling(6).std()
    m3=close.shift(1)/close.shift(4)-1; m6=close.shift(1)/close.shift(7)-1; m12=close.shift(1)/close.shift(13)-1
    ma10=close.rolling(10).mean()
    # residual (beta-neutral) momentum: subtract beta * nifty momentum
    if residual:
        nser=nifty.reindex(idx).ffill(); nret=nser.pct_change()
        beta=ret.rolling(36).cov(nret).div(nret.rolling(36).var(),axis=0)
        nmom=(nser.shift(1)/nser.shift(13)-1)
        m12=m12.sub(beta.mul(nmom,axis=0))       # residualize the 12mo signal
    pos={}      # sym -> [entry_p, cost_basis, cur_val]
    cash=CAP; curve=[]; stcg=0.0; ltcg=0.0; last_yr=None
    sold_holds=[]; ltcg_g=0.0; stcg_g=0.0
    start=int(np.searchsorted(idx.year.values,FROM_YEAR))
    for p in range(start,len(idx)-1):
        yr=idx[p].year
        if apply_tax and last_yr is not None and yr!=last_yr:
            tax=STCG*max(0,stcg)+LTCG*max(0,ltcg-LTCG_EXEMPT); cash-=tax; stcg=0;ltcg=0
        last_yr=yr
        price=close.iloc[p]; tn=turn.iloc[p]; hist=close.iloc[:p+1].notna().sum()
        base=price.index[(price>=PRICE_MIN)&(hist>=14)&tn.notna()]
        elig=[s for s in tn[base].nlargest(min(TURN_TOP,len(base))).index
              if pd.notna(ma10.iloc[p].get(s)) and price[s]>ma10.iloc[p][s]]
        score=pd.Series(0.0,index=elig); nc=0
        for sg in (m3,m6,m12):
            s=(sg.iloc[p][elig]/vol.iloc[p][elig].replace(0,np.nan)).dropna()
            if len(s): score=score.add(s.rank(pct=True),fill_value=0); nc+=1
        if nc==0: curve.append((idx[p+1],cash+sum(v[2] for v in pos.values()))); continue
        ranked=(score/nc).sort_values(ascending=False); rk={s:i for i,s in enumerate(ranked.index)}
        target=set(ranked.head(TOP_N).index)
        # keep incumbents inside buffer
        for s in list(pos):
            if rk.get(s,10**9)<TOP_N*1.67: target.add(s)
        target=set(list(target)[:max(TOP_N,len(target))])
        # SELLS
        for s in list(pos):
            if s in target and rk.get(s,10**9)<TOP_N*1.67: continue
            held=p-pos[s][0]
            still_trending=pd.notna(price.get(s)) and pd.notna(ma10.iloc[p].get(s)) and price[s]>ma10.iloc[p][s]
            if tax_aware and defer_from<=held<12 and still_trending: continue   # defer near-12mo winners for LTCG
            g=pos[s][2]-pos[s][1]
            if held>=12: ltcg+=g; ltcg_g+=max(g,0)
            else: stcg+=g; stcg_g+=max(g,0)
            sold_holds.append(held); cash+=pos[s][2]*(1-COST); del pos[s]
        # BUYS to equal weight
        equity=cash+sum(v[2] for v in pos.values())
        tgt=equity/TOP_N
        for s in ranked.index:
            if len(pos)>=TOP_N: break
            if s in pos: continue
            buy=min(cash, tgt)
            if buy<1000: continue
            cash-=buy*(1+COST); pos[s]=[p,buy,buy]
        # mark to market to next month
        for s in list(pos):
            r=close.iloc[p+1].get(s); r0=close.iloc[p].get(s)
            if pd.notna(r) and pd.notna(r0): pos[s][2]*=(r/r0)
        curve.append((idx[p+1],cash+sum(v[2] for v in pos.values())))
    eq=pd.Series(dict(curve)).sort_index()
    avg_hold=np.mean(sold_holds) if sold_holds else 0
    ltcg_pct=ltcg_g/(ltcg_g+stcg_g)*100 if (ltcg_g+stcg_g)>0 else 0
    return eq, avg_hold, ltcg_pct

def metrics(eq,start):
    yrs=max((eq.index[-1]-eq.index[0]).days/365.25,0.1)
    return (eq.iloc[-1]/start)**(1/yrs)-1, ((eq-eq.cummax())/eq.cummax()).min()

def show(eq,label,avg_hold=None,ltcg=None):
    ie=eq[eq.index.year<=IS_END]; oe=eq[eq.index.year>IS_END]
    ic,_=metrics(ie,CAP); oc,odd=metrics(oe,ie.iloc[-1]); fc,fdd=metrics(eq,CAP)
    x=f" | avg-hold {avg_hold:.1f}mo, LTCG {ltcg:.0f}% of gains" if avg_hold is not None else ""
    print(f"  {label:<40} IS {ic*100:>+5.1f}% | OOS {oc*100:>+5.1f}%/DD{odd*100:.0f}% | Full {fc*100:>+5.1f}%{x}")

def main():
    close,turn,nifty=load()
    print(f"{'='*104}\n  TAX-AWARE MOMENTUM  (STCG 20% / LTCG 12.5%, Rs1.25L exempt)  IS 2010-18 / OOS 2019-26\n{'='*104}")
    e0,h0,l0=sim(close,turn,nifty, tax_aware=False, apply_tax=False)
    show(e0,"standard momentum, PRE-tax",h0,l0)
    e1,h1,l1=sim(close,turn,nifty, tax_aware=False, apply_tax=True)
    show(e1,"standard momentum, POST-tax",h1,l1)
    e2,h2,l2=sim(close,turn,nifty, tax_aware=True, apply_tax=True, defer_from=0)
    show(e2,"TAX-AWARE (defer all winners), POST-tax",h2,l2)
    e2b,h2b,l2b=sim(close,turn,nifty, tax_aware=True, apply_tax=True, defer_from=9)
    show(e2b,"TAX-AWARE (defer only 9-11mo), POST-tax",h2b,l2b)
    e2c,h2c,l2c=sim(close,turn,nifty, tax_aware=True, apply_tax=True, defer_from=10)
    show(e2c,"TAX-AWARE (defer only 10-11mo), POST-tax",h2c,l2c)
    print(f"\n{'='*104}\n  RESIDUAL (beta-neutral) MOMENTUM — does it lower drawdown?\n{'='*104}")
    e3,h3,l3=sim(close,turn,nifty, tax_aware=False, apply_tax=False, residual=True)
    show(e3,"residual momentum, PRE-tax",h3,l3)
    e4,h4,l4=sim(close,turn,nifty, tax_aware=True, apply_tax=True, residual=True)
    show(e4,"residual + TAX-AWARE, POST-tax",h4,l4)

if __name__=="__main__": main()
