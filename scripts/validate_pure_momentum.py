"""
VALIDATE the pure-momentum logic before deploying System 4 & 5.
Tests BOTH variants honestly IS 2010-18 / OOS 2019-26 on our real data:
  System 4 = pure MID-cap momentum + regime CASH defense
  System 5 = pure MID-cap momentum, ALWAYS invested (no cash-out)
Confirms: config sound, IS≈OOS (not overfit), realistic returns, sensible picks.
Honest costs (0.10%/side), point-in-time universe, trend confirm, rank buffer.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np, pandas as pd, duckdb

CACHE="data_store/mom_v2_panels.parquet"; DB="data_store/piedpiper.duckdb"
CAP=200_000.0; PRICE_MIN=30.0; FROM_YEAR=2010; IS_END=2018
TOP_N=15; TURN_TOP=500; RT=2*(0.0010+0.0010)+0.0005; CASH_MO=(1.065)**(1/12)-1

def load():
    pk=pd.read_parquet(CACHE); close=pk.xs("close",1,0); turn=pk.xs("turn",1,0)
    c=duckdb.connect(DB,read_only=True)
    n=c.execute("SELECT dt,close FROM adjusted_ohlcv WHERE symbol='Nifty 50' ORDER BY dt").df();c.close()
    n["dt"]=pd.to_datetime(n["dt"]); nifty=n.set_index("dt")["close"].resample("ME").last()
    return close,turn,nifty

VOL_FLOOR=0.02   # exclude near-zero-vol instruments (cash/liquid/gilt/bond ETFs, NOT stocks)
ETF_PAT=("BEES","LIQUID","GILT","GSEC","CASH","ETF","NIFTY","SENSEX","SILVER","GOLDCASE","BHARATBOND")
def is_stock(sym):  # name-based ETF/fund exclusion (belt-and-suspenders)
    s=sym.upper()
    return not (s.endswith("BEES") or s.endswith("ETF") or s.endswith("IETF")
                or any(p in s for p in ("LIQUID","GILT","GSEC","BHARATBOND","CASHIETF")))

def pick_stocks(close, turn, p, prev, m3,m6,m12,vol,ma10):
    """The exact live stock-picking logic: MID-cap tier, multi-TF risk-adj mom,
    trend confirm, top-15 with rank buffer. Excludes non-stock ETFs/funds. """
    pr=close.iloc[p]; tn=turn.iloc[p]; hist=close.iloc[:p+1].notna().sum(); v=vol.iloc[p]
    base=pr.index[(pr>=PRICE_MIN)&(hist>=14)&tn.notna()&(tn>0)&(v>=VOL_FLOOR)]  # vol floor removes cash-like
    base=[s for s in base if is_stock(s)]                     # name filter removes equity ETFs
    if len(base)<100: return []
    rank=tn[base].rank(pct=True)
    mid=list(rank.index[(rank>0.60)&(rank<=0.90)])            # MID-cap tier
    elig=[s for s in mid if pr[s]>ma10.iloc[p].get(s,1e9)]     # trend confirm
    score=pd.Series(0.0,index=elig); nc=0
    for sg in (m3,m6,m12):
        x=(sg.iloc[p][elig]/vol.iloc[p][elig].replace(0,np.nan)).dropna()
        if len(x): score=score.add(x.rank(pct=True),fill_value=0); nc+=1
    if nc==0: return []
    ranked=(score/nc).sort_values(ascending=False); rk={s:i for i,s in enumerate(ranked.index)}
    picks=[s for s in prev if rk.get(s,10**9)<TOP_N*1.67]      # rank buffer
    for s in ranked.index:
        if len(picks)>=TOP_N: break
        if s not in picks: picks.append(s)
    return picks[:TOP_N]

def run(close,turn,nifty, always_invested):
    idx=close.index; nser=nifty.reindex(idx).ffill(); nema=nser.ewm(span=10,adjust=False).mean()
    ret=close.pct_change(); vol=ret.rolling(6).std()
    m3=close.shift(1)/close.shift(4)-1; m6=close.shift(1)/close.shift(7)-1; m12=close.shift(1)/close.shift(13)-1
    ma10=close.rolling(10).mean(); above=(close>ma10)
    prev=[]; rets=[]; start=max(14,int(np.searchsorted(idx.year.values,FROM_YEAR)))
    for p in range(start,len(idx)-1):
        pr=close.iloc[p]; tn=turn.iloc[p]
        libase=pr.index[(pr>=PRICE_MIN)&tn.notna()]
        liq=tn[libase].nlargest(min(TURN_TOP,len(libase))).index
        breadth=above.iloc[p][liq].mean() if len(liq) else 0.5
        risk_on=(nser.iloc[p]>nema.iloc[p]) and (nser.iloc[p]/nser.iloc[p-6]-1>-0.02) and breadth>0.45
        picks=pick_stocks(close,turn,p,prev,m3,m6,m12,vol,ma10)
        hold = (always_invested or risk_on) and len(picks)>0
        if hold:
            r=[close.iloc[p+1][s]/close.iloc[p][s]-1 for s in picks if pd.notna(close.iloc[p+1].get(s)) and pd.notna(close.iloc[p][s])]
            gross=float(np.mean(r)) if r else CASH_MO
            to=len(set(picks).symmetric_difference(set(prev)))/max(len(set(picks)|set(prev)),1)
            rets.append((idx[p+1],gross-to*RT)); prev=picks
        else:
            rets.append((idx[p+1],CASH_MO)); prev=[]
    return pd.Series(dict(rets)).sort_index()

def stats(r,label):
    r=r.dropna(); eq=CAP*(1+r).cumprod(); ie=eq[eq.index.year<=IS_END]; oe=eq[eq.index.year>IS_END]
    def cd(e,s):
        y=max((e.index[-1]-e.index[0]).days/365.25,.1)
        return (e.iloc[-1]/s)**(1/y)-1,((e-e.cummax())/e.cummax()).min(),(e.pct_change().dropna().mean()/e.pct_change().dropna().std()*np.sqrt(12) if e.pct_change().dropna().std()>0 else float('nan'))
    ic,idd,ish=cd(ie,CAP); oc,odd,osh=cd(oe,ie.iloc[-1]); fc,fdd,fsh=cd(eq,CAP)
    print(f"  {label:<40} IS {ic*100:>+6.1f}%/Sh{ish:.2f} | OOS {oc*100:>+6.1f}%/DD{odd*100:.0f}%/Sh{osh:.2f} | Full {fc*100:>+6.1f}%/DD{fdd*100:.0f}%")

def main():
    close,turn,nifty=load()
    print(f"{'='*104}\n  PURE MOMENTUM VALIDATION (MID-cap, IS 2010-18/OOS 2019-26) — the logic for Systems 4 & 5\n{'='*104}")
    stats(run(close,turn,nifty, always_invested=False), "System 4: pure momentum + CASH defense")
    stats(run(close,turn,nifty, always_invested=True),  "System 5: pure momentum, ALWAYS invested")
    stats(nifty.reindex(close.index).pct_change().dropna()[lambda s:s.index.year>=FROM_YEAR], "Nifty 50 (benchmark)")
    # show CURRENT picks the live system would make
    print(f"\n{'='*104}\n  CURRENT PICKS (what System 5 holds RIGHT NOW) — eyeball for sanity\n{'='*104}")
    idx=close.index; ret=close.pct_change(); vol=ret.rolling(6).std()
    m3=close.shift(1)/close.shift(4)-1; m6=close.shift(1)/close.shift(7)-1; m12=close.shift(1)/close.shift(13)-1
    ma10=close.rolling(10).mean()
    p=len(idx)-1
    picks=pick_stocks(close,turn,p,[],m3,m6,m12,vol,ma10)
    print(f"  as-of {idx[p].date()} | universe screened: {int((close.iloc[p].notna()&(close.iloc[p]>=PRICE_MIN)).sum())} stocks")
    print(f"  top-15 MID-cap momentum picks: {picks}")

if __name__=="__main__": main()
