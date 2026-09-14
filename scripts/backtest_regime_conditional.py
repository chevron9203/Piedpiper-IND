"""
Regime-CONDITIONAL allocation — fix the 'only matches Nifty in bulls' problem.
Risk-ON (Nifty+breadth healthy = a bull): go HEAVY momentum (capture the rally).
Risk-OFF (regime broken = a bear): go defensive (gold/US/cash).
Compares vs Nifty and vs the static adaptive risk-parity, on bears & bulls.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np, pandas as pd, yfinance as yf, duckdb

CAP=200_000.0; TOP_N=15; TURN_TOP=500; PRICE_MIN=30.0; RT=2*(0.0010+0.0010)+0.0005; CASH=(1.065)**(1/12)-1

def load():
    pk=pd.read_parquet("data_store/mom_v2_panels.parquet"); close=pk.xs("close",1,0); turn=pk.xs("turn",1,0)
    c=duckdb.connect("data_store/piedpiper.duckdb",read_only=True)
    n=c.execute("SELECT dt,close FROM adjusted_ohlcv WHERE symbol='Nifty 50' ORDER BY dt").df();c.close()
    n["dt"]=pd.to_datetime(n["dt"]); nifty=n.set_index("dt")["close"].resample("ME").last()
    def mpx(t): d=yf.download(t,start="2010-06-01",auto_adjust=True,progress=False); s=d["Close"].squeeze().dropna(); s.index=pd.to_datetime(s.index); return s.resample("ME").last()
    return close,turn,nifty,mpx("GOLDBEES.NS"),mpx("MON100.NS")

def components(close,turn,nifty,gold,us):
    idx=close.index; nser=nifty.reindex(idx).ffill(); nema=nser.ewm(span=10,adjust=False).mean()
    ret=close.pct_change(); vol=ret.rolling(6).std()
    m3=close.shift(1)/close.shift(4)-1; m6=close.shift(1)/close.shift(7)-1; m12=close.shift(1)/close.shift(13)-1
    ma10=close.rolling(10).mean(); above=(close>ma10)
    gp=gold.reindex(idx).ffill(); up=us.reindex(idx).ffill()
    gret=gp.pct_change(); uret=up.pct_change(); gema=gp.ewm(span=10,adjust=False).mean(); uema=up.ewm(span=10,adjust=False).mean()
    prev=set(); rows=[]; start=int(np.searchsorted(idx.year.values,2011))
    for p in range(start,len(idx)-1):
        pr=close.iloc[p]; tn=turn.iloc[p]; base=pr.index[(pr>=PRICE_MIN)&tn.notna()]
        liq=tn[base].nlargest(min(TURN_TOP,len(base))).index; breadth=above.iloc[p][liq].mean() if len(liq) else .5
        risk_on=bool((nser.iloc[p]>nema.iloc[p]) and (nser.iloc[p]/nser.iloc[p-6]-1>-0.02) and breadth>0.45)
        mom=CASH
        if risk_on:
            elig=[s for s in liq if pr[s]>ma10.iloc[p].get(s,1e9)]; sc=pd.Series(0.0,index=elig); nc=0
            for sg in (m3,m6,m12):
                x=(sg.iloc[p][elig]/vol.iloc[p][elig].replace(0,np.nan)).dropna()
                if len(x): sc=sc.add(x.rank(pct=True),fill_value=0); nc+=1
            if nc:
                rk=(sc/nc).sort_values(ascending=False); rank={s:i for i,s in enumerate(rk.index)}
                picks=[s for s in prev if rank.get(s,1e9)<TOP_N*1.67]
                for s in rk.index:
                    if len(picks)>=TOP_N: break
                    if s not in picks: picks.append(s)
                picks=set(picks[:TOP_N])
                rr=[close.iloc[p+1][s]/close.iloc[p][s]-1 for s in picks if pd.notna(close.iloc[p+1].get(s)) and pd.notna(close.iloc[p][s])]
                to=len(picks.symmetric_difference(prev))/max(len(picks|prev),1); mom=(float(np.mean(rr)) if rr else CASH)-to*RT; prev=picks
        else: prev=set()
        rows.append((idx[p+1], mom, gret.iloc[p+1] if pd.notna(gret.iloc[p+1]) else CASH,
                     uret.iloc[p+1] if pd.notna(uret.iloc[p+1]) else CASH, risk_on,
                     bool(gp.iloc[p]>gema.iloc[p]), bool(up.iloc[p]>uema.iloc[p])))
    return pd.DataFrame(rows,columns=["dt","mom","gold","us","risk_on","gold_on","us_on"]).set_index("dt")

def adaptive_rp(D):
    vm=D["mom"].rolling(6).std().shift(1); vg=D["gold"].rolling(6).std().shift(1); vu=D["us"].rolling(6).std().shift(1); tot=1/vm+1/vg+1/vu
    return (((1/vm)/tot)*D["mom"]+((1/vg)/tot)*D["gold"]+((1/vu)/tot)*D["us"]).fillna(D["mom"])

def regime_conditional(D, wmom=0.70):
    out=[]
    for t,r in D.iterrows():
        g=r.gold if r.gold_on else CASH; u=r.us if r.us_on else CASH
        if r.risk_on:
            rest=(1-wmom)/2; out.append((t, wmom*r.mom + rest*g + rest*u))
        else:
            out.append((t, 0.45*g + 0.35*u + 0.20*CASH))
    return pd.Series(dict(out))

def stats(r): eq=CAP*(1+r).cumprod(); yrs=(eq.index[-1]-eq.index[0]).days/365.25; return (eq.iloc[-1]/CAP)**(1/yrs)-1,((eq-eq.cummax())/eq.cummax()).min(),r.mean()/r.std()*np.sqrt(12)

def main():
    close,turn,nifty,gold,us=load(); D=components(close,turn,nifty,gold,us)
    nret=nifty.reindex(D.index).pct_change().fillna(0)
    cfgs={"NIFTY":nret,"Adaptive RP (static)":adaptive_rp(D),
          "Regime-cond 70% mom":regime_conditional(D,0.70),"Regime-cond 80% mom":regime_conditional(D,0.80)}
    print("="*86); print("  REGIME-CONDITIONAL (heavy momentum in bulls) — fixing bull capture"); print("="*86)
    print(f"  {'config':<24}{'CAGR':>8}{'MaxDD':>7}{'Sharpe':>8}")
    for k,r in cfgs.items(): c,d,s=stats(r); print(f"  {k:<24}{c*100:>+7.1f}%{d*100:>6.0f}%{s:>8.2f}")
    def win(r,a,b): e=CAP*(1+r).cumprod(); x=e[(e.index>=a)&(e.index<=b)]; return (x.iloc[-1]/x.iloc[0]-1) if len(x)>1 else float('nan')
    ev=[("2011 bear","🐻","2011-01-31","2011-12-31"),("2013-14 bull","🐂","2013-08-31","2014-12-31"),
        ("2015-16 bear","🐻","2015-03-31","2016-02-29"),("2017 bull","🐂","2017-01-31","2017-12-31"),
        ("2020 COVID","🐻","2020-01-31","2020-03-31"),("2020-21 bull","🐂","2020-04-30","2021-12-31"),
        ("2022 correction","🐻","2022-01-31","2022-06-30"),("2023-24 bull","🐂","2023-03-31","2024-09-30")]
    print(f"\n  {'PERIOD':<18}{'':>3}{'NIFTY':>8}{'staticRP':>10}{'regime70':>10}{'regime80':>10}")
    for nm,ty,a,b in ev:
        print(f"  {nm:<18}{ty:>3}{win(nret,a,b)*100:>+7.0f}%{win(cfgs['Adaptive RP (static)'],a,b)*100:>+9.0f}%"
              f"{win(cfgs['Regime-cond 70% mom'],a,b)*100:>+9.0f}%{win(cfgs['Regime-cond 80% mom'],a,b)*100:>+9.0f}%")

if __name__=="__main__": main()
