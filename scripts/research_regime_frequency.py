"""
RESEARCH: how often should the REGIME (cash-vs-invested) decision be checked?
Separate from stock rebalancing (monthly). Tests the timing overlay: hold the
market when regime-ON, cash when OFF — regime re-checked MONTHLY vs WEEKLY vs DAILY.
Isolates the responsiveness-vs-whipsaw tradeoff. Daily Nifty 50, 2010-2026.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np, pandas as pd, duckdb

DB="data_store/piedpiper.duckdb"; CAP=200_000.0
def load_daily_nifty():
    c=duckdb.connect(DB,read_only=True)
    n=c.execute("SELECT dt,close FROM adjusted_ohlcv WHERE symbol='Nifty 50' ORDER BY dt").df();c.close()
    n["dt"]=pd.to_datetime(n["dt"]); return n.set_index("dt")["close"]

def regime_series(px, freq):
    """Regime signal: Nifty > 200d-MA AND 6mo(126d) return > -2%. Evaluated at freq,
    then held constant until next check (realistic — you only act at check points)."""
    ma200=px.rolling(200).mean()
    r6=px/px.shift(126)-1
    raw=((px>ma200)&(r6>-0.02))         # daily raw signal
    if freq=="daily":
        sig=raw
    else:
        rule={"monthly":"ME","weekly":"W-FRI"}[freq]
        checkpoints=raw.resample(rule).last()        # signal AS OF each check date
        sig=checkpoints.reindex(px.index, method="ffill")  # hold until next check
    return sig.shift(1).fillna(False)   # act next day (no look-ahead)

def backtest(px, freq):
    ret=px.pct_change().fillna(0); sig=regime_series(px,freq)
    cash_d=(1.065)**(1/252)-1; SW=0.0005   # switch cost
    pos=sig.astype(float)
    strat=np.where(sig, ret, cash_d)
    switches=int((pos.diff().abs()>0).sum())
    strat=pd.Series(strat,index=px.index) - (pos.diff().abs().fillna(0))*SW
    strat=strat[strat.index.year>=2010]
    eq=CAP*(1+strat).cumprod(); yrs=(eq.index[-1]-eq.index[0]).days/365.25
    cagr=(eq.iloc[-1]/CAP)**(1/yrs)-1; dd=((eq-eq.cummax())/eq.cummax()).min()
    sh=strat.mean()/strat.std()*np.sqrt(252)
    return cagr,dd,sh,switches

def main():
    px=load_daily_nifty()
    print(f"{'='*84}\n  REGIME-CHECK FREQUENCY — hold Nifty when ON / cash when OFF (2010-2026)\n{'='*84}")
    print(f"  {'regime checked':<16}{'CAGR':>8}{'MaxDD':>8}{'Sharpe':>8}{'switches':>10}{'(whipsaw)':>10}")
    for f in ["monthly","weekly","daily"]:
        c,d,s,sw=backtest(px,f)
        print(f"  {f:<16}{c*100:>+7.1f}%{d*100:>7.0f}%{s:>8.2f}{sw:>10}")
    # buy-hold reference
    bh=px[px.index.year>=2010]; eq=CAP*(bh/bh.iloc[0]); yrs=(eq.index[-1]-eq.index[0]).days/365.25
    print(f"  {'buy&hold (no regime)':<16}{((eq.iloc[-1]/CAP)**(1/yrs)-1)*100:>+7.1f}%{((eq-eq.cummax())/eq.cummax()).min()*100:>7.0f}%")

if __name__=="__main__": main()
