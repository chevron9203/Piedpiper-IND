"""
Multi-asset risk-parity book — the Sharpe lever.
Assets: momentum-equity ALPHA (our v4 book, pure) + gold + US-Nasdaq(MON100) + cash.
Each risky asset trend-filtered (hold if > own 10mo EMA, else its weight -> cash).
Weights = inverse-vol (risk parity) computed across all assets; off-trend assets
park in cash. Monthly. Then optional fractional-Kelly leverage overlay.
Period 2011-2026 (MON100 start). IS 2011-18 / OOS 2019-26.  vs momentum-alone.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np, pandas as pd, yfinance as yf
from scripts.backtest_momentum_v4 import load as mom_load, precompute, simulate as mom_sim

CAP=200_000.0; CASH_YR=0.06; BORROW_YR=0.09; IS_END=2018

def get_momentum_equity():
    close,turn,nifty=mom_load(); sig,vol,ma10=precompute(close)
    # pure momentum-equity alpha: MAX config, ALWAYS invested (no regime, no gold)
    r=mom_sim(close,turn,nifty,sig,vol,ma10, signals=("m3","m6","m12"), trend_confirm=True,
              vol_cap=0.0, buffer=1.67, regime=False)
    return r  # monthly returns

def px_monthly(ticker,start="2010-06-01"):
    d=yf.download(ticker,start=start,auto_adjust=True,progress=False)
    s=d["Close"].squeeze().dropna(); s.index=pd.to_datetime(s.index)
    return s.resample("ME").last()

def load_vix():
    import duckdb
    c=duckdb.connect("data_store/piedpiper.duckdb",read_only=True)
    v=c.execute("SELECT dt,close FROM adjusted_ohlcv WHERE symbol='India VIX' ORDER BY dt").df();c.close()
    v["dt"]=pd.to_datetime(v["dt"]); return v.set_index("dt")["close"].resample("ME").last()

def combine(assets_px, lev=1.0, mom_tilt=1.0, vix_m=None):
    """Risk-parity + per-asset trend, off->cash. mom_tilt: over-weight the momentum
    alpha (>1). vix_m: scale down risky exposure when India VIX is elevated."""
    idx=None
    for s in assets_px.values(): idx=s.index if idx is None else idx.union(s.index)
    px={k:v.reindex(idx).ffill() for k,v in assets_px.items()}
    rets={k:v.pct_change() for k,v in px.items()}
    ema={k:v.ewm(span=10,adjust=False).mean() for k,v in px.items()}
    vol={k:rets[k].rolling(6).std() for k,v in px.items()}
    vix=vix_m.reindex(idx).ffill() if vix_m is not None else None
    cash_mo=(1+CASH_YR)**(1/12)-1; borrow_mo=(1+BORROW_YR)**(1/12)-1
    names=list(assets_px); out=[]
    first=max(px[k].first_valid_index() for k in names)
    start=max(13, list(idx).index(first)+13)
    for p in range(start,len(idx)-1):
        inv={k:1.0/max(vol[k].iloc[p],1e-4) for k in names}
        if "mom" in inv: inv["mom"]*=mom_tilt                  # alpha tilt
        tot=sum(inv.values()); base={k:inv[k]/tot for k in names}
        w={}; cash=0.0
        for k in names:
            on = px[k].iloc[p] > ema[k].iloc[p]
            if on and pd.notna(rets[k].iloc[p+1]): w[k]=base[k]
            else: cash+=base[k]
        if vix is not None and pd.notna(vix.iloc[p]):           # VIX de-risking
            scale=float(np.clip(18.0/vix.iloc[p],0.5,1.0)); risky=sum(w.values())
            if risky>0:
                for k in w: w[k]*=scale
                cash+=risky*(1-scale)
        r=sum(w[k]*rets[k].iloc[p+1] for k in w) + cash*cash_mo
        if lev!=1.0: r = lev*r - (lev-1)*borrow_mo
        out.append((idx[p+1], r))
    return pd.Series(dict(out)).sort_index()

def metrics(eq,start):
    yrs=max((eq.index[-1]-eq.index[0]).days/365.25,0.1)
    cagr=(eq.iloc[-1]/start)**(1/yrs)-1; dd=((eq-eq.cummax())/eq.cummax()).min()
    r=eq.pct_change().dropna(); sh=(r.mean()/r.std()*np.sqrt(12)) if r.std()>0 else float("nan")
    return cagr,dd,sh

def show(rets,label):
    rets=rets[rets.index.year>=2011]
    eq=CAP*(1+rets).cumprod()
    ie=eq[eq.index.year<=IS_END]; oe=eq[eq.index.year>IS_END]
    ic,idd,ish=metrics(ie,CAP); oc,odd,osh=metrics(oe,ie.iloc[-1]); fc,fdd,fsh=metrics(eq,CAP)
    print(f"  {label:<44} IS {ic*100:>+5.1f}%/Sh{ish:.2f} | OOS {oc*100:>+5.1f}%/DD{odd*100:.0f}%/Sh{osh:.2f} | Full {fc*100:>+5.1f}%/DD{fdd*100:.0f}%")

def main():
    print("Fetching momentum-equity alpha + assets ...", flush=True)
    mom_r=get_momentum_equity()
    mom_px=CAP*(1+mom_r).cumprod()                      # synthetic price index for the alpha sleeve
    gold=px_monthly("GOLDBEES.NS"); us=px_monthly("MON100.NS")
    print(f"  mom_eq {mom_px.index[0].date()}->{mom_px.index[-1].date()} | gold {len(gold)}mo | us {len(us)}mo\n")
    print("="*112)
    print("  MULTI-ASSET RISK-PARITY BOOK  (momentum-equity + gold + US-Nasdaq + cash)  IS 2011-18 / OOS 2019-26")
    print("="*112)
    vix=load_vix()
    A={"mom":mom_px,"gold":gold,"us":us}
    show(mom_r,                                              "momentum-equity ALONE")
    show(combine(A),                                         "core: risk-parity (tilt=1)")
    print("  "+"-"*108+"\n  REFINEMENT 1 — alpha tilt (over-weight our momentum edge):")
    show(combine(A, mom_tilt=2.0),                           "  tilt=2 (mom 2x weight)")
    show(combine(A, mom_tilt=3.0),                           "  tilt=3 (mom 3x weight)")
    print("  REFINEMENT 2 — VIX-based de-risking:")
    show(combine(A, vix_m=vix),                              "  core + VIX sizing")
    show(combine(A, mom_tilt=2.0, vix_m=vix),               "  tilt=2 + VIX  ← candidate")
    print("  "+"-"*108+"\n  best candidate + modest leverage:")
    for L in (1.3,1.5):
        show(combine(A, mom_tilt=2.0, vix_m=vix, lev=L),    f"  tilt=2 + VIX × {L:.1f}")

if __name__=="__main__": main()
