"""
STRATEGY B — COMMODITY TREND-FOLLOWING (CTA-style, the real 'trading system').
Assets: gold, silver, crude, natgas, copper (COMEX futures = clean MCX proxy).
Signal: time-series momentum (trend) — long if uptrend, SHORT if downtrend.
Position: inverse-vol weighted, monthly. Long AND short. Honest futures costs.
Also: index trend-following (Nifty/BankNifty) for comparison.
IS 2008-2016 / OOS 2017-2026.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np, pandas as pd, yfinance as yf

CAP=200_000.0; IS_END=2016; RT=0.0010    # 0.10% round-trip per position change (futures are cheap)

def px(tk,start="2007-01-01"):
    d=yf.download(tk,start=start,auto_adjust=True,progress=False)["Close"].squeeze().dropna()
    d.index=pd.to_datetime(d.index); return d.resample("ME").last()

def ts_trend_signal(m, lb_fast=3, lb_slow=12, ma=10):
    """Time-series trend: +1 long if (price>MA and 12m ret>0), -1 short if opposite, else 0."""
    ret12=m/m.shift(lb_slow)-1; ret3=m/m.shift(lb_fast)-1
    maser=m.ewm(span=ma,adjust=False).mean()
    sig=pd.Series(0.0,index=m.index)
    up=(m>maser)&(ret12>0); dn=(m<maser)&(ret12<0)
    sig[up]=1.0; sig[dn]=-1.0
    return sig

def backtest_trend(assets, allow_short=True, vol_target=None):
    """assets: dict name->monthly price. Inverse-vol weighted trend portfolio."""
    idx=None
    for s in assets.values(): idx=s.index if idx is None else idx.union(s.index)
    px_={k:v.reindex(idx).ffill() for k,v in assets.items()}
    rets={k:v.pct_change() for k,v in px_.items()}
    vol={k:rets[k].rolling(6).std() for k in assets}
    sig={k:ts_trend_signal(px_[k]) for k in assets}
    names=list(assets); out=[]
    first=max(px_[k].first_valid_index() for k in names)
    start=max(13, list(idx).index(first)+13)
    prev_pos={k:0.0 for k in names}
    for p in range(start,len(idx)-1):
        iv={k:1.0/max(vol[k].iloc[p],1e-4) for k in names}; tot=sum(iv.values())
        pos={}
        for k in names:
            s=sig[k].iloc[p]
            if not allow_short and s<0: s=0.0
            pos[k]=(iv[k]/tot)*s
        # portfolio next-month return
        r=sum(pos[k]*rets[k].iloc[p+1] for k in names if pd.notna(rets[k].iloc[p+1]))
        # turnover cost
        to=sum(abs(pos[k]-prev_pos[k]) for k in names)
        out.append((idx[p+1], r-to*RT)); prev_pos=pos
    return pd.Series(dict(out)).sort_index()

def stats(r,label):
    r=r.dropna(); eq=CAP*(1+r).cumprod()
    ie=eq[eq.index.year<=IS_END]; oe=eq[eq.index.year>IS_END]
    def cd(e,s):
        if len(e)<2: return float('nan'),float('nan'),float('nan')
        yrs=max((e.index[-1]-e.index[0]).days/365.25,.1)
        return (e.iloc[-1]/s)**(1/yrs)-1,((e-e.cummax())/e.cummax()).min(),(e.pct_change().dropna().mean()/e.pct_change().dropna().std()*np.sqrt(12) if e.pct_change().dropna().std()>0 else float('nan'))
    ic,idd,ish=cd(ie,CAP); oc,odd,osh=cd(oe,ie.iloc[-1] if len(ie) else CAP); fc,fdd,fsh=cd(eq,CAP)
    print(f"  {label:<38} IS {ic*100:>+6.1f}%/Sh{ish:.2f} | OOS {oc*100:>+6.1f}%/DD{odd*100:.0f}%/Sh{osh:.2f} | Full {fc*100:>+6.1f}%/DD{fdd*100:.0f}%")

def main():
    print("Fetching commodity + index data ...", flush=True)
    comm={"gold":px("GC=F"),"silver":px("SI=F"),"crude":px("CL=F"),"natgas":px("NG=F"),"copper":px("HG=F")}
    nifty=px("^NSEI"); bank=px("^NSEBANK")
    print(f"\n{'='*104}\n  STRATEGY B — TREND-FOLLOWING (IS 2008-16 / OOS 2017-26, honest costs)\n{'='*104}")
    stats(backtest_trend(comm, allow_short=True),  "Commodity trend (5 assets, long+short)")
    stats(backtest_trend(comm, allow_short=False), "Commodity trend (long-only)")
    stats(backtest_trend({"gold":comm["gold"]}, allow_short=True), "Gold-only trend (long+short)")
    stats(backtest_trend({"crude":comm["crude"]}, allow_short=True), "Crude-only trend (long+short)")
    print()
    stats(backtest_trend({"nifty":nifty}, allow_short=True),  "Nifty trend (long+short)")
    stats(backtest_trend({"nifty":nifty}, allow_short=False), "Nifty trend (long-only, i.e. market timing)")
    stats(backtest_trend({"nifty":nifty,"bank":bank}, allow_short=True), "Nifty+BankNifty trend (long+short)")
    # buy-hold benchmarks
    print()
    for nm,m in [("Nifty buy-hold",nifty),("Gold buy-hold",comm["gold"])]:
        stats(m.pct_change(), nm)

if __name__=="__main__": main()
