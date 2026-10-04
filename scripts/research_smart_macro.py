"""
Macro / regime features for the smart system (date-level, identical for every stock that day; they let the trees
switch factor weights by regime - the market-state features already carry ~25% of the model's weight and removing
them cost 0.014 IC).

  m_vix, m_vix_chg5, m_vix_pct   India VIX level, 5d change, percentile within its trailing year
  m_usdinr21/63                  rupee: 21/63-trading-day change of USD/INR
  m_brent21/63                   Brent crude change
  m_spx21/63                     S&P 500 change
  m_us10y63                      US 10y yield change (points)
  m_dxy63, m_gold63              dollar index, gold

POINT-IN-TIME: at the NSE close of day t the last known US/global print is the one dated <= t-1 (the US session of
t closes after NSE does), so every non-Indian series is lagged one calendar day. India VIX is known at the same close.
Run:  python scripts/research_smart_macro.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(__file__).parent.parent
OUT = BASE/"data_store/smart_macro.parquet"
SERIES = {"^INDIAVIX": ("vix", 0), "USDINR=X": ("usdinr", 1), "BZ=F": ("brent", 1), "^GSPC": ("spx", 1),
          "^TNX": ("us10y", 1), "DX-Y.NYB": ("dxy", 1), "GC=F": ("gold", 1)}


def build():
    import yfinance as yf
    idx = pd.read_parquet(BASE/"data_store/nse_panel.parquet", columns=[("close", "RELIANCE")]).index if False else None
    sys.path.insert(0, str(BASE))
    from scripts import research_smart_exit as E
    idx = E.inputs()["C"].index                                  # NSE trading calendar
    cols = {}
    for tk, (nm, lag) in SERIES.items():
        d = yf.download(tk, start="2007-01-01", progress=False, auto_adjust=False)["Close"]
        d = d.squeeze().dropna(); d.index = pd.DatetimeIndex(d.index).tz_localize(None).normalize()
        if lag:
            d.index = d.index + pd.Timedelta(days=lag)           # value usable from the next calendar day
        cols[nm] = d[~d.index.duplicated(keep="last")].reindex(idx, method="ffill")
    S = pd.DataFrame(cols, index=idx)
    M = pd.DataFrame(index=idx)
    M["m_vix"] = S["vix"]
    M["m_vix_chg5"] = S["vix"]/S["vix"].shift(5) - 1
    M["m_vix_pct"] = S["vix"].rolling(252, min_periods=120).apply(lambda x: (x[-1] >= x).mean(), raw=True)
    for nm, k in (("usdinr", 21), ("usdinr", 63), ("brent", 21), ("brent", 63), ("spx", 21), ("spx", 63), ("dxy", 63), ("gold", 63)):
        M[f"m_{nm}{k}"] = S[nm]/S[nm].shift(k) - 1
    M["m_us10y63"] = S["us10y"] - S["us10y"].shift(63)
    M.index.name = "date"
    M.astype("float32").to_parquet(OUT)
    return M


if __name__ == "__main__":
    M = build()
    print(M.shape, M.index.min().date(), M.index.max().date())
    print("non-null share by column:", {c: f"{M[c].notna().mean():.0%}" for c in M.columns})
    print(M.loc["2020-03-23"].round(3).to_dict())
    print(M.tail(2).round(3).T.to_string())
