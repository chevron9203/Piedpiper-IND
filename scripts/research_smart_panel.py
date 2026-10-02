"""
Full daily OHLCV + delivery panel for the "smart system" research.

The cached daily_panel_full.parquet only carries close + turnover. The research needs
open/high/low (gaps, ranges, close-location), volume, delivery qty and trade counts
(institutional footprint), so this builds a richer panel from the eod2 CSVs.

Point-in-time: nothing here looks forward -- it is raw data, one column per symbol.
Survivorship caveat (unchanged, data limit): stocks fully delisted are absent from eod2.

Run:  python scripts/research_smart_panel.py      (writes data_store/smart_panel.parquet)
"""
from __future__ import annotations
import glob, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.research_tranching import is_stock

BASE = Path(__file__).parent.parent
DAILY = BASE/"data_store/eod2/src/eod2_data/daily"
OUT = BASE/"data_store/smart_panel.parquet"
START = "2007-01-01"
FIELDS = {"Open":"open","High":"high","Low":"low","Close":"close","Volume":"vol",
          "DLV_QTY":"dlv","TOTAL_TRADES":"trades"}


def build():
    cols = {v:{} for v in FIELDS.values()}
    for f in glob.glob(str(DAILY/"*.csv")):
        sym = Path(f).stem.upper()
        if not is_stock(sym): continue
        try:
            d = pd.read_csv(f, usecols=["Date","Series",*FIELDS])
        except Exception:
            continue
        d = d[d["Series"]=="EQ"]
        d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
        d = d.dropna(subset=["Date"])
        d = d[d["Date"]>=START]
        if len(d) < 250: continue
        d = d.set_index("Date").sort_index()
        d = d[~d.index.duplicated(keep="last")]
        for src, dst in FIELDS.items():
            cols[dst][sym] = pd.to_numeric(d[src], errors="coerce").astype("float32")
    frames = {k: pd.DataFrame(v).sort_index() for k, v in cols.items()}
    idx = frames["close"].index
    out = pd.concat({k: v.reindex(idx) for k, v in frames.items()}, axis=1)
    out.to_parquet(OUT)
    return out


def load():
    if not OUT.exists():
        return build()
    return pd.read_parquet(OUT)


if __name__ == "__main__":
    p = build()
    c = p["close"]
    print(f"panel {p.shape}  dates {c.index[0].date()}..{c.index[-1].date()}  symbols {c.shape[1]}")
    dl = p["dlv"].notna().sum(1)
    print("first date with delivery data on >100 names:", dl[dl > 100].index[0].date())
