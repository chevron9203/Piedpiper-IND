"""Rolling adjusted price window, rebuilt from raw NSE files with the research code.

A full 19-year rebuild needs ~6.4 GB RAM; the box has 5.8 GB and other jobs. Features
only look back ~260 trading days, so the live service keeps WINDOW days. The one input
that needs full history is listing age (MIN_AGE filter, log_age feature): state.json
holds every symbol's count of trading days through `asof`, so
    age_offset = total_through_asof - days_inside_window
makes the window's cumulative count equal the full-history count exactly.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

from smart.config import BASE, RAW, STATE, WINDOW

sys.path.insert(0, str(BASE))
from scripts import build_nse_panel as BP                 # noqa: E402
from scripts.research_tranching import is_stock           # noqa: E402

FIELDS = ("open", "high", "low", "close", "vol", "turn", "trades", "dlv", "t2t")


def trading_files(asof=None):
    days = {}
    for p in (RAW/"bhav").glob("*.zip"):
        days[pd.Timestamp(p.name[:8])] = p
    d = pd.Series(days).sort_index()
    return d if asof is None else d[d.index <= pd.Timestamp(asof)]


def etfs():
    return set(pd.read_csv(RAW/"meta/eq_etfseclist.csv")["Symbol"].str.strip())


def build(asof=None, window=WINDOW, quiet=True):
    """Adjusted wide panel {field: DataFrame} over the last `window` trading days."""
    files = trading_files(asof).iloc[-window:]
    bh = pd.concat([BP.read_bhav(p) for p in files.values], ignore_index=True)
    for c in ("open", "high", "low", "close", "prev", "vol", "turn", "trades"):
        bh[c] = pd.to_numeric(bh[c], errors="coerce")
    bh = bh.sort_values("vol").drop_duplicates(["sym", "date"], keep="last")
    mtos = [RAW/"mto"/f"{d:%Y%m%d}.DAT" for d in files.index]
    mto = pd.concat([BP.read_mto(p) for p in mtos if p.exists()], ignore_index=True)
    mto = mto.drop_duplicates(["sym", "date"], keep="last")
    bh = bh.merge(mto, on=["sym", "date"], how="left")
    bh.loc[bh["series"] != "EQ", "dlv"] = np.nan
    bh["t2t"] = (bh["series"] != "EQ").astype("float32")
    ch = BP.symbol_changes()
    bh = BP.apply_renames(bh, ch)
    bh = bh.sort_values("vol").drop_duplicates(["sym", "date"], keep="last")
    if quiet:
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            bh = BP.adjust(bh, BP.corporate_actions(ch))
    else:
        bh = BP.adjust(bh, BP.corporate_actions(ch))
    wide = bh.pivot(index="date", columns="sym", values=list(FIELDS)).astype("float32")
    etf = etfs()
    syms = [s for s in wide["close"].columns if s not in etf and is_stock(s)]
    return {k: wide[k][syms] for k in FIELDS}


# ---------------------------------------------------------------- listing-age state
def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else None


def save_state(st):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp"); tmp.write_text(json.dumps(st)); tmp.replace(STATE)


def init_state_from_panel(close: pd.DataFrame):
    """Bootstrap (Mac): full-history trading-day counts from the research panel."""
    cnt = close.notna().sum()
    return {"asof": str(close.index[-1].date()), "alive": {k: int(v) for k, v in cnt.items() if v > 0}}


def roll_state(st, P):
    """Advance counts from st['asof'] to the window's last day; carry renamed symbols."""
    asof = pd.Timestamp(st["asof"])
    alive = dict(st["alive"])
    for old, new, dt in BP.symbol_changes().itertuples(index=False):
        if dt > asof and old in alive:
            alive[new] = alive.get(new, 0) + alive.pop(old)
    new_rows = P["close"][P["close"].index > asof]
    for s, v in new_rows.notna().sum().items():
        if v:
            alive[s] = alive.get(s, 0) + int(v)
    return {"asof": str(P["close"].index[-1].date()), "alive": alive}


def age_offset(st, P):
    """Trading days each symbol had BEFORE the window (state must be as of window end)."""
    assert pd.Timestamp(st["asof"]) == P["close"].index[-1], "state not rolled to window end"
    total = pd.Series(st["alive"], dtype=float)
    inside = P["close"].notna().sum()
    return (total.reindex(inside.index).fillna(inside) - inside).clip(lower=0)
