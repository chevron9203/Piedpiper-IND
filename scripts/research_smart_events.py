"""
Event features for the smart-system research: earnings reactions and insider buying.

Both are documented Indian under-reactions:
  * post-earnings drift: stocks drift in the direction of the results surprise for
    ~64 days (2002-2017 NSE study). No consensus data needed: the surprise is measured
    by the market itself -- the abnormal return in the 2 days around the release (EAR).
  * insider purchases: +6.67% 90-day abnormal return after high-value insider buys,
    stronger in smaller firms (SEBI PIT disclosures study).

Point-in-time rules (an event only exists once you could have seen it):
  results   event day d0 = release date (results feed, 2009..2025-03) or results board
            meeting date (board-meeting feed, 2024+); EAR window close[d0-1] -> close[d0+1],
            knowable at close d0+1
  insider   knowable the trading day AFTER the disclosure timestamp

Output: data_store/smart_events_s{step}.parquet keyed (date, sym) on decision dates.

Run:  python scripts/research_smart_events.py [--step 5]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))
from scripts.build_nse_panel import apply_renames, symbol_changes
from scripts import research_smart_ml as SM

EV = BASE/"data_store/nse_raw/events"
WINDOW = 63                          # trading days an earnings reaction stays "live"


def results_events(ch):
    rows = []
    for p in sorted((EV/"results").glob("*.json")):
        for x in json.loads(p.read_text()):
            t = pd.to_datetime(x.get("broadCastDate"), format="%d-%b-%Y %H:%M:%S", errors="coerce")
            if pd.notna(t) and x.get("symbol"):
                rows.append((x["symbol"].strip(), t.normalize()))
    for p in sorted((EV/"boardmtg").glob("*.json")):
        for x in json.loads(p.read_text()):
            txt = f"{x.get('bm_purpose','')} {x.get('bm_desc','')}".lower()
            if "financial result" not in txt: continue
            t = pd.to_datetime(x.get("bm_date"), format="%d-%b-%Y", errors="coerce")
            if pd.notna(t) and x.get("bm_symbol"):
                rows.append((x["bm_symbol"].strip(), t))
    ev = pd.DataFrame(rows, columns=["sym", "date"]).drop_duplicates()
    ev = apply_renames(ev, ch).sort_values(["sym", "date"])
    # one event per results season: a standalone + consolidated filing, or a meeting and
    # its filing, a few days apart are the same news -- keep the first
    gap = ev.groupby("sym")["date"].diff().dt.days
    return ev[(gap.isna()) | (gap > 20)].reset_index(drop=True)


def insider_events(ch):
    rows = []
    for p in sorted((EV/"pit").glob("*.json")):
        for x in json.loads(p.read_text()):
            cat = str(x.get("personCategory", ""))
            mode = str(x.get("acqMode", ""))
            if mode not in ("Market Purchase", "Market Sale"): continue
            t = pd.to_datetime(x.get("date"), format="%d-%b-%Y %H:%M", errors="coerce")
            if pd.isna(t) or not x.get("symbol"): continue
            buy = pd.to_numeric(x.get("buyValue"), errors="coerce")
            sell = pd.to_numeric(x.get("sellValue"), errors="coerce")
            val = pd.to_numeric(x.get("secVal"), errors="coerce")
            typ = str(x.get("tdpTransactionType", "")).lower()
            v = buy if buy and buy > 0 else (val if "buy" in typ or mode == "Market Purchase" else 0)
            s = sell if sell and sell > 0 else (val if "sell" in typ or mode == "Market Sale" else 0)
            sign = 1 if mode == "Market Purchase" else -1
            amt = (v if sign > 0 else s) or 0
            rows.append((x["symbol"].strip(), t, sign*float(amt),
                         cat in ("Promoters", "Promoter Group"), str(x.get("acqName", ""))))
    ev = pd.DataFrame(rows, columns=["sym", "date", "value", "promoter", "who"])
    ev = ev.drop_duplicates(["sym", "date", "value", "who"])
    ev["date"] = ev["date"].dt.normalize()
    return apply_renames(ev, ch)


def build(step, P=None, dates=None, age_offset=None, save=True):
    P = SM.load_panel() if P is None else P
    C, R, alive, turn60, age, univ, mkt = SM.base(P, age_offset)
    idx = C.index
    dates = idx[260::step] if dates is None else pd.DatetimeIndex(dates)
    ch = symbol_changes()

    # ---- earnings reaction
    ev = results_events(ch)
    ev = ev[ev["sym"].isin(C.columns)]
    pos = idx.searchsorted(ev["date"].values)          # first trading day on/after d0
    ok = (pos >= 1) & (pos + 1 < len(idx))
    ev, pos = ev[ok].copy(), pos[ok]
    col = C.columns.get_indexer(ev["sym"])
    Cv = C.values; Tv = P["turn"].values; t60 = turn60.values
    mc = (1 + mkt.fillna(0)).cumprod().values
    p0 = Cv[pos - 1, col]; p1 = Cv[pos + 1, col]
    ev["ear"] = (p1/p0) - (mc[pos + 1]/mc[pos - 1])
    ev["ear_vol"] = (np.nan_to_num(Tv[pos, col]) + np.nan_to_num(Tv[pos + 1, col]))/2/t60[pos - 1, col]
    ev["known"] = idx[pos + 1]
    ev = ev.dropna(subset=["ear"]).sort_values(["sym", "known"])
    ev["ear_prev"] = ev.groupby("sym")["ear"].shift(1)

    keys = univ.reindex(dates).stack()
    keys = keys[keys].index.set_names(["date", "sym"])
    K = keys.to_frame(index=False).sort_values("date")
    E = pd.merge_asof(K, ev[["sym", "known", "ear", "ear_vol", "ear_prev"]].sort_values("known"),
                      left_on="date", right_on="known", by="sym", direction="backward")
    E["ear_days"] = idx.get_indexer(E["date"]) - idx.get_indexer(E["known"].fillna(idx[0]))
    stale = E["known"].isna() | (E["ear_days"] > WINDOW)
    E.loc[stale, ["ear", "ear_vol", "ear_prev"]] = np.nan
    E.loc[E["known"].isna(), "ear_days"] = np.nan
    E["ear_days"] = E["ear_days"].clip(upper=WINDOW + 1)
    E = E.set_index(["date", "sym"])[["ear", "ear_vol", "ear_prev", "ear_days"]]

    # ---- insider / promoter market trades, rolling 63 trading days (~90 calendar)
    ins = insider_events(ch)
    ins = ins[ins["sym"].isin(C.columns)]
    ins["known"] = idx[np.minimum(idx.searchsorted(ins["date"].values, side="right"), len(idx) - 1)]
    def rolling_sum(df, name):
        daily = df.groupby(["known", "sym"])["value"].sum().unstack().reindex(idx).fillna(0.0)
        daily = daily.reindex(columns=C.columns, fill_value=0.0)
        return daily.rolling(63, min_periods=1).sum()
    prom = rolling_sum(ins[ins["promoter"]], "prom")
    allin = rolling_sum(ins, "all")
    buyers = (ins[ins["value"] > 0].groupby(["known", "sym"]).size().unstack()
              .reindex(idx).reindex(columns=C.columns).fillna(0).rolling(63, min_periods=1).sum())
    scale = (turn60*63).replace(0, np.nan)
    pit_live = idx >= pd.Timestamp("2015-06-01")
    feats = {"prom_net": prom/scale, "ins_net": allin/scale, "ins_buyers": buyers}
    I = pd.DataFrame({k: v.where(pd.Series(pit_live, index=idx), axis=0).reindex(dates)
                      .stack(future_stack=True).reindex(keys) for k, v in feats.items()})
    out = E.join(I).astype("float32")
    if save:
        out.to_parquet(BASE/f"data_store/smart_events_s{step}.parquet")
    return out, ev, ins


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--step", type=int, default=5)
    a = ap.parse_args()
    out, ev, ins = build(a.step)
    print(f"results events {len(ev):,} ({ev['known'].min().date()}..{ev['known'].max().date()}), "
          f"insider market trades {len(ins):,}")
    print("coverage on decision rows:", {c: f"{out[c].notna().mean():.0%}" for c in out.columns})
    print(out.describe().T[["mean", "50%", "min", "max"]].round(4))
