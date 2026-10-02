"""
Corporate announcements as signals (rules first; an LLM only where rules fall short).

Source: fetch_nse_events.py ann -> every NSE announcement since 2012 with category
(desc), 1-2 line summary (attchmntText) and broadcast timestamp.

  classify   category + keyword rules -> event type; order size parsed from the text
             ("order worth Rs 450 crore") and scaled by the stock's yearly cash turnover
  study      DEV event study: market-adjusted return t+1 -> t+21 / t+63 by event type,
             only stocks in the point-in-time universe. Signal must show up here first.
  build      features on decision dates -> data_store/smart_ann_s{step}.parquet

Point-in-time: an announcement after 15:30 is tradable from the NEXT day's close.
Run:  python scripts/research_smart_ann.py study | build [--step 5]
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))
ANN = BASE/"data_store/nse_raw/events/ann"
TABLE = BASE/"data_store/announcements.parquet"

# order matters: first matching rule wins
RULES = [
    ("default",      r"\bdefault|insolvency|\bnclt\b|\bcirp\b|resolution professional|winding up"),
    ("regulatory",   r"search and seizure|\braid\b|enforcement directorate|\bcbi\b|show cause|penalt|sebi order|adjudicat|income tax (search|survey)|forensic"),
    ("auditor_exit", r"auditor.{0,40}resign|resign.{0,40}auditor"),
    ("key_exit",     r"resign.{0,60}(managing director|\bmd\b|\bceo\b|\bcfo\b|chief executive|chief financial|whole.?time director)|(managing director|\bceo\b|\bcfo\b|chief executive|chief financial).{0,60}resign"),
    ("pledge",       r"pledge|encumbranc|invocation"),
    ("rating_up",    r"(credit )?rating.{0,80}(upgrad|revised upward|positive outlook)|upgrad.{0,40}rating"),
    ("rating_down",  r"(credit )?rating.{0,80}(downgrad|revised downward|negative outlook|watch with negative)|downgrad.{0,40}rating"),
    ("buyback",      r"buy.?back"),
    ("bonus_split",  r"\bbonus\b|sub-?divi|stock split|split of"),
    ("order_win",    r"(receiv|bag|secur|award|bagg|won|win).{0,60}(order|contract|letter of (award|intent)|\bloi\b|\bloa\b)|(order|contract)s?.{0,40}(receiv|bag|secur|award)"),
    ("capacity",     r"capacity|expansion|commission|commercial production|new (plant|unit|facility)|greenfield|brownfield|capex"),
    ("acquisition",  r"acqui|amalgamat|\bmerger\b|scheme of arrangement|demerg|stake in"),
    ("fund_raise",   r"\bqip\b|qualified institution|preferential (issue|allot)|rights issue|warrants|fund.?rais|\bncd\b"),
    ("guidance",     r"investor presentation|analyst|earnings call|conference call|press release"),
    ("clarify",      r"news verification|clarification|price movement|spurt in volume"),
]
_RX = [(k, re.compile(p, re.I)) for k, p in RULES]
AMT = re.compile(r"(?:rs\.?|inr|₹)\s*([\d,]+(?:\.\d+)?)\s*(crore|crores|cr\.?|lakh|lakhs|lacs|million|mn|billion|bn)\b", re.I)
MULT = {"crore": 1e7, "crores": 1e7, "cr": 1e7, "cr.": 1e7, "lakh": 1e5, "lakhs": 1e5, "lacs": 1e5,
        "million": 1e6, "mn": 1e6, "billion": 1e9, "bn": 1e9}


def classify(desc, text):
    s = f"{desc or ''} || {text or ''}"
    for k, rx in _RX:
        if rx.search(s):
            return k
    return "other"


def amount(text):
    vals = [float(v.replace(",", ""))*MULT[u.lower()] for v, u in AMT.findall(text or "")]
    return max(vals) if vals else np.nan


def load():
    if TABLE.exists() and TABLE.stat().st_mtime > max(p.stat().st_mtime for p in ANN.glob("*.json")):
        return pd.read_parquet(TABLE)
    rows = []
    for p in sorted(ANN.glob("*.json")):
        for x in json.loads(p.read_text()):
            rows.append((str(x.get("symbol", "")).strip(), x.get("sort_date") or x.get("an_dt"),
                         x.get("desc"), x.get("attchmntText"), x.get("attchmntFile")))
    D = pd.DataFrame(rows, columns=["sym", "ts", "desc", "text", "file"])
    D["ts"] = pd.to_datetime(D["ts"], errors="coerce")
    D = D.dropna(subset=["ts"]).drop_duplicates(["sym", "ts", "desc", "text"])
    D["type"] = [classify(d, t) for d, t in zip(D["desc"], D["text"])]
    D["amt"] = [amount(t) if ty in ("order_win", "capacity", "acquisition", "fund_raise", "buyback") else np.nan
                for ty, t in zip(D["type"], D["text"])]
    from scripts.build_nse_panel import apply_renames, symbol_changes
    D["date"] = D["ts"]
    D = apply_renames(D, symbol_changes()).drop(columns=["date"])
    D.to_parquet(TABLE)
    return D


def tradable_day(ts, idx):
    after = (ts.dt.hour*60 + ts.dt.minute) > 15*60 + 30
    day = ts.dt.normalize() + pd.to_timedelta(after.astype(int), unit="D")
    pos = idx.searchsorted(day.values)
    return pos                                   # index of first close at/after availability


def cmd_study(a):
    from scripts import research_smart_ml as SM
    D = load()
    P = SM.load_panel()
    C, R, alive, turn60, age, univ, mkt = SM.base(P)
    idx = C.index
    D = D[D["sym"].isin(C.columns)].copy()
    D["pos"] = tradable_day(D["ts"], idx)
    D = D[(D["pos"] < len(idx) - 64) & (idx[np.minimum(D["pos"], len(idx) - 1)] <= pd.Timestamp("2022-12-31"))]
    col = C.columns.get_indexer(D["sym"])
    D = D[univ.values[D["pos"].values, col]]            # in the investable universe that day
    col = C.columns.get_indexer(D["sym"]); pos = D["pos"].values
    Cf = C.ffill().values
    mc = (1 + mkt.fillna(0)).cumprod().values
    yearly_turn = (turn60.values[pos, col]*250)
    D["size"] = D["amt"]/yearly_turn
    print(f"DEV 2012-2022 announcements on universe stocks: {len(D):,}  ({D.sym.nunique()} names)\n")
    print(f"{'type':<13}{'n':>8}  {'day0 react':>10}  {'+21d':>8} {'t':>5}  {'+63d':>8} {'t':>5}   (market-adjusted, entry at t+1 close)")
    out = []
    for ty, g in D.groupby("type"):
        p, c = g["pos"].values, C.columns.get_indexer(g["sym"])
        react = Cf[p, c]/Cf[p - 1, c] - 1 - (mc[p]/mc[p - 1] - 1)
        res = {}
        for h in (21, 63):
            r = Cf[p + 1 + h, c]/Cf[p + 1, c] - 1 - (mc[p + 1 + h]/mc[p + 1] - 1)
            r = r[np.isfinite(r)]
            res[h] = (r.mean(), r.mean()/r.std()*np.sqrt(len(r)) if len(r) > 2 else np.nan)
        out.append((ty, len(g), np.nanmean(react), res[21], res[63]))
    for ty, n, rc, r21, r63 in sorted(out, key=lambda x: -x[3][0]):
        print(f"{ty:<13}{n:>8}  {rc:>+10.2%}  {r21[0]:>+8.2%} {r21[1]:>5.1f}  {r63[0]:>+8.2%} {r63[1]:>5.1f}")
    # order wins: does the SIZE matter?
    o = D[(D["type"] == "order_win") & D["size"].notna()]
    if len(o) > 50:
        o = o.assign(b=pd.qcut(o["size"], 4, labels=["small", "mid", "large", "huge"]))
        p, c = o["pos"].values, C.columns.get_indexer(o["sym"])
        o["r63"] = Cf[p + 64, c]/Cf[p + 1, c] - 1 - (mc[p + 64]/mc[p + 1] - 1)
        print("\norder wins by size (order value / yearly turnover), +63d market-adjusted:")
        print(o.groupby("b", observed=True)["r63"].agg(["count", "mean"]).to_string())


def build(step):
    from scripts import research_smart_ml as SM
    D = load()
    P = SM.load_panel()
    X, C, univ, mkt = SM.compute_features(P, step)
    idx = C.index
    D = D[D["sym"].isin(C.columns)].copy()
    D["pos"] = tradable_day(D["ts"], idx)
    D = D[D["pos"] < len(idx)]
    D["day"] = idx[D["pos"].values]
    turn60 = P["turn"].rolling(60, min_periods=40).median()
    col = C.columns.get_indexer(D["sym"])
    D["size"] = (D["amt"]/(turn60.values[D["pos"].values, col]*250)).clip(0, 20)
    feats = {}
    types = ["order_win", "capacity", "acquisition", "fund_raise", "buyback", "rating_up", "rating_down",
             "key_exit", "auditor_exit", "pledge", "regulatory", "default", "clarify", "guidance"]
    for ty in types:
        g = D[D["type"] == ty]
        daily = g.groupby(["day", "sym"]).size().unstack().reindex(index=idx, columns=C.columns).fillna(0)
        feats[f"ann_{ty}"] = daily.rolling(63, min_periods=1).sum()
    osz = D[D["type"] == "order_win"].groupby(["day", "sym"])["size"].sum().unstack() \
        .reindex(index=idx, columns=C.columns).fillna(0)
    feats["order_size63"] = osz.rolling(63, min_periods=1).sum()
    feats["ann_count21"] = (D.groupby(["day", "sym"]).size().unstack()
                            .reindex(index=idx, columns=C.columns).fillna(0).rolling(21, min_periods=1).sum())
    live = idx >= pd.Timestamp("2012-03-01")              # feed starts 2012
    dates = X.index.get_level_values("date").unique()
    out = pd.DataFrame({k: v.where(pd.Series(live, index=idx), axis=0).reindex(dates)
                        .stack(future_stack=True).reindex(X.index) for k, v in feats.items()}).astype("float32")
    out.to_parquet(BASE/f"data_store/smart_ann_s{step}.parquet")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["study", "build"])
    ap.add_argument("--step", type=int, default=5)
    a = ap.parse_args()
    if a.cmd == "study":
        cmd_study(a)
    else:
        out = build(a.step)
        print("non-zero share:", {c: f"{(out[c] > 0).mean():.1%}" for c in out.columns})
