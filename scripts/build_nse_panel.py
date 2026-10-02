"""
Survivorship-free, split/bonus-adjusted daily panel built from raw NSE archives.

Why not eod2: eod2 keeps only symbols that still trade, so delisted / merged / bust
names vanish from history -- a stock-picking backtest on it never meets the losers it
would really have bought. NSE's daily bhavcopy contains every security that traded.

Adjustment: bonus/split/consolidation ratios come from NSE's corporate-action filings
(fetch_nse_events.py corpact), which cover delisted names too. Each filing is accepted
only if the raw overnight gap on its ex-date agrees with the ratio. (Bhavcopy PREVCLOSE
is NOT adjusted on ex-dates -- tried, RELIANCE's three bonuses were all missed.)
Prices before an ex-date are multiplied by the cumulative product of later factors;
volumes divided. Validated against eod2's independently adjusted series (--validate).

Identity: date-aware symbol changes (NSE symbolchange.csv) are applied in chronological
order so TELCO -> TATAMOTORS -> TMPV is one continuous history, while a symbol later
re-used by a different company is not merged into the old one.

Series kept: EQ (normal), BE/BZ (trade-to-trade / non-compliant: where distressed
names go, which is exactly what survivorship hides). SME (SM/ST) excluded.

Outputs (data_store/):
  nse_panel.parquet   wide panel, fields open/high/low/close/vol/turn/trades/dlv/t2t
  nse_index.parquet   official index closes + P/E, P/B, div yield (2012-02+)

Run:  python scripts/build_nse_panel.py [--validate]
"""
from __future__ import annotations
import argparse, io, json, re, sys, zipfile
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(__file__).parent.parent
RAW = BASE/"data_store/nse_raw"
OUT = BASE/"data_store/nse_panel.parquet"
OUT_IDX = BASE/"data_store/nse_index.parquet"
SERIES = {"EQ", "BE", "BZ"}

OLD = {"SYMBOL":"sym","SERIES":"series","OPEN":"open","HIGH":"high","LOW":"low",
       "CLOSE":"close","PREVCLOSE":"prev","TOTTRDQTY":"vol","TOTTRDVAL":"turn",
       "TOTALTRADES":"trades"}
UDIFF = {"TckrSymb":"sym","SctySrs":"series","OpnPric":"open","HghPric":"high",
         "LwPric":"low","ClsPric":"close","PrvsClsgPric":"prev","TtlTradgVol":"vol",
         "TtlTrfVal":"turn","TtlNbOfTxsExctd":"trades"}


def read_bhav(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        raw = z.read(z.namelist()[0])
    d = pd.read_csv(io.BytesIO(raw), skipinitialspace=True)
    d.columns = [c.strip() for c in d.columns]
    cmap = UDIFF if "TckrSymb" in d.columns else OLD
    d = d[[c for c in cmap if c in d.columns]].rename(columns=cmap)
    d["series"] = d["series"].astype(str).str.strip()
    d = d[d["series"].isin(SERIES)]
    d["sym"] = d["sym"].astype(str).str.strip()
    d["date"] = pd.Timestamp(path.name[:8])
    return d


def read_mto(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        p = line.split(",")
        if len(p) >= 6 and p[0].strip() == "20" and p[3].strip() == "EQ":
            rows.append((p[2].strip(), p[5].strip()))
    d = pd.DataFrame(rows, columns=["sym", "dlv"])
    d["dlv"] = pd.to_numeric(d["dlv"], errors="coerce")
    d["date"] = pd.Timestamp(path.name[:8])
    return d


def symbol_changes() -> pd.DataFrame:
    rows = []
    for line in (RAW/"meta/symbolchange.csv").read_text(errors="ignore").splitlines():
        p = line.rsplit(",", 3)                 # company name may itself contain commas
        if len(p) != 4: continue
        dt = pd.to_datetime(p[3].strip(), format="%d-%b-%Y", errors="coerce")
        if pd.notna(dt):
            rows.append((p[1].strip(), p[2].strip(), dt))
    return pd.DataFrame(rows, columns=["old", "new", "date"]).sort_values("date")


def apply_renames(d: pd.DataFrame, ch: pd.DataFrame) -> pd.DataFrame:
    """Chronological: rows of OLD before the change date take NEW's name. Processing
    oldest change first lets chains (A->B->C) resolve to the latest name."""
    for old, new, dt in ch.itertuples(index=False):
        m = (d["sym"].values == old) & (d["date"].values < np.datetime64(dt))
        if m.any():
            d.loc[m, "sym"] = new
    return d


BONUS = re.compile(r"bonus\W*(\d+)\s*:\s*(\d+)", re.I)
SPLIT = re.compile(r"(?:split|sub-?div|consolidat).*?r[se]\.?\s*([\d.]+).*?\bto\b\s*r[se]\.?\s*([\d.]+)", re.I)


def corporate_actions(ch) -> pd.DataFrame:
    """Price factors from NSE corporate-action filings: Bonus a:b -> b/(a+b);
    face-value split/consolidation X -> Y -> Y/X. Dividends/rights/demergers ignored
    (eod2 does not adjust them either, so the series stay comparable)."""
    rows = []
    for p in sorted((RAW/"events/corpact").glob("*.json")):
        for x in json.loads(p.read_text()):
            subj = str(x.get("subject", "")); f = 1.0
            for a, b in BONUS.findall(subj):
                if float(a) > 0 and float(b) > 0: f *= float(b)/(float(a) + float(b))
            m = SPLIT.search(subj)
            if m and float(m.group(1)) > 0:
                f *= float(m.group(2))/float(m.group(1))
            dt = pd.to_datetime(x.get("exDate"), format="%d-%b-%Y", errors="coerce")
            if f != 1.0 and pd.notna(dt):
                rows.append((str(x.get("symbol", "")).strip(), dt, " ".join(subj.lower().split()), f))
    # de-dupe on the filing text, not the ratio: a split AND a bonus on the same day
    # can both be 0.5 and must compound to 0.25
    ca = pd.DataFrame(rows, columns=["sym", "date", "subj", "f"]).drop_duplicates(["sym", "date", "subj"])
    ca = apply_renames(ca[["sym", "date", "f"]], ch)
    return ca.groupby(["sym", "date"])["f"].prod().reset_index()


# only ratios beyond NSE's widest price band (20%): a smaller "clean" gap such as x0.8
# is indistinguishable from a lower-circuit day, so small unfiled bonuses stay unadjusted
CLEAN = sorted(r for r in ({b/(a + b) for a in range(1, 6) for b in range(1, 6)}
                           | {1/k for k in (2, 2.5, 4, 5, 10, 20, 25, 50, 100)} | {2, 5, 10})
               if abs(np.log(r)) >= 0.4)


def detect_unfiled(d, gap, fac):
    """Some splits/bonuses are missing from NSE's filing feed (e.g. JSWSTEEL's 2017
    10:1 split). Unadjusted, each one is a fake -50..-90% crash. A real split is told
    apart from a real crash on three counts:
      1 the overnight gap is beyond any circuit band and sits on a clean ratio
        (1/2, 1/5, 2/3, b/(a+b)...) within 4%
      2 the day itself is quiet once that ratio is removed (close-to-close within 15%)
      3 shares traded scale by ~1/ratio while VALUE traded stays roughly flat --
        a crash instead blows value-traded up (panic) and shares far beyond 1/ratio.
        A split that lands on its ratio within 1.5% at BOTH open and close may show a
        value boom too (cheaper shares draw retail) -- a crash essentially never lands
        that precisely, so that tier tolerates value up to 6x."""
    sym = d["sym"]
    g_close = d["close"]/d.groupby("sym")["close"].shift(1)
    vol, turn = d["vol"], d["turn"]
    roll = lambda s, k: s.groupby(sym).transform(lambda x: x.rolling(k, min_periods=5).median())
    v_pre = roll(vol, 20).groupby(sym).shift(1); t_pre = roll(turn, 20).groupby(sym).shift(1)
    rev = lambda s: s[::-1].groupby(sym[::-1]).transform(
        lambda x: x.rolling(20, min_periods=5).median())[::-1].groupby(sym).shift(-1)
    v_post, t_post = rev(vol), rev(turn)
    lg = np.log(gap.values)
    prev = d.groupby("sym")["close"].shift(1).values
    # penny stocks (< Rs 5) move a whole tick at a time: 0.15 -> 0.10 looks like a 2:3 split
    cand = np.isfinite(lg) & (np.abs(lg) > 0.36) & (fac == 1.0) & (prev >= 5)
    found = 0
    for k in np.where(cand)[0]:
        r = min(CLEAN, key=lambda c: abs(np.log(gap.values[k]/c)))
        if abs(np.log(gap.values[k]/r)) > 0.04: continue
        if abs(np.log(g_close.values[k]/r)) > 0.15: continue
        qr = v_post.values[k]/v_pre.values[k]; tr = t_post.values[k]/t_pre.values[k]
        if not (np.isfinite(qr) and np.isfinite(tr)): continue
        # share count must move in the split's direction (loosely: post-split retail
        # liquidity often overshoots 1/ratio); traded VALUE is the strict crash test
        precise = (abs(np.log(gap.values[k]/r)) < 0.015
                   and abs(np.log(g_close.values[k]/r)) < 0.05)
        if abs(np.log(qr*r)) < 1.2 and (0.4 < tr < 2.5 or (precise and tr < 6)):
            fac[k] = r; found += 1
    print(f"unfiled splits/bonuses detected from the tape: {found}")
    return fac


def adjust(d: pd.DataFrame, ca: pd.DataFrame) -> pd.DataFrame:
    """Apply filed corporate actions, each one checked against the tape: the raw
    overnight move on the ex-date must be closer to the filed ratio than to no change,
    otherwise the filing (wrong date / re-used symbol / typo) is rejected."""
    d = d.sort_values(["sym", "date"]).reset_index(drop=True)
    prev_close = d.groupby("sym")["close"].shift(1)
    gap = (d["open"].where(d["open"] > 0, d["close"])/prev_close)
    # snap each filing to the first trading row on/after its ex-date for that symbol
    key = d[["sym", "date"]].reset_index()
    ca = ca.sort_values("date")
    snapped = pd.merge_asof(ca, key.sort_values("date"), on="date", by="sym", direction="forward",
                            tolerance=pd.Timedelta(days=10))
    snapped = snapped.dropna(subset=["index"])
    i = snapped["index"].astype(int).values
    g = gap.values[i]; f = snapped["f"].values
    # reject only when the tape clearly contradicts the filing: the gap is nearer "no
    # change" than the ratio AND is far from the ratio (a 1:4 bonus is x0.8, so a stock
    # rallying 10% on its ex-date must not get its bonus thrown away)
    miss = np.abs(np.log(g/f))
    ok = np.isfinite(g) & ~((np.abs(np.log(g)) < miss) & (miss > 0.15))
    self_ok = ok.sum(); print(f"corporate actions: {len(ca)} filed, {len(snapped)} matched to tape, "
                              f"{self_ok} confirmed by price gap, {len(snapped)-self_ok} rejected")
    fac = np.ones(len(d)); fac[i[ok]] = f[ok]
    fac = detect_unfiled(d, gap, fac)
    d["ca_factor"] = fac
    f = d["ca_factor"]
    # cumulative factor applying to each row = product of factors strictly AFTER it
    rev = f[::-1].groupby(d["sym"][::-1]).cumprod()[::-1]
    after = rev/f
    for c in ("open", "high", "low", "close"):
        d[c] = d[c]*after
    for c in ("vol", "dlv"):
        if c in d: d[c] = d[c]/after
    return d


def etf_symbols() -> set:
    """ETFs trade in the EQ series but are not stocks; research universes must drop them."""
    sys.path.insert(0, str(BASE))
    from scripts.research_tranching import is_stock
    lst = set(pd.read_csv(RAW/"meta/eq_etfseclist.csv")["Symbol"].str.strip())
    return lst | {s for s in _all_symbols() if not is_stock(s)}


def _all_symbols():
    return {p.stem.upper() for p in (BASE/"data_store/eod2/src/eod2_data/daily").glob("*.csv")}


def build_index():
    frames = []
    for p in sorted((RAW/"ind").glob("*.csv")):
        try:
            x = pd.read_csv(p)
        except Exception:
            continue
        x.columns = [c.strip() for c in x.columns]
        x = x.rename(columns={"Index Name":"index","Closing Index Value":"close",
                              "P/E":"pe","P/B":"pb","Div Yield":"dy"})
        x["date"] = pd.Timestamp(p.name[:8])
        frames.append(x[["date","index","close","pe","pb","dy"]])
    idx = pd.concat(frames)
    idx["index"] = idx["index"].str.strip().str.upper()
    for c in ("close","pe","pb","dy"):
        idx[c] = pd.to_numeric(idx[c], errors="coerce")
    idx.to_parquet(OUT_IDX)
    return idx


def build():
    files = sorted(p for p in (RAW/"bhav").glob("*.zip"))
    bh = pd.concat([read_bhav(p) for p in files], ignore_index=True)
    for c in ("open","high","low","close","prev","vol","turn","trades"):
        bh[c] = pd.to_numeric(bh[c], errors="coerce")
    # same symbol on EQ and BE the same day (rare): keep the more traded row
    bh = bh.sort_values("vol").drop_duplicates(["sym","date"], keep="last")
    mto = pd.concat([read_mto(p) for p in sorted((RAW/"mto").glob("*.DAT"))], ignore_index=True)
    mto = mto.drop_duplicates(["sym","date"], keep="last")
    bh = bh.merge(mto, on=["sym","date"], how="left")
    bh.loc[bh["series"] != "EQ", "dlv"] = np.nan     # T2T is 100% delivery by rule
    bh["t2t"] = (bh["series"] != "EQ").astype("float32")
    ch = symbol_changes()
    bh = apply_renames(bh, ch)
    bh = bh.sort_values("vol").drop_duplicates(["sym","date"], keep="last")
    bh = adjust(bh, corporate_actions(ch))
    fields = ["open","high","low","close","vol","turn","trades","dlv","t2t","ca_factor"]
    wide = bh.pivot(index="date", columns="sym", values=fields).astype("float32")
    wide.to_parquet(OUT)
    return wide


def validate(wide):
    """Adjusted close vs eod2's adjusted close: the ratio should be flat over time."""
    import glob
    c = wide["close"]
    etf = etf_symbols()
    ok = tot = 0; worst = []
    for f in glob.glob(str(BASE/"data_store/eod2/src/eod2_data/daily/*.csv")):
        sym = Path(f).stem.upper()
        if sym not in c.columns or sym in etf: continue
        e = pd.read_csv(f, usecols=["Date","Close","Series"])
        e = e[e["Series"].isin(SERIES)]
        e["Date"] = pd.to_datetime(e["Date"], errors="coerce")
        e = e.dropna().drop_duplicates("Date", keep="last").set_index("Date")["Close"]
        both = pd.concat([c[sym], pd.to_numeric(e, errors="coerce").astype("float64")], axis=1).dropna()
        if len(both) < 250: continue
        r = both.iloc[:,0]/both.iloc[:,1]
        r = r/r.iloc[-1]
        good = ((r - 1).abs() < 0.01).mean()
        ok += good*len(r); tot += len(r)
        worst.append((good, sym, len(r)))
    worst.sort()
    print(f"validated vs eod2: {ok/tot:.2%} of {tot:,} symbol-days within 1%")
    print("worst agreement:", [(s, f"{g:.0%}") for g, s, _ in worst[:15]])
    return worst


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()
    w = build(); idx = build_index()
    c = w["close"]
    print(f"panel {c.shape}  {c.index[0].date()}..{c.index[-1].date()}")
    print("names/date by year:", c.notna().sum(axis=1).groupby(c.index.year).median().astype(int).to_dict())
    print("indices:", idx["index"].nunique(), idx["date"].min().date(), "..", idx["date"].max().date())
    if a.validate: validate(w)
