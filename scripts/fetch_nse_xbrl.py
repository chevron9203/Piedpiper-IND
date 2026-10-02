"""
Quarterly results as numbers: download every XBRL file linked from the NSE results feed
(fetch_nse_events.py results, mid-2018+) and parse it into one table.

  data_store/nse_raw/xbrl/<file>.xml        raw filings (skipped if present)
  data_store/fundamentals.parquet           one row per filing:
      sym, known (broadcast timestamp -> when the market could see it), period_end,
      cons (consolidated?), bank, revenue, other_inc, pbt, pat, exceptional, fin_cost,
      dep, employee, eps

Point-in-time: a filing exists from its broadCastDate. Revisions are separate filings
with their own (later) broadcast time, so a backtest never sees a figure early.

Run:  python scripts/fetch_nse_xbrl.py [--workers 6] [--parse-only]
"""
from __future__ import annotations
import argparse, json, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
import pandas as pd, requests

BASE = Path(__file__).parent.parent
RES = BASE/"data_store/nse_raw/events/results"
OUT = BASE/"data_store/nse_raw/xbrl"
TABLE = BASE/"data_store/fundamentals.parquet"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126 Safari/537.36")
_local = threading.local()
PAUSE = 0.8          # seconds between requests per worker: 6 workers flat-out got the IP 403-blocked

# first tag found wins (banks / NBFCs / insurers file different line items)
FIELDS = {
    "revenue":     ["RevenueFromOperations", "InterestEarned", "TotalRevenueFromOperations", "Income"],
    "other_inc":   ["OtherIncome"],
    "pbt":         ["ProfitBeforeTax", "ProfitLossFromOrdinaryActivitiesBeforeTax"],
    "pat":         ["ProfitLossForPeriod", "ProfitLossForThePeriod", "NetProfitLossForThePeriod"],
    "exceptional": ["ExceptionalItemsBeforeTax", "ExceptionalItems"],
    "fin_cost":    ["FinanceCosts", "InterestExpended"],
    "dep":         ["DepreciationDepletionAndAmortisationExpense"],
    "employee":    ["EmployeeBenefitExpense", "EmployeesCost"],
    "eps":         ["BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations",
                    "BasicEarningsPerShareAfterExtraordinaryItems"],
}


def index():
    rows = []
    for p in sorted(RES.glob("*.json")):
        for x in json.loads(p.read_text()):
            u = x.get("xbrl")
            if not (u and str(u).endswith(".xml")):
                continue
            rows.append({"sym": str(x.get("symbol", "")).strip(), "xbrl": u,
                         "known": pd.to_datetime(x.get("broadCastDate"), format="%d-%b-%Y %H:%M:%S", errors="coerce"),
                         "cons": x.get("consolidated") == "Consolidated", "bank": x.get("bank") == "Y",
                         "period_end": pd.to_datetime(x.get("toDate"), format="%d-%b-%Y", errors="coerce")})
    return pd.DataFrame(rows).drop_duplicates("xbrl")


def fetch(u):
    out = OUT/u.rsplit("/", 1)[-1]
    if out.exists():
        return "skip"
    if not hasattr(_local, "s"):
        _local.s = requests.Session(); _local.s.headers.update({"User-Agent": UA})
    for k in range(4):
        try:
            r = _local.s.get(u, timeout=40)
            if r.status_code == 200 and b"<" in r.content[:200]:
                out.write_bytes(r.content); time.sleep(PAUSE); return "ok"
            if r.status_code == 404:
                return "404"
            if r.status_code == 403:          # Akamai rate block: back off hard, never hammer
                time.sleep(120*(k + 1)); continue
        except requests.RequestException:
            pass
        time.sleep(2*(k + 1))
    return "fail"


def parse(path):
    try:
        t = Path(path).read_text(errors="ignore")
    except OSError:
        return None
    # the reporting-quarter context: the duration context ending on DateOfEndOfReportingPeriod
    end = re.search(r"DateOfEndOfReportingPeriod[^>]*>([\d-]+)<", t)
    ctxs = re.findall(r'<xbrli:context id="([^"]+)">.*?<xbrli:endDate>([\d-]+)</xbrli:endDate>', t, re.S)
    want = {c for c, e in ctxs if end and e == end.group(1)} or {"OneD"}
    rec = {}
    for k, tags in FIELDS.items():
        for tg in tags:
            m = re.findall(rf'<in-bse-fin:{tg}\b[^>]*contextRef="([^"]+)"[^>]*>([-\d.Ee]+)<', t)
            vals = [float(v) for c, v in m if c in want]
            if vals:
                rec[k] = vals[0]; break
    rec["file"] = Path(path).name
    return rec


def build_table(ix):
    files = [OUT/u.rsplit("/", 1)[-1] for u in ix["xbrl"]]
    with ProcessPoolExecutor(6) as ex:
        recs = [r for r in ex.map(parse, files, chunksize=200) if r]
    F = pd.DataFrame(recs)
    ix = ix.assign(file=[u.rsplit("/", 1)[-1] for u in ix["xbrl"]])
    T = ix.merge(F, on="file", how="inner").drop(columns=["xbrl"])
    from scripts.build_nse_panel import apply_renames, symbol_changes
    T["date"] = T["known"]
    T = apply_renames(T, symbol_changes()).drop(columns=["date"])
    T.to_parquet(TABLE)
    return T


if __name__ == "__main__":
    sys.path.insert(0, str(BASE))
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--parse-only", action="store_true")
    a = ap.parse_args()
    ix = index(); OUT.mkdir(parents=True, exist_ok=True)
    print(f"{len(ix):,} XBRL filings, {ix.sym.nunique()} symbols, {ix.known.min()}..{ix.known.max()}", flush=True)
    if not a.parse_only:
        counts = {}; t0 = time.time()
        with ThreadPoolExecutor(a.workers) as ex:
            for i, r in enumerate(ex.map(fetch, ix["xbrl"]), 1):
                counts[r] = counts.get(r, 0) + 1
                if i % 5000 == 0:
                    print(f"{i}/{len(ix)} {counts} {time.time()-t0:.0f}s", flush=True)
        print("download", counts, flush=True)
    T = build_table(ix)
    print(f"parsed {len(T):,} filings; field coverage:",
          {k: f"{T[k].notna().mean():.0%}" for k in FIELDS})
