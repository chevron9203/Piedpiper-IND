"""
Quarterly results from NSE's INTEGRATED FILING feed (replaces the old results feed, which stops carrying
numbers after 2025-03 when SEBI moved listed companies to integrated filings).

  /api/integrated-filing-results?index=equities&from_date=DD-MM-YYYY&to_date=DD-MM-YYYY&page=N   (20 rows/page)
  row: symbol, qe_Date (quarter end), consolidated, type_Sub (Original/Revised), broadcast_Date (point-in-time),
       xbrl (plain XBRL XML, same line-item names as the old feed, namespace in-capmkt), ixbrl (inline HTML)

One JSON per day -> data_store/nse_raw/events/integrated/YYYYMMDD.json. Days already on disk are skipped
except the last 5 (late and revised filings). fetch_nse_xbrl.py then downloads and parses the XML files.

Run:  python scripts/fetch_nse_integrated.py [--start 2025-01-01]
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import pandas as pd, requests

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))
from scripts.fetch_nse_events import API, new_session       # noqa: E402

OUT = BASE/"data_store/nse_raw/events/integrated"


def page_json(s, path):
    for attempt in range(6):
        try:
            r = s.get(API + path, timeout=60)
            if r.status_code == 200 and "json" in r.headers.get("content-type", ""):
                return r.json()
            if r.status_code in (401, 403):
                time.sleep(60*(attempt + 1)); s.cookies.clear(); s.get("https://www.nseindia.com/", timeout=30); continue
        except (requests.RequestException, ValueError):
            pass
        time.sleep(4*(attempt + 1)); s.cookies.clear()
    return None


def fetch_day(s, d):
    rows, page = [], 1
    while True:
        j = page_json(s, f"/integrated-filing-results?index=equities&from_date={d:%d-%m-%Y}&to_date={d:%d-%m-%Y}&page={page}")
        if j is None:
            return None
        data = j.get("data", []) or []
        rows += data
        try:
            total = int(j.get("totalCount") or 0)
        except ValueError:
            total = 0
        if len(data) < 20 or (total and len(rows) >= total):
            return rows
        page += 1
        time.sleep(0.8)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--start", default="2025-01-01")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    s = new_session(); s.get("https://www.nseindia.com/", timeout=30)
    today = pd.Timestamp.today().normalize()
    n_days = n_rows = 0
    for d in pd.date_range(a.start, today):
        f = OUT/f"{d:%Y%m%d}.json"
        if f.exists() and d < today - pd.Timedelta(days=5):
            continue
        rows = fetch_day(s, d)
        if rows is None:
            print(f"{d.date()} FAILED", flush=True); continue
        f.write_text(json.dumps(rows)); n_days += 1; n_rows += len(rows)
        if rows or d.day == 1:
            print(f"{d.date()} {len(rows)}", flush=True)
        time.sleep(0.8)
    tot = {}
    for p in sorted(OUT.glob("*.json")):
        tot[p.stem[:6]] = tot.get(p.stem[:6], 0) + len(json.loads(p.read_text()))
    print(f"fetched {n_days} days / {n_rows} rows; rows per month on disk: {tot}")


if __name__ == "__main__":
    main()
