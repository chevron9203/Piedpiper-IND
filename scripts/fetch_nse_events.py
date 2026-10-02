"""
Download corporate event history from NSE's JSON API (needs a browser-like session).

  results  quarterly financial-results filings with exact broadcast timestamps (2009+)
           -> event dates for post-earnings drift; the surprise is measured from the
              price reaction around the broadcast, so no consensus-estimate data needed
  pit      SEBI PIT insider / promoter trade disclosures (2015+)
  boardmtg board meetings with purpose (2024+). The results feed dries up from 2025
           (SEBI integrated filings), so results-meeting dates fill in event dates
  ann      every corporate announcement (2012+): category, 1-2 line summary, timestamp,
           attachment link. Monthly files (verified uncapped: Jan-2012 monthly = sum of days)
  corpact  corporate actions (bonus / split / dividend ...) with ex-dates (2006+)
           -> price adjustment for EVERY stock, delisted ones included

Raw JSON per month lands in data_store/nse_raw/events/{results,pit}/YYYYMM.json.
Months already on disk are skipped except the last 3 (they may still be filling in).

Run:  python scripts/fetch_nse_events.py [feed ...]     (default: all feeds)
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import pandas as pd, requests

BASE = Path(__file__).parent.parent
OUT = BASE/"data_store/nse_raw/events"
API = "https://www.nseindia.com/api"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126 Safari/537.36")
FEEDS = {
    "results": ("2009-01-01", "/corporates-financial-results?index=equities&period=Quarterly"
                              "&from_date={f}&to_date={t}"),
    "pit":     ("2015-01-01", "/corporates-pit?index=equities&from_date={f}&to_date={t}"),
    "corpact": ("2006-01-01", "/corporates-corporateActions?index=equities&from_date={f}&to_date={t}"),
    "boardmtg": ("2024-01-01", "/corporate-board-meetings?index=equities&from_date={f}&to_date={t}"),
    "ann":     ("2012-01-01", "/corporate-announcements?index=equities&from_date={f}&to_date={t}"),
}
DAILY: set[str] = set()              # feeds needing one file per day (monthly queries verified uncapped)


def new_session():
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept": "application/json, text/plain, */*",
                      "Referer": "https://www.nseindia.com/"})
    return s


def get(s, url):
    for attempt in range(5):
        try:
            r = s.get(url, timeout=60)
            if r.status_code == 200:
                j = r.json()
                return j.get("data", j) if isinstance(j, dict) else j
        except (requests.RequestException, ValueError):
            pass
        time.sleep(3*(attempt+1))
        s.cookies.clear()
    return None


def main():
    s = new_session()
    today = pd.Timestamp.today().normalize()
    recent = today - pd.DateOffset(months=3)
    only = set(sys.argv[1:]) or set(FEEDS)
    for feed, (start, path) in FEEDS.items():
        if feed not in only: continue
        (OUT/feed).mkdir(parents=True, exist_ok=True)
        daily = feed in DAILY
        for m in pd.date_range(start, today, freq="D" if daily else "MS"):
            out = OUT/feed/(f"{m:%Y%m%d}.json" if daily else f"{m:%Y%m}.json")
            if out.exists() and m < (today - pd.Timedelta(days=7) if daily else recent):
                continue
            end = m if daily else min(m + pd.offsets.MonthEnd(0), today)
            rows = get(s, API + path.format(f=f"{m:%d-%m-%Y}", t=f"{end:%d-%m-%Y}"))
            if rows is None:
                print(f"{feed} {m:%Y-%m} FAILED", flush=True)
                continue
            out.write_text(json.dumps(rows))
            if not daily or m.day == 1:
                print(f"{feed} {m:%Y-%m-%d} {len(rows)}", flush=True)
            time.sleep(0.6 if daily else 1.2)


if __name__ == "__main__":
    main()
