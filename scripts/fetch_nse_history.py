"""
Download raw NSE daily archives so the research runs on survivorship-free data.

eod2 only keeps symbols that still trade, so ~every stock that was delisted, merged or
went bust since 2007 is missing -- the exact names a stock-picker must learn to avoid.
NSE's own daily files contain every security that traded that day:

  bhav  full cash-market bhavcopy (OHLC, volume, PREVCLOSE, ISIN, series)
        old format until 2024-07-05, UDiFF format after
  mto   security-wise delivery position (deliverable qty) -- delivery% back to 2007
  ind   ind_close_all -- official index closes (Nifty 50/500/Midcap/Smallcap ...)
  fo    F&O bhavcopy (--fo): every stock/index future and option, OI and change in OI
        -- positioning data for the ~200 F&O stocks (old format / UDiFF like bhav)

Files land in data_store/nse_raw/{bhav,mto,ind}/ and are skipped if already present,
so the script is safe to re-run (it also fills in only the newest days).

Run:  python scripts/fetch_nse_history.py [--start 2007-01-01] [--workers 6]
"""
from __future__ import annotations
import argparse, sys, time, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd, requests

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE = Path(__file__).parent.parent
RAW = BASE/"data_store/nse_raw"
HOST = "https://nsearchives.nseindia.com"
UDIFF_FROM = pd.Timestamp("2024-07-08")
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126 Safari/537.36")
_local = threading.local()


def session():
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        _local.s.headers.update({"User-Agent": UA, "Accept": "*/*"})
    return _local.s


def urls(d: pd.Timestamp):
    mon = d.strftime("%b").upper()
    if d < UDIFF_FROM:
        bhav = (f"{HOST}/content/historical/EQUITIES/{d.year}/{mon}/"
                f"cm{d.strftime('%d')}{mon}{d.year}bhav.csv.zip", f"bhav/{d:%Y%m%d}.csv.zip")
    else:
        bhav = (f"{HOST}/content/cm/BhavCopy_NSE_CM_0_0_0_{d:%Y%m%d}_F_0000.csv.zip",
                f"bhav/{d:%Y%m%d}.udiff.csv.zip")
    mto = (f"{HOST}/archives/equities/mto/MTO_{d:%d%m%Y}.DAT", f"mto/{d:%Y%m%d}.DAT")
    ind = (f"{HOST}/content/indices/ind_close_all_{d:%d%m%Y}.csv", f"ind/{d:%Y%m%d}.csv")
    return [bhav, mto, ind]


def fo_urls(d: pd.Timestamp):
    mon = d.strftime("%b").upper()
    if d < UDIFF_FROM:
        return [(f"{HOST}/content/historical/DERIVATIVES/{d.year}/{mon}/"
                 f"fo{d.strftime('%d')}{mon}{d.year}bhav.csv.zip", f"fo/{d:%Y%m%d}.csv.zip")]
    return [(f"{HOST}/content/fo/BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv.zip",
             f"fo/{d:%Y%m%d}.udiff.csv.zip")]


def fetch(job):
    url, rel = job
    out = RAW/rel
    miss = out.with_suffix(out.suffix + ".404")
    if out.exists() or miss.exists():
        return "skip"
    for attempt in range(4):
        try:
            r = session().get(url, timeout=30)
        except requests.RequestException:
            time.sleep(2*(attempt+1)); continue
        if r.status_code == 200 and len(r.content) > 600:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(r.content)
            return "ok"
        if r.status_code == 404:
            day = pd.Timestamp(Path(rel).name[:8])
            if day < pd.Timestamp.today().normalize() - pd.Timedelta(days=5):
                miss.parent.mkdir(parents=True, exist_ok=True)
                miss.touch()                  # remember: no file for this (old) date
            return "404"
        time.sleep(2*(attempt+1))             # 403 / 5xx: back off and retry
    return "fail"


def trading_days(start):
    """Weekdays from start to today (non-trading days simply 404 and are remembered),
    plus weekend special sessions (budget days, muhurat) taken from eod2's calendar."""
    days = pd.bdate_range(start, pd.Timestamp.today().normalize())
    ref = BASE/"data_store/eod2/src/eod2_data/daily/reliance.csv"
    if ref.exists():
        cal = pd.to_datetime(pd.read_csv(ref, usecols=["Date"])["Date"], errors="coerce").dropna()
        days = days.union(pd.DatetimeIndex(cal[(cal >= start) & (cal.dt.weekday >= 5)]))
    return days


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2007-01-01")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--fo", action="store_true", help="fetch F&O bhavcopies instead")
    a = ap.parse_args()
    jobs = [j for d in trading_days(a.start) for j in (fo_urls(d) if a.fo else urls(d))]
    counts = {}
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        for i, res in enumerate(ex.map(fetch, jobs), 1):
            counts[res] = counts.get(res, 0) + 1
            if i % 1000 == 0:
                print(f"{i}/{len(jobs)} {counts} {time.time()-t0:.0f}s", flush=True)
    print("done", counts, f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
