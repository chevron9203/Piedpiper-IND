"""
SYSTEM HEALTH — continuous validation of the live paper book.

Answers one question every evening: "is anything quietly broken, and is the edge
still behaving like the backtest said it would?" No models, no AI -- just checks
against known-good reference data. Every check returns PASS / WARN / FAIL with the
numbers that produced the verdict, so a red line is actionable rather than vibes.

Four families:
  DATA    — the inputs are intact (this is what caught the 2026-09-11 merged-row
            corruption that silently deleted a trading day from 466 stocks)
  SIGNAL  — the emitted signal is internally consistent and investable
  DRIFT   — live returns are inside the distribution the backtest predicted
  PLUMBING— every scheduled job actually ran

Run:  python scripts/system_health.py            (write status + log)
      python scripts/system_health.py --quiet    (exit code only, for cron alerting)
Exit code 0 = all PASS/WARN, 1 = at least one FAIL.
"""
from __future__ import annotations
import sys, json, re, glob, os
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np, pandas as pd

BASE = Path(__file__).parent.parent
DAILY = BASE/"data_store/eod2/src/eod2_data/daily"
HIST = BASE/"data_store/perf_history.csv"
OUT = BASE/"data_store/health_status.json"
REF = BASE/"data_store/backtest_reference.json"
LOG = BASE/"logs/system_health.log"

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
DATE_MID = re.compile(r"20\d\d-\d\d-\d\d,")


class Checks:
    def __init__(self):
        self.rows = []

    def add(self, family, name, status, detail, **nums):
        self.rows.append(dict(family=family, name=name, status=status,
                              detail=detail, **nums))

    @property
    def failed(self):
        return [r for r in self.rows if r["status"] == FAIL]

    @property
    def warned(self):
        return [r for r in self.rows if r["status"] == WARN]


# ------------------------------------------------------------------ DATA
def check_data(ck):
    files = glob.glob(str(DAILY/"*.csv"))
    if not files:
        ck.add("DATA", "eod2 present", FAIL, "no daily CSVs found"); return
    ck.add("DATA", "eod2 present", PASS, f"{len(files)} daily CSVs", n=len(files))

    # 1. merged-row corruption (a truncated write merges two days into one line;
    #    pandas then silently DROPS the second day when usecols= is used)
    corrupt = []
    for f in files:
        try:
            txt = Path(f).read_text()
        except Exception:
            continue
        for line in txt.splitlines():
            hits = [m for m in DATE_MID.finditer(line)
                    if m.start() > 0 and line[m.start()-1].isdigit()]
            if hits:
                corrupt.append(Path(f).stem); break
    if corrupt:
        ck.add("DATA", "row integrity", FAIL,
               f"{len(corrupt)} files have merged rows (silent missing days): "
               f"{', '.join(sorted(corrupt)[:5])}...", n=len(corrupt))
    else:
        ck.add("DATA", "row integrity", PASS, "no merged rows", n=0)

    # 2. parse integrity under the FULL schema (what eod2's own updater uses)
    bad = []
    for f in files[:4000]:
        try:
            pd.read_csv(f, nrows=5)
        except Exception:
            bad.append(Path(f).stem)
    ck.add("DATA", "parse integrity", FAIL if bad else PASS,
           f"{len(bad)} unparseable" + (f": {bad[:5]}" if bad else ""), n=len(bad))

    # 3. freshness — the benchmark file should be at the last trading day
    ref = DAILY/"nifty 50.csv"
    if ref.exists():
        d = pd.read_csv(ref, usecols=["Date"])["Date"].dropna()
        last = pd.to_datetime(d.iloc[-1])
        lag = np.busday_count(last.date(), datetime.now().date())
        st = PASS if lag <= 1 else (WARN if lag <= 3 else FAIL)
        ck.add("DATA", "freshness", st,
               f"last bar {last:%Y-%m-%d} ({lag} business days behind)", lag=int(lag))

    # 4. universe size stability — a silent shrink changes which stocks can be picked
    n_ok = 0
    for f in files:
        try:
            d = pd.read_csv(f, usecols=["Date","Close","Volume","Series"])
        except Exception:
            continue
        if len(d[d["Series"] == "EQ"]) >= 260: n_ok += 1
    st = PASS if n_ok >= 2000 else (WARN if n_ok >= 1500 else FAIL)
    ck.add("DATA", "universe size", st, f"{n_ok} stocks with >=260 EQ bars", n=n_ok)

    # 5. DuckDB STOCK freshness, measured separately from indices.
    #    A case-sensitivity bug once froze every stock at 2026-08-26 while the two
    #    index rows kept updating -- and ingest's own "date range" line reported the
    #    max across ALL symbols, so the healthy indices masked 200 dead stocks.
    try:
        import duckdb
        con = duckdb.connect(str(BASE/"data_store/piedpiper.duckdb"), read_only=True)
        row = con.execute(
            "select max(dt), count(distinct symbol) from adjusted_ohlcv "
            "where symbol not in ('Nifty 50','India VIX')").fetchone()
        con.close()
        smax, nsym = row[0], row[1]
        lag = int(np.busday_count(smax, datetime.now().date())) if smax else 999
        st = PASS if lag <= 2 else (WARN if lag <= 4 else FAIL)
        ck.add("DATA", "duckdb stock freshness", st,
               f"newest STOCK bar {smax} ({lag} business days behind), {nsym} symbols",
               lag=lag)
    except Exception as e:
        ck.add("DATA", "duckdb stock freshness", WARN, f"could not query: {e!r}")


# ---------------------------------------------------------------- SIGNAL
def check_signal(ck):
    for tag, path in [("S1", "live_signal.json"),
                      ("S4", "live_signal_s4.json"),
                      ("S5", "live_signal_s5.json")]:
        p = BASE/"data_store"/path
        if not p.exists():
            ck.add("SIGNAL", f"{tag} exists", WARN, f"{path} missing"); continue
        sig = json.load(open(p))
        # weights must sum to 1
        tot = sum(sig.get("allocation", {}).values()) + sig.get("cash", 0.0)
        st = PASS if abs(tot-1.0) < 1e-6 else FAIL
        ck.add("SIGNAL", f"{tag} weights", st, f"alloc+cash = {tot:.4f}", total=round(tot, 6))
        # as_of should be the previous calendar month-end
        asof = pd.to_datetime(sig.get("as_of")) if sig.get("as_of") else None
        if asof is not None:
            exp = (pd.Timestamp.now().normalize().replace(day=1) - pd.Timedelta(days=1))
            stale_m = (exp.to_period("M") - asof.to_period("M")).n
            st = PASS if stale_m == 0 else (WARN if stale_m == 1 else FAIL)
            ck.add("SIGNAL", f"{tag} as_of", st,
                   f"as_of {asof:%Y-%m-%d}, expected month-end {exp:%Y-%m-%d} "
                   f"({stale_m} month(s) stale)", stale_months=int(stale_m))
        # every pick needs an entry price
        picks = sig.get("momentum_picks", []); ent = sig.get("entry_prices", {})
        if picks:
            miss = [s for s in picks if s not in ent]
            ck.add("SIGNAL", f"{tag} entry prices", FAIL if miss else PASS,
                   f"{len(picks)} picks, {len(miss)} missing entry price"
                   + (f": {miss[:5]}" if miss else ""), n_missing=len(miss))


# ----------------------------------------------------------------- DRIFT
def build_reference(monthly_returns, label="pure_momentum"):
    """Freeze the backtest's monthly return distribution as the comparison baseline."""
    r = pd.Series(monthly_returns).dropna()
    ref = dict(label=label, n=len(r), mean=float(r.mean()), sd=float(r.std()),
               p05=float(r.quantile(0.05)), p50=float(r.quantile(0.50)),
               p95=float(r.quantile(0.95)),
               worst_month=float(r.min()), best_month=float(r.max()),
               built=str(datetime.now()))
    json.dump(ref, open(REF, "w"), indent=2)
    return ref


def _live_start(nav):
    """First row where this system's track actually begins.

    S4/S5 were seeded flat at CAP for rows predating their launch; measuring from
    row 0 would score weeks of synthetic zero-return as real performance."""
    v = nav.values
    i = 0
    while i+1 < len(v) and v[i+1] == v[0]:
        i += 1
    return i


def check_drift(ck):
    if not REF.exists():
        ck.add("DRIFT", "reference", WARN,
               "no backtest_reference.json — run scripts/build_health_reference.py")
        return
    raw = json.load(open(REF))
    # per-system refs; tolerate the older flat single-distribution file
    systems = raw.get("systems") or {"s1": raw}
    if not HIST.exists():
        ck.add("DRIFT", "live history", WARN, "no perf_history.csv yet"); return
    h = pd.read_csv(HIST); h["date"] = pd.to_datetime(h["date"])
    if len(h) < 2:
        ck.add("DRIFT", "live history", WARN, f"only {len(h)} rows — need more data"); return

    for key, ref in systems.items():
        col = f"{key}_nav"
        if col not in h.columns: continue
        nav = h[col].astype(float)
        i0 = _live_start(nav)
        seg = nav.iloc[i0:]
        days = len(seg)-1
        if days < 2:
            ck.add("DRIFT", f"{key.upper()} return vs backtest", WARN,
                   f"only {days} tracked days since launch"); continue
        live = seg.iloc[-1]/seg.iloc[0] - 1
        mo = days/21.0
        exp_mu = ref["mean"]*mo
        exp_sd = ref["sd"]*np.sqrt(max(mo, 1e-9))
        z = (live-exp_mu)/exp_sd if exp_sd > 0 else 0.0
        if days < 63:
            st = WARN if z < -2.5 else PASS
            note = f"{days}d live — statistically inconclusive"
        else:
            st = PASS if z > -1.65 else (WARN if z > -2.33 else FAIL)
            note = ""
        ck.add("DRIFT", f"{key.upper()} return vs backtest", st,
               f"live {live*100:+.2f}% over {days}d vs expected {exp_mu*100:+.2f}% "
               f"(sd {exp_sd*100:.2f}pp), z={z:+.2f}. {note}".strip(),
               z=round(float(z), 3), live_pct=round(live*100, 3))

        # drawdown against THIS system's own worst backtested month
        dd = float((seg/seg.cummax()-1).min())
        lim = ref.get("worst_month", -0.20)
        st = PASS if dd > lim else (WARN if dd > lim*1.5 else FAIL)
        ck.add("DRIFT", f"{key.upper()} drawdown vs backtest", st,
               f"live MaxDD {dd*100:.2f}% vs worst backtest month {lim*100:.2f}%",
               dd_pct=round(dd*100, 3))

    # benchmark relative — the whole point of the book is to beat the index
    rel = (h["s1_nav"].iloc[-1]/h["s1_nav"].iloc[0]) - (h["nifty50_nav"].iloc[-1]/h["nifty50_nav"].iloc[0])
    ck.add("DRIFT", "S1 vs Nifty 50", PASS,
           f"{rel*100:+.2f}pp since inception ({len(h)} days)", rel_pp=round(rel*100, 3))


# -------------------------------------------------------------- PLUMBING
def check_plumbing(ck):
    """Every scheduled job should have touched its log recently."""
    jobs = [("eod2_update.log", 1), ("ingest_data.log", 1),
            ("momentum_monitor.log", 1), ("perf_tracker.log", 1)]
    for fn, max_bd in jobs:
        p = BASE/"logs"/fn
        if not p.exists():
            ck.add("PLUMBING", fn, FAIL, "log missing — job never ran"); continue
        age_d = (datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)).days
        lag = np.busday_count((datetime.now()-timedelta(days=age_d)).date(),
                              datetime.now().date())
        st = PASS if lag <= max_bd else (WARN if lag <= 3 else FAIL)
        ck.add("PLUMBING", fn, st, f"last write {age_d}d ago ({lag} business days)",
               lag=int(lag))

    # perf_history must not skip trading days
    if HIST.exists():
        h = pd.read_csv(HIST); d = pd.to_datetime(h["date"])
        if len(d) > 1:
            span = np.busday_count(d.iloc[0].date(), d.iloc[-1].date()) + 1
            try:
                hol = set(json.load(open(BASE/"data_store/nse_holidays.json")))
            except Exception:
                hol = set()
            nhol = sum(1 for x in hol
                       if d.iloc[0] <= pd.Timestamp(x) <= d.iloc[-1]
                       and pd.Timestamp(x).weekday() < 5)
            expected = span - nhol
            gap = expected - len(d)
            st = PASS if gap <= 0 else (WARN if gap <= 2 else FAIL)
            ck.add("PLUMBING", "perf_history continuity", st,
                   f"{len(d)} rows vs {expected} expected trading days (gap {gap})",
                   gap=int(gap))


# ------------------------------------------------------------------ main
def main():
    quiet = "--quiet" in sys.argv
    ck = Checks()
    for fn in (check_data, check_signal, check_drift, check_plumbing):
        try:
            fn(ck)
        except Exception as e:
            ck.add(fn.__name__.replace("check_", "").upper(), "checker crashed", FAIL, repr(e))

    overall = FAIL if ck.failed else (WARN if ck.warned else PASS)
    payload = dict(checked=str(datetime.now()), overall=overall, checks=ck.rows)
    json.dump(payload, open(OUT, "w"), indent=2)

    stamp = f"{datetime.now():%Y-%m-%d %H:%M}"
    lines = [f"{stamp} | HEALTH {overall} "
             f"({len(ck.failed)} fail, {len(ck.warned)} warn, {len(ck.rows)} checks)"]
    for r in ck.rows:
        if r["status"] != PASS or not quiet:
            lines.append(f"{stamp} |   [{r['status']}] {r['family']}/{r['name']}: {r['detail']}")
    out = "\n".join(lines)
    if not quiet:
        print(out)
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a") as f:
        f.write(out + "\n")
    sys.exit(1 if ck.failed else 0)


if __name__ == "__main__":
    main()
