"""smartpiper entry point (PAPER trading -- no orders are ever sent to a broker).

  daily    fetch NSE files -> roll price window + listing-age state -> fill yesterday's
           paper orders at today's close (per-stock costs) -> mark book -> nav.csv.
           On the week's last trading day also runs `weekly`.
  weekly   features for today -> append to store -> score -> decide (the backtest's own
           decide()) -> signal.json/.txt + pending orders for the next close.
  monthly  label rows whose horizon has elapsed -> retrain 3 horizons x 3 seeds.
  status   print book, NAV, last signal.

Run:  python -m smart.run daily|weekly|monthly|status [--no-fetch] [--asof YYYY-MM-DD]
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd

from smart.config import (BASE, BOOK, BOOKS, CAPITAL, CORR_WIN, DATA, HORIZONS, LABELS, LEDGER, LIVE,
                          LOGS, NAV, SIGNAL, STORE)
from smart import window as W

sys.path.insert(0, str(BASE))
PY = sys.executable


def log(msg):
    print(f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} {msg}", flush=True)


# ---------------------------------------------------------------- notify (optional)
def notify(text):
    env = dict(l.strip().split("=", 1) for l in (BASE/".env").read_text().splitlines()
               if "=" in l and not l.startswith("#")) if (BASE/".env").exists() else {}
    tok, chat = env.get("TELEGRAM_BOT_TOKEN"), env.get("TELEGRAM_CHAT_ID")
    if not (tok and chat):
        return
    import requests
    try:
        requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                      json={"chat_id": chat, "text": text[:4000]}, timeout=20)
    except Exception as e:                          # alerts must never break the pipeline
        log(f"telegram failed: {e}")


# ---------------------------------------------------------------- calendar
def holidays():
    p = DATA/"nse_holidays.json"
    return {pd.Timestamp(x) for x in json.loads(p.read_text())} if p.exists() else set()


def is_week_end(d):
    """Last trading day of the ISO week: no trading day left before Saturday."""
    hol = holidays()
    nxt = d + pd.Timedelta(days=1)
    while nxt.weekday() < 5:
        if nxt not in hol:
            return False
        nxt += pd.Timedelta(days=1)
    return True


# ---------------------------------------------------------------- ledger
def load_ledger(book="smart"):
    p = BOOKS[book]["ledger"]
    if not p.exists():
        return {"cash": CAPITAL, "holdings": {}, "pending": None, "start": None, "history": []}
    return json.loads(p.read_text())


def save_ledger(L, book="smart"):
    p = BOOKS[book]["ledger"]
    tmp = p.with_suffix(".tmp"); tmp.write_text(json.dumps(L, indent=1)); tmp.replace(p)


def nav_of(L):
    return L["cash"] + sum(h["value"] for h in L["holdings"].values())


def mark(L, P, day):
    """Grow each holding by its adjusted return since its last mark (split-safe)."""
    C = P["close"].ffill(limit=5)
    for s, h in L["holdings"].items():
        last = pd.Timestamp(h["marked"])
        if s in C.columns and last in C.index and day in C.index:
            p0, p1 = C.at[last, s], C.at[day, s]
            if np.isfinite(p0) and np.isfinite(p1) and p0 > 0:
                h["value"] = float(h["value"]*p1/p0)
                h["marked"] = str(day.date())
                h["px"] = float(P["close"].at[day, s]) if np.isfinite(P["close"].at[day, s]) else h.get("px")
            else:
                h["stale_days"] = h.get("stale_days", 0) + 1
    return L


def fill(L, P, day):
    """Execute pending target weights at `day`'s close with per-stock costs."""
    pend = L.get("pending")
    if not pend or pd.Timestamp(pend["decided"]) >= day:
        return L, []
    from scripts import research_smart_ml as SM
    cost = SM.cost_model(P, CAPITAL, BOOK["top_n"]).loc[day]
    nav = nav_of(L)
    target = pend["target"]
    hold = set(pend.get("hold", []))
    cur = {s: h["value"]/nav for s, h in L["holdings"].items()}
    trades, paid = [], 0.0
    for s in sorted(set(cur) | set(target)):
        if s in hold and s in cur:
            continue
        dw = target.get(s, 0.0) - cur.get(s, 0.0)
        if abs(dw) < 1e-9:
            continue
        c = float(abs(dw)*nav*(cost.get(s, 0.01) if np.isfinite(cost.get(s, np.nan)) else 0.01))
        paid += c
        px = float(P["close"].at[day, s]) if s in P["close"].columns and np.isfinite(P["close"].at[day, s]) else None
        trades.append({"sym": s, "side": "BUY" if dw > 0 else "SELL", "dw": round(dw, 5),
                       "value": round(dw*nav, 2), "px": px, "cost": round(c, 2)})
        if target.get(s, 0.0) <= 1e-9:
            L["holdings"].pop(s, None)
        else:
            h = L["holdings"].setdefault(s, {"value": 0.0, "entry": str(day.date()), "entry_px": px})
            h["value"] = float(target[s]*nav)
            h["marked"] = str(day.date()); h["px"] = px
    L["cash"] = float(nav - sum(h["value"] for h in L["holdings"].values()) - paid)
    L["history"].append({"date": str(day.date()), "decided": pend["decided"], "trades": trades,
                         "cost": round(paid, 2)})
    L["pending"] = None
    if L.get("start") is None:
        L["start"] = str(day.date())
    return L, trades


# ---------------------------------------------------------------- jobs
def fetch():
    start = (pd.Timestamp.today() - pd.Timedelta(days=12)).strftime("%Y-%m-%d")
    for cmd in ([PY, "scripts/fetch_nse_history.py", "--start", start, "--workers", "2"],
                [PY, "scripts/fetch_nse_events.py", "results", "pit", "corpact", "boardmtg"]):
        r = subprocess.run(cmd, cwd=BASE, capture_output=True, text=True, timeout=3600)
        log(f"{Path(cmd[1]).name}: exit {r.returncode} {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ''}")


def daily(a):
    if not a.no_fetch:
        fetch()
    P = W.build(asof=a.asof)
    day = P["close"].index[-1]
    if NAV.exists() and not a.force_weekly:
        last = pd.read_csv(NAV)["date"].iloc[-1]
        if last == str(day.date()):                 # holiday / data not out yet: nothing new
            log(f"daily: no new trading day after {last} (NSE file not out or holiday) - skipped")
            return
    st = W.load_state()
    if pd.Timestamp(st["asof"]) < day:
        st = W.roll_state(st, P); W.save_state(st)
    for book, B in BOOKS.items():
        L = load_ledger(book)
        L = mark(L, P, day)
        L, trades = fill(L, P, day)
        save_ledger(L, book)
        nav = nav_of(L)
        new = not B["nav"].exists()
        with open(B["nav"], "a") as f:
            if new: f.write("date,nav,cash,n\n")
            f.write(f"{day.date()},{nav:.2f},{L['cash']:.2f},{len(L['holdings'])}\n")
        log(f"daily {day.date()} [{book}]: NAV Rs {nav:,.0f} ({nav/CAPITAL - 1:+.2%}), "
            f"{len(L['holdings'])} holdings, {len(trades)} paper fills")
        if trades and book == "smart":
            notify(f"smartpiper filled {len(trades)} paper orders at {day.date()} close. NAV Rs {nav:,.0f}")
    if is_week_end(day) or a.force_weekly:
        weekly(a, P=P, st=st)


def weekly(a, P=None, st=None):
    from smart import features as F, model as M
    from scripts.research_smart_book import decide
    P = W.build(asof=a.asof) if P is None else P
    st = W.load_state() if st is None else st
    day = P["close"].index[-1]
    t0 = time.time()
    X, C, univ, mkt = F.rows(P, pd.DatetimeIndex([day]), W.age_offset(st, P))
    store = pd.read_parquet(STORE)
    if day not in store.index.get_level_values("date"):
        store = pd.concat([store, X[store.columns].astype("float32")]).sort_index()
        store.to_parquet(STORE)
    del store
    from scripts import research_smart_ml as SM
    scores = {"smart": M.score(X),
              "momentum": SM.rank_features(X)["mom_riskadj"].rank(pct=True).droplevel("date")
                          .sort_values(ascending=False)}
    Rres = C.pct_change(fill_method=None).sub(mkt, axis=0).iloc[-CORR_WIN:]
    for book, B in BOOKS.items():
        score = scores[book]
        L = load_ledger(book)
        nav = nav_of(L)
        w = pd.Series({s: h["value"]/nav for s, h in L["holdings"].items()}, dtype=float)
        target = decide(score, w, Rres, **BOOK)
        rank = pd.Series(np.arange(1, len(score) + 1), index=score.index)
        buys = [s for s in target.index if s not in w.index]
        sells = [s for s in w.index if s not in target.index]
        sig = {"book": book, "decided": str(day.date()), "execute": "next trading day close", "nav": round(nav, 2),
               "target": {s: round(float(v), 5) for s, v in target.items()},
               "buys": [{"sym": s, "rank": int(rank[s]), "value": round(float(target[s])*nav)} for s in buys],
               "sells": [{"sym": s, "rank": int(rank.get(s, -1)) if s in rank.index else None} for s in sells],
               "holds": [s for s in target.index if s in w.index],
               "top30": [{"sym": s, "score": round(float(score[s]), 4)} for s in score.index[:30]]}
        B["signal"].write_text(json.dumps(sig, indent=1))
        # names the band left untouched are NOT re-trimmed at the fill either (they drift
        # between decision and fill; trimming them back is exactly the churn the band removes)
        band_hold = [x for x in target.index if x in w.index and target[x] == w[x]]
        L["pending"] = {"decided": sig["decided"], "target": sig["target"], "hold": band_hold}
        save_ledger(L, book)
        txt = (f"{book} weekly signal {day.date()} (paper, execute next close)\n"
               f"BUY ({len(buys)}): " + ", ".join(f"{b['sym']}#{b['rank']}" for b in sig["buys"]) + "\n"
               f"SELL ({len(sells)}): " + ", ".join(sells) + "\n"
               f"HOLD ({len(sig['holds'])}): " + ", ".join(sig["holds"]))
        B["txt"].write_text(txt + "\n")
        log(txt.replace("\n", " | ") + f"  [{time.time()-t0:.0f}s]")
        if book == "smart":
            notify(txt)


def monthly(a):
    from smart import model as M
    from scripts import research_smart_ml as SM
    P = W.build(asof=a.asof)
    store = pd.read_parquet(STORE)
    Y = pd.read_parquet(LABELS).reindex(store.index)
    C = P["close"]
    for h in HORIZONS:
        col = f"y{h}"
        miss = Y[col].isna() & store.index.get_level_values("date").isin(C.index)
        dates = store.index[miss].get_level_values("date").unique()
        y = M.complete_labels(C, dates, h)
        if len(y):
            Y.loc[Y.index.intersection(y.index), col] = y.reindex(Y.index.intersection(y.index)).values
            log(f"labelled {int(y.reindex(Y.index).notna().sum())} new rows for h{h}")
    Y.to_parquet(LABELS)
    del P, C
    M.train(store, Y, log=log)


def status(a):
    for book, B in BOOKS.items():
        L = load_ledger(book); nav = nav_of(L)
        print(f"[{book}] paper NAV Rs {nav:,.0f} ({nav/CAPITAL - 1:+.2%}) since {L.get('start')}, cash Rs {L['cash']:,.0f}")
        for s, h in sorted(L["holdings"].items(), key=lambda x: -x[1]["value"]):
            print(f"  {s:<12} Rs {h['value']:>9,.0f}  since {h['entry']}  marked {h['marked']}")
        if L.get("pending"):
            print(f"  pending orders decided {L['pending']['decided']} ({len(L['pending']['target'])} names)")
        if B["txt"].exists():
            print("  " + B["txt"].read_text().replace("\n", "\n  "))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["daily", "weekly", "monthly", "status"])
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--asof", default=None, help="replay as of a past date (testing)")
    ap.add_argument("--force-weekly", action="store_true")
    a = ap.parse_args()
    LOGS.mkdir(exist_ok=True); LIVE.mkdir(parents=True, exist_ok=True)
    {"daily": daily, "weekly": weekly, "monthly": monthly, "status": status}[a.cmd](a)


if __name__ == "__main__":
    main()
