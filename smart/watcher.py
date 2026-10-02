"""Announcements watcher (long-running; systemd smartpiper-watcher).

Research finding this is built on: GOOD news is priced the same day (chasing it loses),
BAD news drifts for weeks (pledge -5%, default -5%, downgrade -8% over 3 months). So the
watcher's job is to tell you, the day it happens, when something material hits a stock
you HOLD -- and to flag red flags on the next names in line.

  market hours (09:00-17:30 IST, trading days): poll NSE every POLL seconds
  holdings   any non-routine filing -> LLM one-liner (Groq) -> alert
  watchlist  red-flag types only -> alert
  17:30      digest of the day's holding news
Output: smart_live/alerts.jsonl (+ Telegram if TELEGRAM_* are in .env). Paper only.
"""
from __future__ import annotations
import json, re, sys, time
import pandas as pd, requests

from smart.config import ALERTS, BASE, LEDGER, LIVE, SIGNAL

sys.path.insert(0, str(BASE))
from scripts.fetch_nse_events import API, get, new_session      # noqa: E402
from scripts.research_smart_ann import classify, amount          # noqa: E402
from smart.run import holidays, log, notify                       # noqa: E402

POLL = 180
RED = {"pledge", "default", "rating_down", "regulatory", "key_exit", "auditor_exit",
       "clarify", "fund_raise", "bonus_split"}
ROUTINE = re.compile(r"trading window|newspaper|book closure|record date|agm|annual general|"
                     r"compliance certificate|loss of share|duplicate share|investor grievance|"
                     r"regulation 74|shareholding pattern|esop|allotment of esop", re.I)
LLM_MODEL = "qwen/qwen3.8-27b"
PROMPT = ("You are an equity analyst. An Indian listed company ({sym}) filed this with NSE:\n"
          "Category: {desc}\nText: {text}\n\nReply JSON: {{\"summary\": <one line, plain English, "
          "what happened and why it matters>, \"sentiment\": <-2..2>, \"material\": <0 or 1>, "
          "\"risk\": <one short phrase or empty>}}")


def env():
    p = BASE/".env"
    return dict(l.strip().split("=", 1) for l in p.read_text().splitlines()
                if "=" in l and not l.startswith("#")) if p.exists() else {}


def llm(sym, desc, text, key):
    if not key:
        return {}
    body = {"model": LLM_MODEL, "temperature": 0, "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": PROMPT.format(sym=sym, desc=desc, text=text[:1500])}]}
    for k in range(3):
        try:
            r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                              headers={"Authorization": f"Bearer {key}"}, json=body, timeout=60)
            if r.status_code == 429:
                time.sleep(float(r.headers.get("retry-after", 20)) + 1); continue
            if r.ok:
                return json.loads(r.json()["choices"][0]["message"]["content"])
        except Exception:
            pass
        time.sleep(5*(k + 1))
    return {}


def books():
    held = set(json.loads(LEDGER.read_text())["holdings"]) if LEDGER.exists() else set()
    watch = {x["sym"] for x in json.loads(SIGNAL.read_text()).get("top30", [])} if SIGNAL.exists() else set()
    return held, watch - held


def in_hours(now, hol):
    return (now.weekday() < 5 and now.normalize() not in hol
            and (9, 0) <= (now.hour, now.minute) <= (17, 30))


def run_once(s, seen, key, digest):
    today = pd.Timestamp.now()
    rows = get(s, API + f"/corporate-announcements?index=equities&from_date={today:%d-%m-%Y}&to_date={today:%d-%m-%Y}") or []
    held, watch = books()
    new = 0
    for x in rows:
        sid = str(x.get("seq_id") or (x.get("symbol"), x.get("an_dt"), x.get("desc")))
        body = (x.get("symbol"), x.get("desc"), (x.get("attchmntText") or "")[:200])
        if sid in seen or body in seen:             # NSE sometimes posts the same filing twice
            continue
        seen.add(sid); seen.add(body); new += 1
        sym = str(x.get("symbol", "")).strip()
        desc, text = x.get("desc") or "", x.get("attchmntText") or ""
        typ = classify(desc, text)
        if sym in held:
            if typ in ("other", "guidance") and ROUTINE.search(f"{desc} {text}"):
                continue
        elif not (sym in watch and typ in RED):
            continue
        a = {"ts": x.get("sort_date") or x.get("an_dt"), "sym": sym, "held": sym in held,
             "type": typ, "red_flag": typ in RED, "desc": desc, "text": text[:600],
             "amount_rs": amount(text) if typ == "order_win" else None, "link": x.get("attchmntFile")}
        a.update({f"llm_{k}": v for k, v in llm(sym, desc, text, key).items()})
        with open(ALERTS, "a") as f:
            f.write(json.dumps(a, default=str) + "\n")
        tag = "RED FLAG" if a["red_flag"] else ("held" if a["held"] else "watch")
        line = f"[{tag}] {sym}: {a.get('llm_summary') or desc}"
        log(line)
        if a["held"]:
            digest.append(line)
        if a["red_flag"] or (a["held"] and int(a.get("llm_material") or 0) == 1):
            notify("smartpiper " + line + (f"\n{a['link']}" if a.get("link") else ""))
    return new


def main():
    LIVE.mkdir(parents=True, exist_ok=True)
    key = env().get("GROQ_API_KEY")
    log(f"watcher up (poll {POLL}s, LLM {'on' if key else 'OFF - no GROQ_API_KEY'})")
    s, seen, day, digest, sent = new_session(), set(), None, [], False
    hol = holidays()
    while True:
        now = pd.Timestamp.now()
        if now.date() != day:                       # new day: reset de-dupe + digest
            day, seen, digest, sent, s = now.date(), set(), [], False, new_session()
        if in_hours(now, hol):
            try:
                n = run_once(s, seen, key, digest)
                if n:
                    log(f"{n} new announcements checked")
            except Exception as e:                  # never die on a bad poll
                log(f"poll error: {e}"); s = new_session()
            time.sleep(POLL)
        else:
            if not sent and (now.hour, now.minute) >= (17, 30) and now.weekday() < 5:
                notify("smartpiper daily digest (holdings):\n" + ("\n".join(digest) or "no material news"))
                sent = True
            time.sleep(300)


if __name__ == "__main__":
    main()
