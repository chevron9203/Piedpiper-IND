"""Real-money order ticket for the smart book (MANUAL execution - the system never places orders).

The paper book is the model. For real money the ticket re-derives the trades from YOUR actual holdings:
  * target = equal weight over the signal's target names (1/N of `capital` each), whole shares
  * a name you already hold is left alone while its weight is within +-25% of target (the backtest's band: no tiny trims)
  * names that left the target are sold in full; new names are bought
Prices are the last close; the backtest assumes a fill at the NEXT trading day's close, so place orders in the closing
window (15:00-15:30 IST, NSE's close is the VWAP of the last 30 minutes).

Files (data_store/smart_live/): real_capital.json, real_holdings.json, real_trades.jsonl, ticket.json, ticket.txt
CLI:  python -m smart.run capital 200000 | ticket [--capital X] | record BUY|SELL SYM QTY PRICE [--date YYYY-MM-DD]
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd

from smart.config import BASE, BOOKS, LIVE

sys.path.insert(0, str(BASE))
REAL_CAP = LIVE/"real_capital.json"
REAL_HOLD = LIVE/"real_holdings.json"
REAL_LOG = LIVE/"real_trades.jsonl"
TICKET = LIVE/"ticket.json"
TICKET_TXT = LIVE/"ticket.txt"
BAND = 0.25
DEFAULT_CAPITAL = 200_000.0


def capital() -> float:
    try:
        return float(json.loads(REAL_CAP.read_text())["capital"])
    except Exception:
        return DEFAULT_CAPITAL


def set_capital(x: float):
    LIVE.mkdir(parents=True, exist_ok=True)
    REAL_CAP.write_text(json.dumps({"capital": float(x), "set": str(pd.Timestamp.now().date())}))


def holdings() -> dict:
    try:
        return json.loads(REAL_HOLD.read_text())
    except Exception:
        return {}


def record(side: str, sym: str, qty: int, price: float, date: str | None = None):
    """Record a fill YOU made. BUY adds to the holding (average cost), SELL reduces it."""
    side = side.upper(); sym = sym.upper().strip()
    assert side in ("BUY", "SELL") and qty > 0 and price > 0, "usage: record BUY|SELL SYMBOL QTY PRICE"
    H = holdings(); h = H.get(sym, {"qty": 0, "avg": 0.0, "since": date or str(pd.Timestamp.now().date())})
    if side == "BUY":
        h["avg"] = (h["avg"]*h["qty"] + price*qty)/(h["qty"] + qty); h["qty"] += qty
    else:
        assert qty <= h["qty"], f"cannot sell {qty} {sym}: you hold {h['qty']}"
        h["qty"] -= qty
    if h["qty"] == 0:
        H.pop(sym, None)
    else:
        H[sym] = h
    LIVE.mkdir(parents=True, exist_ok=True)
    REAL_HOLD.write_text(json.dumps(H, indent=1))
    with open(REAL_LOG, "a") as f:
        f.write(json.dumps({"date": date or str(pd.Timestamp.now().date()), "side": side, "sym": sym, "qty": qty, "price": price}) + "\n")
    return H


def make_ticket(P, cap: float | None = None, book: str = "smart"):
    """Order list from the latest weekly signal and your recorded holdings. P = price window (smart.window.build)."""
    from scripts import research_smart_ml as SM
    sig = json.loads(BOOKS[book]["signal"].read_text())
    cap = float(cap if cap is not None else capital())
    day = P["close"].index[-1]
    px = SM.traded_price(P).ffill(limit=5).iloc[-1]
    turn = P["turn"].rolling(60, min_periods=30).median().iloc[-1]
    target = list(sig["target"]); n = len(target); w_t = 1.0/n
    rank = {x["sym"]: x["rank"] for x in sig.get("buys", [])}
    H = holdings(); lines, warn, skipped = [], [], []
    for s in sorted(set(target) | set(H)):
        p = float(px.get(s, np.nan)) if s in px.index else float("nan")
        if not np.isfinite(p) or p <= 0:
            warn.append(f"{s}: no recent price - check manually")
            continue
        cur = int(H.get(s, {}).get("qty", 0))
        tgt = int(math.floor(w_t*cap/p)) if s in target else 0
        cur_w = cur*p/cap
        if s not in target and cur > 0:
            act, q, why = "SELL", cur, "left the top-100 ranking: sell all"
        elif s in target and cur == 0:
            act, q, why = "BUY", tgt, f"new entry (rank #{rank[s]})" if s in rank else "new entry"
            if tgt == 0:
                skipped.append((s, p))
                continue
        elif s in target and abs(cur_w - w_t) > BAND*w_t and tgt != cur:
            act, q, why = ("BUY", tgt - cur, "top-up (weight fell >25% below target)") if tgt > cur else ("SELL", cur - tgt, "trim (weight >25% above target)")
        else:
            continue
        val = q*p
        liq = val/max(float(turn.get(s, np.nan)), 1.0) if s in turn.index and np.isfinite(turn.get(s, np.nan)) else float("nan")
        lines.append({"action": act, "sym": s, "qty": int(q), "ref_price": round(p, 2), "value": round(val),
                      "pct_of_daily_volume": round(liq*100, 4) if np.isfinite(liq) else None, "why": why})
        if np.isfinite(liq) and liq > 0.01:
            warn.append(f"{s}: order is {liq*100:.1f}% of a normal day's traded value - split it over two days")
    # a target stock whose single share costs more than a slot cannot be bought: take the next-ranked affordable name
    used = set(target) | set(H)
    for s, p in skipped:
        rep = None
        for c in sig.get("top30", []):
            cs = c["sym"]; cp = float(px.get(cs, np.nan)) if cs in px.index else float("nan")
            if cs not in used and np.isfinite(cp) and cp > 0 and math.floor(w_t*cap/cp) >= 1:
                rep = (cs, cp); break
        if rep:
            used.add(rep[0]); q = int(math.floor(w_t*cap/rep[1]))
            lines.append({"action": "BUY", "sym": rep[0], "qty": q, "ref_price": round(rep[1], 2), "value": round(q*rep[1]),
                          "pct_of_daily_volume": None, "why": f"replaces {s} (one share costs Rs {p:,.0f}, more than your Rs {w_t*cap:,.0f} slot)"})
            warn.append(f"{s} skipped: one share (Rs {p:,.0f}) is bigger than a Rs {w_t*cap:,.0f} slot; ticket buys {rep[0]} instead. "
                        f"A larger capital (Rs 5 lakh+) avoids this.")
        else:
            warn.append(f"{s} skipped: one share (Rs {p:,.0f}) is bigger than a Rs {w_t*cap:,.0f} slot and no affordable replacement found")
    lines.sort(key=lambda r: (r["action"] != "SELL", -r["value"]))
    buys = sum(r["value"] for r in lines if r["action"] == "BUY"); sells = sum(r["value"] for r in lines if r["action"] == "SELL")
    out = {"decided": sig["decided"], "price_date": str(day.date()), "capital": cap, "slot": round(w_t*cap), "names": n,
           "lines": lines, "warnings": warn, "buy_value": round(buys), "sell_value": round(sells),
           "how": "Place at the NEXT trading day's close window (15:00-15:30 IST), limit within +0.5% of the reference price. "
                  "Quantities are sized on the last close: if the live price has moved more than ~3%, change the quantity so the order value is about the slot size. "
                  "If a stock is locked at the upper circuit (cannot buy) skip it and buy the next-ranked name; if it is locked "
                  "at the lower circuit (cannot sell) sell it the next day. After each fill run: python -m smart.run record BUY|SELL SYMBOL QTY PRICE"}
    TICKET.write_text(json.dumps(out, indent=1))
    txt = [f"ORDER TICKET  decision {out['decided']}  prices as of {out['price_date']}  capital Rs {cap:,.0f}  ({n} names, Rs {w_t*cap:,.0f} each)"]
    txt += [f"  {r['action']:<4} {r['sym']:<12} {r['qty']:>5} sh  ~Rs {r['ref_price']:>9,.2f}  = Rs {r['value']:>8,.0f}   {r['why']}" for r in lines] or ["  (nothing to trade this week)"]
    txt += [f"  total buys Rs {buys:,.0f} | sells Rs {sells:,.0f}"] + [f"  WARNING {w}" for w in warn] + ["  " + out["how"]]
    TICKET_TXT.write_text("\n".join(txt) + "\n")
    return out
