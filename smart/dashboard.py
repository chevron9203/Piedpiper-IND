"""smartpiper dashboard: one server-rendered page (no JavaScript), localhost only.

Reads the files the service writes -- never writes anything. Separate from piedpiper's
dashboards (5001, 5002); this one is 127.0.0.1:5010.

Run:  python -m smart.dashboard            (systemd: smartpiper-dashboard)
View: ssh -L 5010:localhost:5010 ubuntu@<box>  then  http://localhost:5010
"""
from __future__ import annotations
import html, json
from http.server import BaseHTTPRequestHandler, HTTPServer

import pandas as pd

from smart.config import ALERTS, BOOKS, CAPITAL, DATA, LEDGER, LIVE, NAV, SCORECARD, SIGNAL
from smart.ticket import REAL_HOLD, TICKET

PORT = 5010          # 5001 = piedpiper dashboard, 5002 = piedpiper paper_dashboard
E = html.escape


def _read_json(p, default):
    try:
        return json.loads(p.read_text())
    except Exception:
        return default


def nifty500(start):
    """Nifty 500 closes from the raw ind_close_all files (same source as the research)."""
    out = {}
    for p in sorted((DATA/"nse_raw/ind").glob("*.csv")):
        d = pd.Timestamp(p.name[:8])
        if d < start:
            continue
        try:
            x = pd.read_csv(p)
            x.columns = [c.strip() for c in x.columns]
            r = x[x["Index Name"].str.strip().str.upper() == "NIFTY 500"]
            if len(r):
                out[d] = float(r["Closing Index Value"].iloc[0])
        except Exception:
            continue
    return pd.Series(out, dtype=float)


def chart(nav, bench, w=760, h=220, pad=34, others=None):
    if len(nav) < 2:
        return "<p class=muted>NAV chart appears after the first two trading days.</p>"
    others = {k: (v.reindex(nav.index).ffill()/v.reindex(nav.index).ffill().dropna().iloc[0] - 1)*100
              for k, v in (others or {}).items() if len(v.dropna()) > 1}
    s = (nav/nav.iloc[0] - 1)*100
    b = (bench.reindex(nav.index).ffill()/bench.reindex(nav.index).ffill().dropna().iloc[0] - 1)*100 \
        if len(bench.dropna()) else None
    vals = pd.concat([s] + ([b] if b is not None else []) + list(others.values()))
    lo, hi = min(vals.min(), 0), max(vals.max(), 0)
    hi = hi if hi > lo else lo + 1
    n = len(s)
    X = lambda i: pad + i*(w - 2*pad)/max(n - 1, 1)
    Y = lambda v: h - pad - (v - lo)*(h - 2*pad)/(hi - lo)
    def path(series):
        return " ".join(f"{'M' if i == 0 else 'L'}{X(i):.1f},{Y(v):.1f}"
                        for i, v in enumerate(series.values) if pd.notna(v))
    g = [f'<line x1="{pad}" x2="{w-pad}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" class="zero"/>']
    for v in (lo, hi):
        g.append(f'<text x="4" y="{Y(v)+4:.1f}" class="ax">{v:+.1f}%</text>')
    g.append(f'<text x="{pad}" y="{h-8}" class="ax">{s.index[0]:%d %b}</text>')
    g.append(f'<text x="{w-pad}" y="{h-8}" class="ax" text-anchor="end">{s.index[-1]:%d %b %Y}</text>')
    if b is not None:
        g.append(f'<path d="{path(b)}" class="bench"/>')
    for n_, v in enumerate(others.values()):
        g.append(f'<path d="{path(v)}" class="{"nav2" if n_ == 0 else "nav3"}"/>')
    g.append(f'<path d="{path(s)}" class="nav"/>')
    leg = "".join(f'&nbsp; <span class="k {"nav2" if n_ == 0 else "nav3"}"></span>{E(k)}' for n_, k in enumerate(others))
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Paper NAV vs Nifty 500">{"".join(g)}</svg>'
            f'<div class=legend><span class="k nav"></span>smart paper {leg} &nbsp; '
            f'<span class="k bench"></span>Nifty 500</div>')


def page():
    L = _read_json(LEDGER, {"cash": CAPITAL, "holdings": {}, "pending": None, "history": []})
    sig = _read_json(SIGNAL, {})
    navdf = pd.read_csv(NAV, parse_dates=["date"]).drop_duplicates("date", keep="last").set_index("date") \
        if NAV.exists() else pd.DataFrame(columns=["nav"])
    nav = navdf["nav"] if len(navdf) else pd.Series(dtype=float)
    total = L["cash"] + sum(h["value"] for h in L["holdings"].values())
    ret = total/CAPITAL - 1
    bench = nifty500(nav.index[0]) if len(nav) else pd.Series(dtype=float)
    bret = (bench.iloc[-1]/bench.iloc[0] - 1) if len(bench) > 1 else None
    rows = []
    for s, h in sorted(L["holdings"].items(), key=lambda x: -x[1]["value"]):
        cost_basis = None
        for ev in L.get("history", []):
            for t in ev["trades"]:
                if t["sym"] == s and t["side"] == "BUY":
                    cost_basis = (cost_basis or 0) + t["value"]
        pnl = (h["value"] - cost_basis) if cost_basis else None
        cls = "" if pnl is None else ("up" if pnl >= 0 else "dn")
        rows.append(f"<tr><td>{E(s)}</td><td class=num>₹{h['value']:,.0f}</td>"
                    f"<td class='num {cls}'>{'' if pnl is None else ('+' if pnl >= 0 else '−') + f'₹{abs(pnl):,.0f}'}</td>"
                    f"<td>{E(h.get('entry', ''))}</td></tr>")
    hold = "".join(rows) or "<tr><td colspan=4 class=muted>No holdings yet — first paper fill pending.</td></tr>"
    pend = L.get("pending")
    pend_html = (f"<p><b>Pending paper orders</b> decided {E(pend['decided'])}, fill at the next trading "
                 f"close ({len(pend['target'])} names in target).</p>") if pend else ""
    def chips(items, cls):
        return " ".join(f"<span class='chip {cls}'>{E(x['sym'] if isinstance(x, dict) else x)}"
                        f"{'' if not isinstance(x, dict) or x.get('rank') is None else ' #' + str(x['rank'])}</span>"
                        for x in items) or "<span class=muted>none</span>"
    sig_html = (f"<p class=muted>Decided {E(sig.get('decided', '—'))} · executes {E(sig.get('execute', ''))}</p>"
                f"<p><b>Buy</b> {chips(sig.get('buys', []), 'buy')}</p>"
                f"<p><b>Sell</b> {chips(sig.get('sells', []), 'sell')}</p>"
                f"<p><b>Hold</b> {chips(sig.get('holds', []), '')}</p>") if sig else "<p class=muted>No signal yet.</p>"
    alerts = []
    if ALERTS.exists():
        for line in ALERTS.read_text().splitlines()[-40:][::-1]:
            try:
                a = json.loads(line)
            except ValueError:
                continue
            tag = "RED FLAG" if a.get("red_flag") else ("held" if a.get("held") else "watch")
            link = f" <a href='{E(a['link'])}' target=_blank rel=noopener>filing</a>" if a.get("link") else ""
            alerts.append(f"<li><span class='chip {'sell' if a.get('red_flag') else ''}'>{tag}</span> "
                          f"<b>{E(a.get('sym', ''))}</b> <span class=muted>{E(str(a.get('ts', ''))[:16])}</span><br>"
                          f"{E(a.get('llm_summary') or a.get('desc') or '')}{link}</li>")
    al = "".join(alerts) or "<li class=muted>No alerts yet — the watcher runs 09:00–17:30 on trading days.</li>"
    cmp_navs, cmp_html = {}, ""
    notes = {"momentum": "pure momentum score", "ml": "ML picks only, no momentum"}
    for key, title in (("momentum", "momentum only"), ("ml", "ML picks only")):
        MB = BOOKS[key]
        Lm = _read_json(MB["ledger"], {"cash": CAPITAL, "holdings": {}, "pending": None})
        navm = pd.read_csv(MB["nav"], parse_dates=["date"]).drop_duplicates("date", keep="last").set_index("date")["nav"] \
            if MB["nav"].exists() else pd.Series(dtype=float)
        cmp_navs[f"{key} book"] = navm
        totm = Lm["cash"] + sum(h["value"] for h in Lm["holdings"].values())
        sigm = _read_json(MB["signal"], {})
        cmp_html += (f"<div class=card><h2>Comparison book: {title}</h2>"
                     f"<div class=muted>Same data, rules and costs — only the stock score differs ({notes[key]}). "
                     f"Here to show, live, which approach earns more in today's market.</div>"
                     f"<div class='big {'up' if totm >= CAPITAL else 'dn'}' style='font-size:22px'>₹{totm:,.0f} "
                     f"<span style='font-size:15px'>{totm/CAPITAL - 1:+.2%}</span></div>"
                     f"<p><b>Holdings</b> {chips(sorted(Lm['holdings']), '')}</p>"
                     f"<p class=muted>Latest signal {E(sigm.get('decided', '—'))}: buy {chips(sigm.get('buys', []), 'buy')} "
                     f"sell {chips(sigm.get('sells', []), 'sell')}</p></div>")
    tk = _read_json(TICKET, {})
    # ---- guardrails: pre-committed pause rules (see RUNBOOK.md)
    g_state, g_msgs = "GO", []
    def _worse(cur, new):
        return new if ["GO", "REVIEW", "PAUSE"].index(new) > ["GO", "REVIEW", "PAUSE"].index(cur) else cur
    if len(nav):
        dd_now = float(nav.iloc[-1]/nav.max() - 1)
        age = (pd.Timestamp.now().normalize() - nav.index[-1]).days
        if age > 5:
            g_state = _worse(g_state, "PAUSE"); g_msgs.append(f"data stale: last daily update {nav.index[-1]:%d %b} ({age} days ago) — do not trade on an old signal")
        if dd_now <= -0.35:
            g_state = _worse(g_state, "PAUSE"); g_msgs.append(f"smart book is {dd_now:.0%} from its peak (backtest worst −36%)")
        elif dd_now <= -0.25:
            g_state = _worse(g_state, "REVIEW"); g_msgs.append(f"smart book is {dd_now:.0%} from its peak")
    else:
        g_msgs.append("no paper NAV history yet (first fills Monday)")
    _scd = _read_json(SCORECARD, {})
    for r in _scd.get("summary", []):
        if r["score"] == "blend" and r["h"] == 21 and r["n"] >= 12 and r["excess"] < 0:
            g_state = _worse(g_state, "REVIEW"); g_msgs.append(f"edge check: top-20 21-day excess {r['excess']*100:+.2f}% over {r['n']} decisions (backtest ≈ +2.2%)")
    g_cls = {"GO": "up", "REVIEW": "", "PAUSE": "dn"}[g_state]
    g_html = (f"<p><b>Guardrails: <span class='{g_cls}'>{g_state}</span></b>"
              + (" — " + "; ".join(E(m) for m in g_msgs) if g_msgs else " — all pre-committed checks pass") + "</p>")
    if tk.get("lines") is not None:
        def tr(r):
            c = "up" if r["action"] == "BUY" else "dn"
            return (f"<tr><td class={c}><b>{r['action']}</b></td><td>{E(r['sym'])}</td><td class=num>{r['qty']}</td>"
                    f"<td class=num>₹{r['ref_price']:,.2f}</td><td class=num>₹{r['value']:,.0f}</td><td class=muted>{E(r['why'])}</td></tr>")
        rows_t = "".join(tr(r) for r in tk["lines"]) or "<tr><td colspan=6 class=muted>Nothing to trade this week.</td></tr>"
        warns = "".join(f"<li>{E(w)}</li>" for w in tk.get("warnings", []))
        rh = _read_json(REAL_HOLD, {})
        held = ", ".join(f"{k} {v['qty']}" for k, v in sorted(rh.items())) or "none recorded yet"
        tk_html = (f"<div class=card><h2>Order ticket <span class=muted style='font-weight:400'>(optional — only if you ever trade this for real; you are on PAPER)</span></h2><p class=muted>Decision {E(tk['decided'])} · prices as of {E(tk['price_date'])} · "
                   f"capital ₹{tk['capital']:,.0f} ({tk['names']} names, ₹{tk['slot']:,.0f} each). Nothing to do while paper trading — the paper books fill themselves. Guardrails below still apply to the paper smart book.</p>{g_html}"
                   f"<table><tr><th></th><th>Stock</th><th class=num>Qty</th><th class=num>Ref price</th><th class=num>Value</th><th>Why</th></tr>{rows_t}</table>"
                   f"<p>Buys ₹{tk['buy_value']:,.0f} · sells ₹{tk['sell_value']:,.0f}</p>"
                   + (f"<ul>{warns}</ul>" if warns else "") +
                   f"<p class=muted>{E(tk['how'])}</p><p class=muted>Your recorded holdings: {E(held)}</p></div>")
    else:
        tk_html = "<div class=card><h2>Order ticket <span class=muted style='font-weight:400'>(optional — paper mode)</span></h2><p class=muted>Appears after the next weekly decision.</p></div>"
    scd = _read_json(SCORECARD, {})
    if scd.get("summary"):
        ref = scd.get("ref", {})
        trs = ""
        for r in sorted(scd["summary"], key=lambda r: (r["h"], ["blend", "ML", "momentum"].index(r["score"]))):
            cls = "up" if r["excess"] > 0 else "dn"
            trs += (f"<tr><td>{r['h']}d</td><td>{E(r['score'])}</td><td class=num>{r['n']}</td>"
                    f"<td class='num {cls}'>{r['excess']*100:+.2f}%</td><td class=num>{r['hit']:.0%}</td><td class=num>{r['ic']:+.3f}</td></tr>")
        scd_html = (f"<div class=card><h2>Edge check (live)</h2><p class=muted>Do the stocks we RANK highest actually beat the market afterwards? "
                    f"Each weekly decision is scored once its horizon has passed (entry at the next close). Backtest reference: top-20 "
                    f"+{ref.get('excess', {}).get('10', 0.011)*100:.1f}% (10d) / +{ref.get('excess', {}).get('21', 0.022)*100:.1f}% (21d) over the universe, "
                    f"rank-IC about +0.08 to +0.10. Needs 8+ decisions before it means anything.</p>"
                    f"<table><tr><th>Horizon</th><th>Score</th><th class=num>Decisions</th><th class=num>Top-20 excess</th>"
                    f"<th class=num>Beat market</th><th class=num>Rank-IC</th></tr>{trs}</table></div>")
    else:
        scd_html = ("<div class=card><h2>Edge check (live)</h2><p class=muted>Collecting: each weekly decision is archived and scored once "
                    "10 trading days have passed. First results appear in about two weeks.</p></div>")
    last = f"{nav.index[-1]:%d %b %Y}" if len(nav) else "—"
    bline = f" · Nifty 500 {bret:+.2%}" if bret is not None else ""
    return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>smartpiper</title>
<style>
:root{{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e4e3de;--accent:#2f6fde;--accent2:#d9822b;--accent3:#8e5bd6;--bench:#9a9a92;--up:#127a46;--dn:#b3261e}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151514;--card:#1f1f1d;--ink:#ecebe6;--muted:#9b9a93;--line:#33322f;--accent:#7aa7ff;--accent2:#f0a35e;--accent3:#c3a1ff;--bench:#77766f;--up:#5cc58d;--dn:#ff8a80}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:900px;margin:0 auto;padding:20px 16px 48px}}
h1{{font-size:20px;margin:0}} h2{{font-size:15px;margin:0 0 10px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin-top:14px}}
.big{{font-size:30px;font-weight:650;font-variant-numeric:tabular-nums}} .muted{{color:var(--muted)}}
.up{{color:var(--up)}} .dn{{color:var(--dn)}}
table{{width:100%;border-collapse:collapse}} td,th{{padding:6px 4px;border-bottom:1px solid var(--line);text-align:left}}
.num{{text-align:right;font-variant-numeric:tabular-nums}} th{{color:var(--muted);font-weight:500;font-size:13px}}
svg{{width:100%;height:auto}} .nav{{fill:none;stroke:var(--accent);stroke-width:2.2}} .bench{{fill:none;stroke:var(--bench);stroke-width:1.6;stroke-dasharray:4 3}}
.zero{{stroke:var(--line)}} .ax{{fill:var(--muted);font-size:11px}}
.legend{{font-size:13px;color:var(--muted)}} .k{{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:4px}}
.k.nav{{background:var(--accent)}} .k.bench{{background:var(--bench)}} .k.nav2{{background:var(--accent2)}} .k.nav3{{background:var(--accent3)}}
.nav2{{fill:none;stroke:var(--accent2);stroke-width:1.8}} .nav3{{fill:none;stroke:var(--accent3);stroke-width:1.8}}
.chip{{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:1px 8px;margin:2px;font-size:13px}}
.chip.buy{{border-color:var(--up);color:var(--up)}} .chip.sell{{border-color:var(--dn);color:var(--dn)}}
ul{{padding-left:18px}} li{{margin-bottom:8px}} a{{color:var(--accent)}}
</style></head><body><main>
<h1>smartpiper <span class=muted style="font-weight:400">· paper trading, no real orders</span></h1>
<div class=card><div class=muted>Paper NAV (₹{CAPITAL:,.0f} start)</div>
<div class="big {'up' if ret >= 0 else 'dn'}">₹{total:,.0f} <span style="font-size:18px">{ret:+.2%}</span></div>
<div class=muted>as of {last}{bline} · since {E(str(L.get('start') or 'first fill pending'))} · cash ₹{L['cash']:,.0f}</div>
{chart(nav, bench, others=cmp_navs)}</div>
{tk_html}
{cmp_html}
{scd_html}
<div class=card><h2>Smart book holdings ({len(L['holdings'])})</h2>{pend_html}
<table><tr><th>Stock</th><th class=num>Value</th><th class=num>P&amp;L</th><th>Since</th></tr>{hold}</table></div>
<div class=card><h2>Latest weekly signal</h2>{sig_html}</div>
<div class=card><h2>Announcement alerts</h2><ul>{al}</ul></div>
<p class=muted style="font-size:12px">Reads {E(str(LIVE))}. Refresh the page for the latest data.</p>
</main></body></html>"""


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ("/", "/index.html"):
            self.send_response(404); self.end_headers(); return
        try:
            body, code = page().encode(), 200
        except Exception as e:                       # show the error, never crash the server
            body, code = f"<pre>dashboard error: {E(repr(e))}</pre>".encode(), 500
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers(); self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", PORT), H).serve_forever()
