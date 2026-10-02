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

from smart.config import ALERTS, CAPITAL, DATA, LEDGER, LIVE, NAV, SIGNAL

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


def chart(nav, bench, w=760, h=220, pad=34):
    if len(nav) < 2:
        return "<p class=muted>NAV chart appears after the first two trading days.</p>"
    s = (nav/nav.iloc[0] - 1)*100
    b = (bench.reindex(nav.index).ffill()/bench.reindex(nav.index).ffill().dropna().iloc[0] - 1)*100 \
        if len(bench.dropna()) else None
    vals = pd.concat([s, b]) if b is not None else s
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
    g.append(f'<path d="{path(s)}" class="nav"/>')
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Paper NAV vs Nifty 500">{"".join(g)}</svg>'
            f'<div class=legend><span class="k nav"></span>smartpiper paper &nbsp; '
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
    last = f"{nav.index[-1]:%d %b %Y}" if len(nav) else "—"
    bline = f" · Nifty 500 {bret:+.2%}" if bret is not None else ""
    return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>smartpiper</title>
<style>
:root{{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e4e3de;--accent:#2f6fde;--bench:#9a9a92;--up:#127a46;--dn:#b3261e}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151514;--card:#1f1f1d;--ink:#ecebe6;--muted:#9b9a93;--line:#33322f;--accent:#7aa7ff;--bench:#77766f;--up:#5cc58d;--dn:#ff8a80}}}}
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
.k.nav{{background:var(--accent)}} .k.bench{{background:var(--bench)}}
.chip{{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:1px 8px;margin:2px;font-size:13px}}
.chip.buy{{border-color:var(--up);color:var(--up)}} .chip.sell{{border-color:var(--dn);color:var(--dn)}}
ul{{padding-left:18px}} li{{margin-bottom:8px}} a{{color:var(--accent)}}
</style></head><body><main>
<h1>smartpiper <span class=muted style="font-weight:400">· paper trading, no real orders</span></h1>
<div class=card><div class=muted>Paper NAV (₹{CAPITAL:,.0f} start)</div>
<div class="big {'up' if ret >= 0 else 'dn'}">₹{total:,.0f} <span style="font-size:18px">{ret:+.2%}</span></div>
<div class=muted>as of {last}{bline} · since {E(str(L.get('start') or 'first fill pending'))} · cash ₹{L['cash']:,.0f}</div>
{chart(nav, bench)}</div>
<div class=card><h2>Holdings ({len(L['holdings'])})</h2>{pend_html}
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
