"""
Can an LLM read announcements better than keyword rules?  (Groq, key in .env)

Sample DEV-period announcements the rules could NOT place (type other/guidance), from
stocks in the universe. Each text is ANONYMISED first -- company name, symbol and years
stripped -- so the model judges the content, not what it remembers happened to the company
(its training data covers these years: that would be lookahead).

The LLM returns per item: type, sentiment (-2..2), material (0/1). Then measure, on the
sample: day-0 reaction and +21/+63d market-adjusted drift by LLM sentiment. Scores are
cached in data_store/llm_scores.parquet so re-runs are free.

Run:  python scripts/research_smart_llm.py [--n 1000] [--model qwen/qwen3.8-27b]
"""
from __future__ import annotations
import argparse, json, re, sys, time
from pathlib import Path
import numpy as np, pandas as pd, requests

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))
CACHE = BASE/"data_store/llm_scores.parquet"
URL = "https://api.groq.com/openai/v1/chat/completions"

PROMPT = """You are an equity analyst. Below are {n} announcements by Indian listed companies.
Company names are hidden. For EACH, judge only from the text how it affects the company's
future earnings or risk. Reply with JSON: {{"items": [{{"i": <index>, "type": <one of
order|capex|acquisition|fund_raise|governance|legal|rating|results|operations|routine>,
"sentiment": <-2..2>, "material": <0 or 1>}}, ...]}}. Routine compliance filings are
material 0, sentiment 0.

{items}"""


def key():
    return [l.split("=", 1)[1].strip() for l in open(BASE/".env") if l.startswith("GROQ_API_KEY=")][0]


def anonymise(text, sym):
    t = str(text or "")
    t = re.sub(r"^.*?\b(limited|ltd\.?)\b", "[COMPANY]", t, count=1, flags=re.I)
    t = re.sub(rf"\b{re.escape(sym)}\b", "[COMPANY]", t, flags=re.I) if sym else t
    t = re.sub(r"\b(19|20)\d{2}\b", "[YEAR]", t)
    return t[:400]


def score_batch(rows, model, H):
    items = "\n".join(f"[{i}] {t}" for i, t in enumerate(rows))
    body = {"model": model, "temperature": 0, "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": PROMPT.format(n=len(rows), items=items)}]}
    for k in range(6):
        r = requests.post(URL, headers=H, json=body, timeout=120)
        if r.status_code == 429:                       # free-tier limit: wait it out
            time.sleep(float(r.headers.get("retry-after", 20)) + 1); continue
        if r.ok:
            try:
                out = json.loads(r.json()["choices"][0]["message"]["content"])["items"]
                return {int(x["i"]): x for x in out}
            except (KeyError, ValueError, TypeError):
                pass
        time.sleep(5*(k + 1))
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--model", default="qwen/qwen3.8-27b")
    ap.add_argument("--batch", type=int, default=10)
    a = ap.parse_args()
    from scripts import research_smart_ml as SM
    from scripts.research_smart_ann import load, tradable_day
    D = load()
    P = SM.load_panel()
    C, R, alive, turn60, age, univ, mkt = SM.base(P)
    idx = C.index
    D = D[D["sym"].isin(C.columns) & D["type"].isin(["other", "guidance"])].copy()
    D["pos"] = tradable_day(D["ts"], idx)
    D = D[(D["pos"] > 1) & (D["pos"] < len(idx) - 65) & (D["ts"] <= "2022-12-31")]
    col = C.columns.get_indexer(D["sym"])
    D = D[univ.values[D["pos"].values, col] & D["text"].str.len().gt(60)]
    S = D.sample(a.n, random_state=11).reset_index(drop=True)
    S["anon"] = [anonymise(t, s) for t, s in zip(S["text"], S["sym"])]

    done = pd.read_parquet(CACHE) if CACHE.exists() else pd.DataFrame(columns=["anon", "model"])
    have = set(done.loc[done["model"] == a.model, "anon"])
    todo = [t for t in S["anon"].unique() if t not in have]
    H = {"Authorization": f"Bearer {key()}"}
    print(f"sample {len(S)}, to score {len(todo)} with {a.model}", flush=True)
    new = []
    for b in range(0, len(todo), a.batch):
        chunk = todo[b:b + a.batch]
        res = score_batch(chunk, a.model, H)
        for i, t in enumerate(chunk):
            x = res.get(i, {})
            new.append({"anon": t, "model": a.model, "llm_type": x.get("type"),
                        "sentiment": pd.to_numeric(x.get("sentiment"), errors="coerce"),
                        "material": pd.to_numeric(x.get("material"), errors="coerce")})
        if (b//a.batch) % 10 == 0:
            print(f"  scored {b + len(chunk)}/{len(todo)}", flush=True)
        time.sleep(2.5)                                 # ~8k tokens/min free tier
    if new:
        done = pd.concat([done, pd.DataFrame(new)], ignore_index=True)
        done.to_parquet(CACHE)
    S = S.merge(done[done["model"] == a.model].drop_duplicates("anon"), on="anon", how="left")

    Cf = C.ffill().values; mc = (1 + mkt.fillna(0)).cumprod().values
    p, c = S["pos"].values, C.columns.get_indexer(S["sym"])
    S["react"] = Cf[p, c]/Cf[p - 1, c] - 1 - (mc[p]/mc[p - 1] - 1)
    for h in (21, 63):
        S[f"r{h}"] = Cf[p + 1 + h, c]/Cf[p + 1, c] - 1 - (mc[p + 1 + h]/mc[p + 1] - 1)
    print(f"\nscored {S['sentiment'].notna().mean():.0%} of sample; LLM type mix:",
          S["llm_type"].value_counts().head(8).to_dict())
    print("\nby LLM sentiment (market-adjusted; entry next close):")
    g = S.groupby("sentiment")[["react", "r21", "r63"]].agg(["count", "mean"])
    print((g*[1, 100, 1, 100, 1, 100]).round(2).to_string())
    m = S[S["material"] == 1]
    print(f"\nmaterial=1: n={len(m)}  corr(sentiment, day0)={m['sentiment'].corr(m['react'], method='spearman'):+.3f}  "
          f"corr(sentiment, +63d)={m['sentiment'].corr(m['r63'], method='spearman'):+.3f}")


if __name__ == "__main__":
    main()
