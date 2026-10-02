"""
TimesFM, judged the way the book actually trades: top-15, over a long sample.

The IC test (research_timesfm_ic.py) found TimesFM mildly positive (t=2.04) over
2021-26 and, more interestingly, NEGATIVELY correlated with momentum's IC -- it works
in months momentum doesn't. Two reasons that is not yet actionable:
  * IC is a whole-cross-section statistic. The book buys 15 names. A signal can have
    positive IC and a flat top decile.
  * 60 months in a weak-momentum regime is a thin, possibly flattering sample.

So: same point-in-time universe, longer history, and the metric that pays the bills --
mean forward 21d return of the TOP 15 by each signal, against the equal-weight
eligible universe as the benchmark. Also scores a rank-average COMBO, since a
diversifying signal is worth more than a merely better one.

Run:  python scripts/research_timesfm_portfolio.py --from-year 2012
"""
from __future__ import annotations
import sys, argparse, time
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(__file__).parent.parent
PANEL = BASE/"data_store/daily_panel_full.parquet"
OUT = BASE/"data_store/timesfm_portfolio.csv"

PRICE_MIN = 30.0; VOL_FLOOR = 0.02
CONTEXT = 512; HORIZON = 21; TOP_N = 15


def spearman(a, b):
    d = pd.DataFrame({"a": pd.Series(a).astype("float64"),
                      "b": pd.Series(b).astype("float64")}).dropna()
    if len(d) < 10: return np.nan
    ra, rb = d["a"].rank(), d["b"].rank()
    if ra.std() == 0 or rb.std() == 0: return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-year", type=int, default=2012)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()
    import timesfm

    pk = pd.read_parquet(PANEL)
    C = pk.xs("close", 1, 0).astype("float64")
    T = pk.xs("turn", 1, 0).astype("float64")
    turn_sm = T.rolling(21).mean()
    vol = C.pct_change().rolling(126).std()*np.sqrt(21)
    ma_long = C.rolling(210).mean()
    m3 = C/C.shift(63)-1; m6 = C/C.shift(126)-1; m12 = C/C.shift(252)-1
    idx = C.index

    me = pd.Series(idx, index=idx).groupby([idx.year, idx.month]).last().values
    me = [d for d in me if idx.get_loc(d) > CONTEXT+252
          and idx.get_loc(d)+HORIZON < len(idx)
          and pd.Timestamp(d).year >= args.from_year]
    print(f"{len(me)} month-ends: {pd.Timestamp(me[0]).date()} -> {pd.Timestamp(me[-1]).date()}")

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model.compile(timesfm.ForecastConfig(max_context=CONTEXT, max_horizon=HORIZON,
                                         normalize_inputs=True, per_core_batch_size=args.batch))
    print("model ready\n")

    rows = []; t0 = time.time()
    for d in me:
        i = idx.get_loc(d)
        pr = C.iloc[i]; tn = turn_sm.iloc[i]; v = vol.iloc[i]
        hist = C.iloc[:i+1].notna().sum()
        base = pr.index[(pr >= PRICE_MIN) & (hist >= 252) & tn.notna() & (tn > 0)
                        & (v >= VOL_FLOOR)]
        if len(base) < 100: continue
        rank = tn[base].rank(pct=True)
        mid = rank.index[(rank > 0.60) & (rank <= 0.90)]
        ml = ma_long.iloc[i]
        elig = [s for s in mid if pr[s] > ml.get(s, np.inf)]
        if len(elig) < 40: continue

        fwd = C.iloc[i+HORIZON][elig]/C.iloc[i][elig] - 1
        vv = v[elig].replace(0, np.nan)
        sc = pd.Series(0.0, index=elig); nc = 0
        for sg in (m3, m6, m12):
            x = (sg.iloc[i][elig]/vv).dropna()
            if len(x): sc = sc.add(x.rank(pct=True), fill_value=0); nc += 1
        if not nc: continue
        mom = sc/nc

        ctx, keep = [], []
        for s in elig:
            h = C[s].iloc[max(0, i-CONTEXT+1):i+1].dropna()
            if len(h) < 128: continue
            ctx.append(h.values.astype("float32")); keep.append(s)
        if len(keep) < 40: continue
        last = np.array([c[-1] for c in ctx], dtype="float64")   # BEFORE forecast: it pads in place
        point = np.asarray(model.forecast(horizon=HORIZON, inputs=list(ctx))[0])[:len(keep)]
        tf = pd.Series(point[:, -1]/last - 1, index=keep)

        common = [s for s in keep if pd.notna(fwd.get(s)) and pd.notna(mom.get(s))
                  and pd.notna(tf.get(s))]
        if len(common) < 40: continue
        f = fwd[common]; mo = mom[common]; tfc = tf[common]
        combo = mo.rank(pct=True) + tfc.rank(pct=True)     # equal-weight rank blend

        def top(sig): return float(f[sig.nlargest(TOP_N).index].mean())
        rows.append(dict(
            date=pd.Timestamp(d).date(), n=len(common),
            ic_mom=spearman(mo, f), ic_tf=spearman(tfc, f),
            xcorr_mom_tf=spearman(mo, tfc),
            top_mom=top(mo), top_tf=top(tfc), top_combo=top(combo),
            bot_tf=float(f[tfc.nsmallest(TOP_N).index].mean()),
            ew=float(f.mean()),
        ))
        r = rows[-1]
        print(f"  {r['date']} n={r['n']:4d} xcorr{r['xcorr_mom_tf']:+.2f} | "
              f"top15 mom {r['top_mom']*100:+6.2f}%  tf {r['top_tf']*100:+6.2f}%  "
              f"combo {r['top_combo']*100:+6.2f}%  ew {r['ew']*100:+6.2f}%")

    df = pd.DataFrame(rows); df.to_csv(OUT, index=False)
    print(f"\n{len(df)} months in {time.time()-t0:.0f}s")

    print("\n" + "="*78)
    print("WHAT THE BOOK ACTUALLY TRADES — mean forward 21d return of the top 15")
    print("="*78)
    print(f"{'signal':16} {'mean/mo':>9} {'vs EW':>8} {'t-stat':>8} {'win%':>7} {'ann~':>8}")
    print("-"*62)
    ew = df["ew"]
    for col, name in [("top_mom", "momentum"), ("top_tf", "TimesFM"),
                      ("top_combo", "combo (rank avg)"), ("ew", "equal-weight univ")]:
        s = df[col].dropna()
        ex = s - ew.reindex(s.index) if col != "ew" else s
        t = ex.mean()/ex.std()*np.sqrt(len(ex)) if ex.std() > 0 else 0
        ann = (1+s.mean())**12 - 1
        print(f"{name:16} {s.mean()*100:+8.2f}% {ex.mean()*100:+7.2f}% {t:+8.2f} "
              f"{100*(s>0).mean():6.0f}% {ann*100:+7.1f}%")
    print("\n" + "="*78); print("INFORMATION COEFFICIENT"); print("="*78)
    for col, name in [("ic_mom", "momentum"), ("ic_tf", "TimesFM")]:
        s = df[col].dropna()
        print(f"  {name:10} mean {s.mean():+.4f}  t {s.mean()/s.std()*np.sqrt(len(s)):+.2f}")
    print(f"\n  signal cross-correlation (mom vs TimesFM, per month): "
          f"{df['xcorr_mom_tf'].mean():+.3f}")
    print(f"  TimesFM top15 minus bottom15 spread: "
          f"{(df['top_tf']-df['bot_tf']).mean()*100:+.2f}%/mo")
    print(f"\nsaved -> {OUT.relative_to(BASE)}")


if __name__ == "__main__":
    main()
