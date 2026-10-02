"""
Does TimesFM's REAL edge -- stock volatility forecasting -- improve the actual book?

Established so far:
  * TimesFM cannot forecast prices (loses to random walk on index AND stocks)
  * TimesFM DOES forecast individual stock volatility better than EWMA
    (MAE ratio 0.927, t=+13.7, n=30k non-overlapping windows) -- a real edge

The deployed momentum score is  momentum / trailing_vol , so a better vol estimate is
a drop-in upgrade to the signal you already run. The caveat: the score only uses vol
to RANK, and TimesFM's rank-correlation with realised vol barely beats EWMA
(0.557 vs 0.554). So the prior is "statistically real, practically small". Measure it.

Identical universes and dates; only the vol denominator / weighting changes:
  mom_trailing  momentum / trailing realised vol      (DEPLOYED baseline)
  mom_tfvol     momentum / TimesFM-forecast vol
  invvol_tf     deployed picks, 1/TimesFM-vol weighted (vs equal-weight)

Run:  python scripts/research_timesfm_volsignal.py --from-year 2012
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(__file__).parent.parent
PANEL = BASE/"data_store/daily_panel_full.parquet"
OUT = BASE/"data_store/timesfm_volsignal.csv"

PRICE_MIN = 30.0; VOL_FLOOR = 0.02
VCTX = 400          # realised-vol history fed to the model
HORIZON = 21; TOP_N = 15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-year", type=int, default=2012)
    args = ap.parse_args()
    import timesfm

    pk = pd.read_parquet(PANEL)
    C = pk.xs("close", 1, 0).astype("float64")
    T = pk.xs("turn", 1, 0).astype("float64")
    turn_sm = T.rolling(21).mean()
    lr = np.log(C).diff()
    rv = lr.rolling(HORIZON).std()*np.sqrt(252)          # realised-vol panel
    vol_tr = C.pct_change().rolling(126).std()*np.sqrt(21)   # the DEPLOYED denominator
    ma_long = C.rolling(210).mean()
    m3 = C/C.shift(63)-1; m6 = C/C.shift(126)-1; m12 = C/C.shift(252)-1
    idx = C.index

    me = pd.Series(idx, index=idx).groupby([idx.year, idx.month]).last().values
    me = [d for d in me if idx.get_loc(d) > VCTX+252
          and idx.get_loc(d)+HORIZON < len(idx)
          and pd.Timestamp(d).year >= args.from_year]
    print(f"{len(me)} month-ends: {pd.Timestamp(me[0]).date()} -> {pd.Timestamp(me[-1]).date()}")

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model.compile(timesfm.ForecastConfig(max_context=VCTX, max_horizon=HORIZON,
                                         normalize_inputs=True, per_core_batch_size=256))
    print("model ready\n")

    rows = []; t0 = time.time()
    for d in me:
        i = idx.get_loc(d)
        pr = C.iloc[i]; tn = turn_sm.iloc[i]; vt = vol_tr.iloc[i]
        hist = C.iloc[:i+1].notna().sum()
        base = pr.index[(pr >= PRICE_MIN) & (hist >= 252) & tn.notna() & (tn > 0)
                        & (vt >= VOL_FLOOR)]
        if len(base) < 100: continue
        rank = tn[base].rank(pct=True)
        mid = rank.index[(rank > 0.60) & (rank <= 0.90)]
        ml = ma_long.iloc[i]
        elig = [s for s in mid if pr[s] > ml.get(s, np.inf)]
        if len(elig) < 40: continue

        # TimesFM vol forecast for each eligible name
        ctx, keep = [], []
        for s in elig:
            h = rv[s].iloc[max(0, i-VCTX+1):i+1].dropna()
            if len(h) < 128: continue
            ctx.append(h.values.astype("float32")); keep.append(s)
        if len(keep) < 40: continue
        pred = np.asarray(model.forecast(horizon=HORIZON,
                                         inputs=list(ctx))[0])[:len(keep), -1]
        tfvol = pd.Series(np.maximum(pred, 1e-6), index=keep)

        fwd = C.iloc[i+HORIZON][keep]/C.iloc[i][keep] - 1
        vtr = vt[keep].replace(0, np.nan)

        def score(denom):
            sc = pd.Series(0.0, index=keep); nc = 0
            for sg in (m3, m6, m12):
                x = (sg.iloc[i][keep]/denom).dropna()
                if len(x): sc = sc.add(x.rank(pct=True), fill_value=0); nc += 1
            return sc/nc if nc else None

        s_tr, s_tf = score(vtr), score(tfvol)
        if s_tr is None or s_tf is None: continue
        ok = [s for s in keep if pd.notna(fwd.get(s)) and pd.notna(s_tr.get(s))
              and pd.notna(s_tf.get(s))]
        if len(ok) < 40: continue
        f = fwd[ok]

        p_tr = s_tr[ok].nlargest(TOP_N).index
        p_tf = s_tf[ok].nlargest(TOP_N).index
        w = 1/tfvol[p_tr]; w = w/w.sum()                  # inverse-vol on DEPLOYED picks
        rows.append(dict(
            date=pd.Timestamp(d).date(), n=len(ok),
            overlap=len(set(p_tr) & set(p_tf)),
            mom_trailing=float(f[p_tr].mean()),
            mom_tfvol=float(f[p_tf].mean()),
            invvol_tf=float((f[p_tr]*w).sum()),
            ew=float(f.mean()),
        ))
        r = rows[-1]
        print(f"  {r['date']} n={r['n']:4d} overlap {r['overlap']:2d}/15 | "
              f"trailing {r['mom_trailing']*100:+6.2f}%  tfvol {r['mom_tfvol']*100:+6.2f}%  "
              f"invvol {r['invvol_tf']*100:+6.2f}%  ew {r['ew']*100:+6.2f}%")

    df = pd.DataFrame(rows); df.to_csv(OUT, index=False)
    print(f"\n{len(df)} months in {time.time()-t0:.0f}s")
    print("\n" + "="*76)
    print("TOP-15 FORWARD 21d RETURN — does a better vol estimate help?")
    print("="*76)
    print(f"{'variant':24} {'mean/mo':>9} {'vs deployed':>12} {'t-stat':>8} {'ann~':>8}")
    print("-"*64)
    base_col = df["mom_trailing"]
    for col, name in [("mom_trailing", "deployed (trailing)"),
                      ("mom_tfvol", "TimesFM vol denom"),
                      ("invvol_tf", "inv-vol weight (TF)"),
                      ("ew", "equal-weight univ")]:
        s = df[col]
        diff = s - base_col
        t = diff.mean()/diff.std()*np.sqrt(len(diff)) if diff.std() > 0 else 0
        print(f"{name:24} {s.mean()*100:+8.2f}% {diff.mean()*100:+11.2f}% {t:+8.2f} "
              f"{((1+s.mean())**12-1)*100:+7.1f}%")
    print(f"\n  mean pick overlap between the two scores: {df['overlap'].mean():.1f}/15")
    print(f"saved -> {OUT.relative_to(BASE)}")


if __name__ == "__main__":
    main()
