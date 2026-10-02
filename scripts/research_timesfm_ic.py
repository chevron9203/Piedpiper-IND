"""
Does Google's TimesFM contain any cross-sectional signal for NSE mid-caps?

Cheap falsification BEFORE any backtest. A backtest tells you what a signal earned;
the information coefficient tells you whether it knows anything at all. If TimesFM's
rank-IC is ~0, no amount of portfolio construction rescues it and we stop.

Method: at each sampled month-end, take the SAME eligible mid-cap universe the live
book uses, and score every stock three ways --
    momentum   : the deployed multi-timeframe risk-adjusted score
    mom12      : plain 12-month return (naive control)
    timesfm    : forecast the next 21 trading days from trailing daily closes,
                 score = forecast_price[-1]/last_close - 1
then rank-correlate each against the REALIZED next-21-day return.

Strictly point-in-time: the forecast context ends at the month-end bar, the realized
return starts after it.

Run:  python scripts/research_timesfm_ic.py --months 40
      python scripts/research_timesfm_ic.py --months 4 --max-stocks 60   (pilot)

Needs the separate .venv-timesfm (torch + timesfm), NOT the project .venv.
"""
from __future__ import annotations
import sys, argparse, time
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(__file__).parent.parent
PANEL = BASE/"data_store/daily_panel_full.parquet"
OUT = BASE/"data_store/timesfm_ic.csv"

PRICE_MIN = 30.0; VOL_FLOOR = 0.02
CONTEXT = 512          # trailing daily closes fed to the model (~2 years)
HORIZON = 21           # one trading month ahead


def spearman(a, b):
    """Rank correlation, dropping pairs where either side is missing."""
    d = pd.DataFrame({"a": pd.Series(a).astype("float64"),
                      "b": pd.Series(b).astype("float64")}).dropna()
    if len(d) < 10: return np.nan
    ra, rb = d["a"].rank(), d["b"].rank()
    if ra.std() == 0 or rb.std() == 0: return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def eligible(C, turn_sm, vol, ma_long, i):
    """The live book's mid-cap tier at bar i (point-in-time)."""
    pr = C.iloc[i]; tn = turn_sm.iloc[i]; v = vol.iloc[i]
    hist = C.iloc[:i+1].notna().sum()
    base = pr.index[(pr >= PRICE_MIN) & (hist >= 252) & tn.notna() & (tn > 0)
                    & (v >= VOL_FLOOR)]
    if len(base) < 100: return []
    rank = tn[base].rank(pct=True)
    mid = rank.index[(rank > 0.60) & (rank <= 0.90)]
    ml = ma_long.iloc[i]
    return [s for s in mid if pr[s] > ml.get(s, np.inf)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=40)
    ap.add_argument("--max-stocks", type=int, default=0, help="cap per month (0 = all)")
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    import timesfm

    pk = pd.read_parquet(PANEL)
    C = pk.xs("close", 1, 0).astype("float64")
    T = pk.xs("turn", 1, 0).astype("float64")
    turn_sm = T.rolling(21).mean()
    vol = C.pct_change().rolling(126).std()*np.sqrt(21)
    ma_long = C.rolling(210).mean()
    m3 = C/C.shift(63)-1; m6 = C/C.shift(126)-1; m12 = C/C.shift(252)-1

    # month-end bars with a full horizon of future data available
    idx = C.index
    me = pd.Series(idx, index=idx).groupby([idx.year, idx.month]).last().values
    me = [d for d in me if idx.get_loc(d) > CONTEXT+252 and idx.get_loc(d)+HORIZON < len(idx)]
    me = me[-args.months:]
    print(f"sampling {len(me)} month-ends: {pd.Timestamp(me[0]).date()} -> {pd.Timestamp(me[-1]).date()}")

    print("loading TimesFM 2.5 200M (torch) ...")
    t0 = time.time()
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model.compile(timesfm.ForecastConfig(max_context=CONTEXT, max_horizon=HORIZON,
                                         normalize_inputs=True, per_core_batch_size=args.batch))
    print(f"  ready in {time.time()-t0:.1f}s")

    rows = []
    for d in me:
        i = idx.get_loc(d)
        elig = eligible(C, turn_sm, vol, ma_long, i)
        if args.max_stocks: elig = elig[:args.max_stocks]
        if len(elig) < 30: continue

        # --- realized forward return (the thing every signal is trying to predict)
        fwd = (C.iloc[i+HORIZON][elig]/C.iloc[i][elig] - 1)

        # --- existing signals
        v = vol.iloc[i][elig].replace(0, np.nan)
        sc = pd.Series(0.0, index=elig); nc = 0
        for sg in (m3, m6, m12):
            x = (sg.iloc[i][elig]/v).dropna()
            if len(x): sc = sc.add(x.rank(pct=True), fill_value=0); nc += 1
        mom = sc/max(nc, 1)
        mom12 = m12.iloc[i][elig]

        # --- TimesFM
        ctx = []
        keep = []
        for s in elig:
            h = C[s].iloc[max(0, i-CONTEXT+1):i+1].dropna()
            if len(h) < 128: continue
            ctx.append(h.values.astype("float32")); keep.append(s)
        if len(keep) < 30: continue
        # NB: model.forecast() MUTATES the list it is given -- it pads in place up to
        # per_core_batch_size. Snapshot the anchor prices first and hand it a copy,
        # or `last` silently picks up the padding rows.
        last = np.array([c[-1] for c in ctx], dtype="float64")
        t1 = time.time()
        point, _ = model.forecast(horizon=HORIZON, inputs=list(ctx))
        point = np.asarray(point)[:len(keep)]
        if point.shape[0] != len(keep):
            print(f"  ! shape mismatch point={point.shape} keep={len(keep)} — skipping")
            continue
        tf = pd.Series(point[:, -1]/last - 1, index=keep)

        common = [s for s in keep if pd.notna(fwd.get(s)) and pd.notna(mom.get(s))]
        if len(common) < 30: continue
        rows.append(dict(
            date=pd.Timestamp(d).date(), n=len(common), secs=round(time.time()-t1, 1),
            ic_momentum=spearman(mom[common], fwd[common]),
            ic_mom12=spearman(mom12[common], fwd[common]),
            ic_timesfm=spearman(tf[common], fwd[common]),
        ))
        r = rows[-1]
        print(f"  {r['date']}  n={r['n']:4d}  {r['secs']:6.1f}s  "
              f"IC mom {r['ic_momentum']:+.3f}   mom12 {r['ic_mom12']:+.3f}   "
              f"timesfm {r['ic_timesfm']:+.3f}")

    if not rows:
        raise SystemExit("no usable months")
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    print("\n" + "="*72); print("INFORMATION COEFFICIENT (Spearman, vs realized 21d return)"); print("="*72)
    print(f"{'signal':14} {'mean IC':>9} {'sd':>7} {'t-stat':>8} {'% months>0':>11}")
    print("-"*54)
    for col, name in [("ic_momentum", "momentum"), ("ic_mom12", "mom12 (naive)"),
                      ("ic_timesfm", "TimesFM")]:
        s = df[col].dropna()
        t = s.mean()/s.std()*np.sqrt(len(s)) if s.std() > 0 else 0
        print(f"{name:14} {s.mean():+9.4f} {s.std():7.4f} {t:+8.2f} {100*(s>0).mean():10.0f}%")
    print(f"\nn = {len(df)} months. |t| > 2 is the usual bar for 'this signal knows something'.")
    print(f"saved -> {OUT.relative_to(BASE)}")


if __name__ == "__main__":
    main()
