"""
Does TimesFM's volatility edge survive a significance test and generalise beyond Nifty?

The use-case screen found TimesFM beating naive (0.913) and EWMA(0.94) (0.946) at
forecasting Nifty's next-21d realised vol. That is the one place it looked useful --
and it is theoretically the right place, since volatility clusters and is genuinely
forecastable while returns are not. But n=145 with no paired test proves nothing.

This widens the sample to many stocks plus the index and runs a Diebold-Mariano style
paired test on the loss differential (|err_baseline| - |err_timesfm|). Positive mean
with |t| > 2 = a real edge, not a lucky sample.

Targets are the NEXT NON-OVERLAPPING 21 days, so the rolling input window cannot leak.

Run:  python scripts/research_timesfm_vol.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(__file__).parent.parent
PANEL = BASE/"data_store/daily_panel_full.parquet"
DAILY = BASE/"data_store/eod2/src/eod2_data/daily"
CONTEXT = 512; HORIZON = 21; LAM = 0.94


def dm_test(err_base, err_tf, label):
    """Paired test on absolute-error differential. d>0 => TimesFM better."""
    d = np.abs(err_base) - np.abs(err_tf)
    d = d[np.isfinite(d)]
    if len(d) < 20: return None
    t = d.mean()/d.std(ddof=1)*np.sqrt(len(d))
    win = (d > 0).mean()
    print(f"    vs {label:18} mean gain {d.mean():+.5f}  t {t:+6.2f}  "
          f"TimesFM better in {win*100:4.0f}% of windows  "
          f"{'SIGNIFICANT' if abs(t) > 2 else 'not significant'}")
    return dict(mean_gain=float(d.mean()), t=float(t), win=float(win), n=int(len(d)))


def build_samples(series_map, model):
    """For each series: TimesFM / naive / EWMA forecasts of next-21d realised vol."""
    ctx, naive, ewma, fut, tag = [], [], [], [], []
    for name, px in series_map.items():
        px = px.dropna()
        if len(px) < CONTEXT + 3*HORIZON: continue
        r = np.log(px).diff()
        rv = (r.rolling(HORIZON).std()*np.sqrt(252)).dropna()
        ew = (r.ewm(alpha=1-LAM).std()*np.sqrt(252)).reindex(rv.index)
        for i in range(CONTEXT, len(rv)-HORIZON, HORIZON):
            c = rv.iloc[i-CONTEXT:i].values.astype("float32")
            if not np.isfinite(c).all(): continue
            d0 = rv.index[i-1]
            seg = r.loc[r.index > d0].iloc[:HORIZON]
            if len(seg) < HORIZON: continue
            ctx.append(c); naive.append(rv.iloc[i-1]); ewma.append(ew.iloc[i-1])
            fut.append(seg.std()*np.sqrt(252)); tag.append(name)
    preds = []
    for k in range(0, len(ctx), 256):
        ch = ctx[k:k+256]
        preds.append(np.asarray(model.forecast(horizon=HORIZON,
                                               inputs=list(ch))[0])[:len(ch), -1])
    return (np.concatenate(preds) if preds else np.array([]),
            np.array(naive), np.array(ewma), np.array(fut), np.array(tag))


def main():
    import timesfm
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model.compile(timesfm.ForecastConfig(max_context=CONTEXT, max_horizon=HORIZON,
                                         normalize_inputs=True, per_core_batch_size=256))

    def idx(n):
        d = pd.read_csv(DAILY/f"{n}.csv", usecols=["Date", "Close"])
        d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
        return d.dropna(subset=["Date"]).set_index("Date")["Close"].astype("float64").sort_index()

    pk = pd.read_parquet(PANEL)
    C = pk.xs("close", 1, 0).astype("float64")
    rng = np.random.default_rng(1)
    cols = [c for c in C.columns if C[c].notna().sum() > 2500]
    cols = list(rng.choice(cols, size=min(150, len(cols)), replace=False))

    groups = {
        "INDEX (Nifty 50)": {"nifty 50": idx("nifty 50")},
        "STOCKS (150 names)": {c: C[c] for c in cols},
    }
    out = {}
    for gname, smap in groups.items():
        pred, naive, ewma, fut, tag = build_samples(smap, model)
        if not len(pred): continue
        ok = np.isfinite(pred) & np.isfinite(fut) & np.isfinite(naive) & np.isfinite(ewma)
        pred, naive, ewma, fut = pred[ok], naive[ok], ewma[ok], fut[ok]
        mae = lambda x: float(np.mean(np.abs(x-fut)))
        print(f"\n{'='*78}\n{gname}  (n={len(fut)} non-overlapping windows)\n{'='*78}")
        print(f"    MAE  TimesFM {mae(pred):.5f} | naive {mae(naive):.5f} | EWMA {mae(ewma):.5f}")
        print(f"    ratio vs naive {mae(pred)/mae(naive):.3f} | "
              f"vs EWMA {mae(pred)/mae(ewma):.3f}")
        out[gname] = {
            "n": len(fut), "mae_tf": mae(pred), "mae_naive": mae(naive), "mae_ewma": mae(ewma),
            "naive": dm_test(naive-fut, pred-fut, "naive (last vol)"),
            "ewma": dm_test(ewma-fut, pred-fut, "EWMA(0.94)"),
        }
        # correlation with the truth — does it rank high/low vol periods correctly?
        for nm, v in [("TimesFM", pred), ("naive", naive), ("EWMA", ewma)]:
            c = np.corrcoef(pd.Series(v).rank(), pd.Series(fut).rank())[0, 1]
            print(f"    rank-corr with realised: {nm:8} {c:+.3f}")

    rows = []
    for g, v in out.items():
        for b in ("naive", "ewma"):
            if v[b]: rows.append(dict(group=g, baseline=b, **v[b],
                                      mae_tf=v["mae_tf"], mae_base=v[f"mae_{b}"]))
    pd.DataFrame(rows).to_csv(BASE/"data_store/timesfm_vol.csv", index=False)
    print("\nsaved -> data_store/timesfm_vol.csv")


if __name__ == "__main__":
    main()
