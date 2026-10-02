"""
Is TimesFM any good ON THIS DATA -- and is there a job here it actually fits?

The portfolio test only showed it cannot RANK stocks cross-sectionally. That is one
narrow job, and not the one the model was built for. This asks the prior question
(does it forecast at all?) and tests the two tasks that genuinely suit a univariate
foundation model better than stock-picking does.

  A  INDEX LEVEL   forecast Nifty 50 21d ahead        vs naive (last value), drift
  B  STOCK LEVEL   forecast single stocks 21d ahead   vs naive
  C  VOLATILITY    forecast realised vol 21d ahead    vs naive, EWMA(0.94)

Scoring is a ratio of mean absolute error against the baseline (MASE-style):
<1.0 means TimesFM beat the baseline, >1.0 means it lost. Beating "last value" on a
near-random-walk is hard by construction -- that is the point, it is the honest bar.

For C the forecast is evaluated against the NEXT NON-OVERLAPPING 21-day realised vol,
so the rolling window cannot leak the answer into the target.

Run:  python scripts/research_timesfm_usecases.py
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(__file__).parent.parent
PANEL = BASE/"data_store/daily_panel_full.parquet"
DAILY = BASE/"data_store/eod2/src/eod2_data/daily"
CONTEXT = 512; HORIZON = 21


def load_index(name):
    f = DAILY/f"{name}.csv"
    d = pd.read_csv(f, usecols=["Date", "Close"])
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    return (d.dropna(subset=["Date"]).set_index("Date")["Close"]
             .astype("float64").sort_index())


def score(name, pred, actual, baselines: dict):
    """MAE of TimesFM vs each baseline. Ratio < 1 means TimesFM won."""
    pred = np.asarray(pred, dtype="float64"); actual = np.asarray(actual, dtype="float64")
    ok = np.isfinite(pred) & np.isfinite(actual)
    pred, actual = pred[ok], actual[ok]
    mae_tf = np.mean(np.abs(pred-actual))
    out = {"n": int(ok.sum()), "mae_timesfm": mae_tf}
    print(f"\n  {name}  (n={ok.sum()})")
    print(f"    {'model':22} {'MAE':>12} {'ratio vs TimesFM':>18}")
    print(f"    {'TimesFM':22} {mae_tf:12.5f} {'--':>18}")
    for bname, b in baselines.items():
        b = np.asarray(b, dtype="float64")[ok]
        mae_b = np.mean(np.abs(b-actual))
        ratio = mae_tf/mae_b if mae_b > 0 else np.nan
        verdict = "TimesFM better" if ratio < 1 else "baseline better"
        print(f"    {bname:22} {mae_b:12.5f} {ratio:17.3f}  {verdict}")
        out[f"mae_{bname}"] = mae_b; out[f"ratio_vs_{bname}"] = ratio
    return out


def main():
    import timesfm
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model.compile(timesfm.ForecastConfig(max_context=CONTEXT, max_horizon=HORIZON,
                                         normalize_inputs=True, per_core_batch_size=256))
    print("model ready")
    results = {}

    # ---------------------------------------------------------------- A: index level
    nifty = load_index("nifty 50")
    pts = list(range(CONTEXT, len(nifty)-HORIZON, 21))
    ctx = [nifty.iloc[i-CONTEXT:i].values.astype("float32") for i in pts]
    last = np.array([nifty.iloc[i-1] for i in pts])
    drift = np.array([nifty.iloc[i-1] + (nifty.iloc[i-1]-nifty.iloc[i-64])/63*HORIZON
                      for i in pts])
    actual = np.array([nifty.iloc[i+HORIZON-1] for i in pts])
    pred = np.asarray(model.forecast(horizon=HORIZON, inputs=list(ctx))[0])[:len(pts), -1]
    # score in RETURN space so the metric is scale-free across eras
    results["A_index"] = score("A  NIFTY 50 level, 21d ahead (as % return)",
                               pred/last-1, actual/last-1,
                               {"naive (last value)": np.zeros(len(pts)),
                                "drift (63d slope)": drift/last-1})

    # ---------------------------------------------------------------- B: stock level
    pk = pd.read_parquet(PANEL)
    C = pk.xs("close", 1, 0).astype("float64")
    rng = np.random.default_rng(0)
    cols = [c for c in C.columns if C[c].notna().sum() > 2000]
    cols = list(rng.choice(cols, size=min(120, len(cols)), replace=False))
    ctx, lastv, act = [], [], []
    for c in cols:
        s = C[c].dropna()
        for i in range(CONTEXT, len(s)-HORIZON, 252):        # ~1 sample/stock/year
            ctx.append(s.iloc[i-CONTEXT:i].values.astype("float32"))
            lastv.append(s.iloc[i-1]); act.append(s.iloc[i+HORIZON-1])
    lastv = np.array(lastv); act = np.array(act)
    preds = []
    for k in range(0, len(ctx), 256):
        chunk = ctx[k:k+256]
        preds.append(np.asarray(model.forecast(horizon=HORIZON,
                                               inputs=list(chunk))[0])[:len(chunk), -1])
    pred = np.concatenate(preds)
    results["B_stock"] = score("B  single-stock level, 21d ahead (as % return)",
                               pred/lastv-1, act/lastv-1,
                               {"naive (last value)": np.zeros(len(lastv))})

    # ---------------------------------------------------------------- C: volatility
    r = np.log(nifty).diff()
    rv = r.rolling(21).std()*np.sqrt(252)                    # annualised realised vol
    rv = rv.dropna()
    pts = list(range(CONTEXT, len(rv)-HORIZON, 21))
    ctx = [rv.iloc[i-CONTEXT:i].values.astype("float32") for i in pts]
    naive = np.array([rv.iloc[i-1] for i in pts])
    ew = r.ewm(alpha=1-0.94).std()*np.sqrt(252)
    ewma = np.array([ew.reindex(rv.index).iloc[i-1] for i in pts])
    # target: realised vol over the NEXT, non-overlapping 21 days
    fut = []
    for i in pts:
        d0 = rv.index[i-1]
        seg = r.loc[r.index > d0].iloc[:HORIZON]
        fut.append(seg.std()*np.sqrt(252) if len(seg) == HORIZON else np.nan)
    fut = np.array(fut)
    pred = np.asarray(model.forecast(horizon=HORIZON, inputs=list(ctx))[0])[:len(pts), -1]
    results["C_vol"] = score("C  NIFTY realised volatility, next 21d (annualised)",
                             pred, fut, {"naive (last vol)": naive, "EWMA(0.94)": ewma})

    pd.DataFrame(results).T.to_csv(BASE/"data_store/timesfm_usecases.csv")
    print("\n" + "="*72); print("SUMMARY"); print("="*72)
    for k, v in results.items():
        rs = {kk.replace("ratio_vs_", ""): vv for kk, vv in v.items() if kk.startswith("ratio_vs_")}
        best = min(rs.values())
        print(f"  {k:10} best ratio {best:.3f}  -> "
              f"{'TimesFM ADDS VALUE' if best < 0.98 else 'no better than a free baseline'}")
    print("\nsaved -> data_store/timesfm_usecases.csv")


if __name__ == "__main__":
    main()
