"""
F&O positioning features for the smart system (the ~200 F&O stocks = the liquid names,
exactly where the price/volume model's edge is weakest).

Step 1  parse every F&O bhavcopy (old format <= 2024-07-05, UDiFF after) into one
        daily stock-level table  data_store/fo_daily.parquet:
          fut_oi    open interest summed over all expiries (shares)
          fut_val   futures traded value (Rs)
          fut_near  near-month futures settle/close;  dte = days to that expiry
          call_oi / put_oi   option open interest (shares), opt_val traded value
Step 2  features on decision dates, closes up to t only:
          fo_member      stock has F&O on day t
          oi_chg5/21     log change in futures OI (fresh positions vs unwinding)
          oi_px5         r5 * oi_chg5: long build-up (+,+) and short covering (+,-)
                         separate from short build-up (-,+) and long unwinding (-,-)
          basis_ann      annualised next-vs-near future spread (crowded longs / carry);
                         F&O prices are raw and the panel is adjusted, so never vs spot
          pcr / pcr_chg5 put/call OI ratio and its change (hedging demand)
          fut_spec       futures value / cash value, 21d (speculative interest)
          oi_turn        futures OI value / 60d cash turnover (how levered the name is)

Run:  python scripts/research_smart_fo.py [--step 5] [--parse-only]
"""
from __future__ import annotations
import argparse, io, sys, zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))
FO = BASE/"data_store/nse_raw/fo"
DAILY = BASE/"data_store/fo_daily.parquet"


def parse_one(p: Path):
    try:
        z = zipfile.ZipFile(p)
        df = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), low_memory=False)
    except Exception:
        return None
    df.columns = [c.strip() for c in df.columns]
    d = pd.Timestamp(p.name[:8])
    if "INSTRUMENT" in df.columns:                              # old format
        df = df[df["INSTRUMENT"].isin(["FUTSTK", "OPTSTK"])]
        x = pd.DataFrame({"sym": df["SYMBOL"].str.strip(),
                          "fut": df["INSTRUMENT"].eq("FUTSTK"),
                          "exp": pd.to_datetime(df["EXPIRY_DT"], format="%d-%b-%Y", errors="coerce"),
                          "opt": df["OPTION_TYP"].str.strip(),
                          "close": pd.to_numeric(df["SETTLE_PR"], errors="coerce"),
                          "oi": pd.to_numeric(df["OPEN_INT"], errors="coerce"),
                          "val": pd.to_numeric(df["VAL_INLAKH"], errors="coerce")*1e5})
    else:                                                       # UDiFF
        df = df[df["FinInstrmTp"].isin(["STF", "STO"])]
        x = pd.DataFrame({"sym": df["TckrSymb"].str.strip(),
                          "fut": df["FinInstrmTp"].eq("STF"),
                          "exp": pd.to_datetime(df["XpryDt"], errors="coerce"),
                          "opt": df["OptnTp"].fillna("XX").astype(str).str.strip(),
                          "close": pd.to_numeric(df["SttlmPric"], errors="coerce"),
                          "oi": pd.to_numeric(df["OpnIntrst"], errors="coerce"),
                          "val": pd.to_numeric(df["TtlTrfVal"], errors="coerce")})
    if x.empty:
        return None
    f = x[x["fut"]]
    o = x[~x["fut"]]
    g = f.groupby("sym")
    fs = f.sort_values("exp")
    near = fs.groupby("sym").nth(0).set_index("sym")
    nxt = fs.groupby("sym").nth(1).set_index("sym")
    out = pd.DataFrame({"fut_oi": g["oi"].sum(), "fut_val": g["val"].sum(),
                        "fut_near": near["close"], "fut_next": nxt["close"],
                        "gap": (nxt["exp"] - near["exp"]).dt.days, "dte": (near["exp"] - d).dt.days})
    out["call_oi"] = o[o["opt"] == "CE"].groupby("sym")["oi"].sum()
    out["put_oi"] = o[o["opt"] == "PE"].groupby("sym")["oi"].sum()
    out["opt_val"] = o.groupby("sym")["val"].sum()
    out["date"] = d
    return out.reset_index()


def parse_all():
    files = sorted(p for p in FO.glob("*.zip"))
    with ProcessPoolExecutor(6) as ex:
        parts = [r for r in ex.map(parse_one, files, chunksize=20) if r is not None]
    D = pd.concat(parts, ignore_index=True)
    from scripts.build_nse_panel import apply_renames, symbol_changes
    D = apply_renames(D, symbol_changes())
    # renamed duplicates on one day: add quantities, keep the first price/expiry
    agg = {k: "sum" for k in ("fut_oi", "fut_val", "call_oi", "put_oi", "opt_val")}
    agg.update(fut_near="first", fut_next="first", gap="first", dte="first")
    D = D.groupby(["date", "sym"]).agg(agg).reset_index()
    D.to_parquet(DAILY)
    return D


def build(step):
    from scripts import research_smart_ml as SM
    D = pd.read_parquet(DAILY) if DAILY.exists() else parse_all()
    P = SM.load_panel()
    X, C, univ, mkt = SM.compute_features(P, step)
    idx, cols = C.index, C.columns
    W = {k: D.pivot_table(index="date", columns="sym", values=k, aggfunc="sum")
          .reindex(index=idx, columns=cols) for k in
         ("fut_oi", "fut_val", "fut_near", "fut_next", "gap", "call_oi", "put_oi")}
    member = W["fut_oi"].notna() & (W["fut_oi"] > 0)
    oi = W["fut_oi"].where(member)
    lo = np.log(oi)
    # a split/bonus multiplies share-OI overnight: clip log changes so it can't fake a build-up
    oi_chg5 = (lo - lo.shift(5)).clip(-1.5, 1.5)
    oi_chg21 = (lo - lo.shift(21)).clip(-2, 2)
    r5 = C/C.shift(5) - 1
    # F&O prices are RAW, the equity panel is split/bonus-ADJUSTED: never mix them.
    # carry = next-month vs near-month future (raw/raw), annualised over the expiry gap
    gap = W["gap"].where(member).clip(lower=7)
    basis_ann = ((W["fut_next"]/W["fut_near"] - 1)*365/gap).where(member).clip(-1, 1)
    pcr = (W["put_oi"]/W["call_oi"].replace(0, np.nan)).where(member).clip(0, 5)
    cash_val = P["turn"]
    fut_spec = (W["fut_val"].rolling(21, min_periods=10).sum()
                / cash_val.rolling(21, min_periods=10).sum()).where(member).clip(0, 50)
    oi_turn = (oi*W["fut_near"]/cash_val.rolling(60, min_periods=40).median()).clip(0, 200)
    F = {"fo_member": member.astype(float), "oi_chg5": oi_chg5, "oi_chg21": oi_chg21,
         "oi_px5": r5*oi_chg5, "basis_ann": basis_ann, "pcr": pcr,
         "pcr_chg5": pcr - pcr.shift(5), "fut_spec": fut_spec, "oi_turn": oi_turn}
    dates = X.index.get_level_values("date").unique()
    out = pd.DataFrame({k: v.reindex(dates).stack(future_stack=True).reindex(X.index)
                        for k, v in F.items()}).astype("float32")
    out["fo_member"] = out["fo_member"].fillna(0)
    out.to_parquet(BASE/f"data_store/smart_fo_s{step}.parquet")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--parse-only", action="store_true")
    a = ap.parse_args()
    if a.parse_only:
        D = parse_all(); print(D.shape, D["date"].min(), D["date"].max(), D["sym"].nunique(), "syms")
        print(D.groupby(D["date"].dt.year)["sym"].nunique().to_string())
    else:
        out = build(a.step)
        print("coverage:", {c: f"{out[c].notna().mean():.0%}" for c in out.columns})
        print("F&O share of universe rows:", f"{out['fo_member'].mean():.0%}")
        print(out[out.fo_member > 0].describe().T[["mean", "50%", "min", "max"]].round(3))
