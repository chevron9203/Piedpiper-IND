"""
"Smart system" research: can a learned model pick Indian stocks better than momentum?

Everything on the survivorship-free NSE panel (build_nse_panel.py): delisted, merged and
trade-to-trade names are IN the universe, so the model has to learn to avoid them.

Design (each choice is there to stop the backtest lying):
  universe   point-in-time top-UNIV stocks by 60d median turnover, price >= PRICE_MIN,
             >= 250 days listed. No index membership, no hindsight.
  decision   every STEP trading days, using closes up to day t only
  execution  at day t+1 close (you cannot trade on a close you have not seen)
  label      forward H-day return from t+1 close, ranked cross-sectionally
  dead names if a stock never trades again, its last price minus DEAD_HAIRCUT is the exit
  model      LightGBM, refit every January on all data whose labels END before the test
             year starts (purged) -- walk-forward, never sees the future
  costs      RT round trip charged on every entry+exit in the portfolio simulation
  judge      portfolio level (not just IC): TimesFM had positive IC and still lost money

Run:  python scripts/research_smart_ml.py [--h 10] [--step 5] [--top 20]
"""
from __future__ import annotations
import argparse, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))
PANEL = BASE/"data_store/nse_panel.parquet"
FEAT_CACHE = BASE/"data_store/smart_features.parquet"

UNIV = 800; PRICE_MIN = 20.0; MIN_AGE = 250
RT = 2*(0.0010+0.0010)+0.0005        # same round-trip cost as the deployed research
DEAD_HAIRCUT = 0.30
TEST_FROM = 2013
TEST_TO = 9999


# ---------------------------------------------------------------- data
def load_panel():
    from scripts.build_nse_panel import etf_symbols
    from scripts.research_tranching import is_stock
    p = pd.read_parquet(PANEL)
    etf = etf_symbols()
    syms = [s for s in p["close"].columns if s not in etf and is_stock(s)]
    return {k: p[k][syms] for k in ("open","high","low","close","vol","turn","trades","dlv","t2t","ca_factor")}


def traded_price(P):
    """Price actually traded that day. Panel prices are back-adjusted for LATER splits and
    bonuses: fine for every ratio feature, but a price LEVEL then carries the future (a stock
    that will split 1:10 looks 10x cheaper years before it happens). 22.8% of universe days
    were >5% off and 295k stock-days were wrongly cut by the Rs-20 floor (audit 2026-10-03).
    The live window only knows past actions, so this is also what live sees."""
    C = P["close"]
    if "ca_factor" not in P:
        return C
    f = P["ca_factor"].reindex(index=C.index, columns=C.columns).fillna(1.0)
    after = f.iloc[::-1].cumprod().iloc[::-1]/f          # product of factors strictly after t
    return C/after


def xs_rank(df):
    return df.rank(axis=1, pct=True)


# ---------------------------------------------------------------- features
def base(P, age_offset=None):
    """Prices, point-in-time universe and market return -- cheap, never cached."""
    C, T = P["close"].ffill(limit=5), P["turn"]  # brief no-trade days, not a price change
    R = C.pct_change(fill_method=None)
    alive = P["close"].notna()
    turn60 = T.rolling(60, min_periods=40).median()
    age = alive.cumsum()
    if age_offset is not None:              # live: trading days listed BEFORE the price window
        age = age + age_offset.reindex(age.columns).fillna(0).values
    elig = (age >= MIN_AGE) & (traded_price(P).ffill(limit=5) >= PRICE_MIN) & alive & turn60.notna()
    trank = turn60.where(elig).rank(axis=1, ascending=False)
    univ = elig & (trank <= UNIV)
    # market = equal-weight universe return (cap-agnostic, exists for the whole sample)
    mkt = R.where(univ.shift(1, fill_value=False)).mean(axis=1)
    return C, R, alive, turn60, age, univ, mkt


def compute_features(P, step, dates=None, age_offset=None, use_cache=True):
    """dates/age_offset/use_cache=False: the live service passes a short price window and
    asks for features on explicit decision dates only (identical maths, no disk cache)."""
    cache = FEAT_CACHE.with_name(f"smart_features_s{step}.parquet")
    C, R, alive, turn60, age, univ, mkt = base(P, age_offset)
    if use_cache and dates is None and cache.exists() and cache.stat().st_mtime > PANEL.stat().st_mtime:
        return pd.read_parquet(cache), C, univ, mkt
    O, H, L, V, T = P["open"], P["high"], P["low"], P["vol"], P["turn"]
    mkt_c = (1 + mkt.fillna(0)).cumprod()

    F = {}
    for n in (1, 5, 21, 63, 126, 252):
        F[f"r{n}"] = C/C.shift(n) - 1
    F["r252_21"] = C.shift(21)/C.shift(252) - 1
    vol21 = R.rolling(21, min_periods=15).std()
    vol63 = R.rolling(63, min_periods=40).std()
    F["vol21"], F["vol63"] = vol21, vol63
    F["dvol63"] = R.clip(upper=0).pow(2).rolling(63, min_periods=40).mean().pow(0.5)
    F["vol_ratio"] = vol21/vol63
    # deployed-style risk-adjusted multi-timeframe momentum (rank average)
    F["mom_riskadj"] = (xs_rank(F["r63"]/vol63) + xs_rank(F["r126"]/vol63)
                        + xs_rank(F["r252"]/vol63))/3
    # lottery / MAX effect: largest daily return in the last month
    F["max5"] = R.rolling(21, min_periods=15).max()
    # beta / idio vol / relative strength on down vs up market days
    mr = mkt.values[:, None]
    Rv = R.values
    def roll_mean(x, w, mp):
        return pd.DataFrame(x, index=R.index, columns=R.columns).rolling(w, min_periods=mp).mean()
    cov = roll_mean(Rv*mr, 126, 80) - roll_mean(Rv, 126, 80)*mkt.rolling(126, min_periods=80).mean().values[:, None]
    var_m = mkt.rolling(126, min_periods=80).var().values[:, None]
    beta = cov/var_m
    F["beta126"] = beta
    resid = R - beta*mkt.values[:, None]
    F["ivol63"] = resid.rolling(63, min_periods=40).std()
    down = (mkt < -0.005).values[:, None]; up = (mkt > 0.005).values[:, None]
    ex = R.values - mr
    F["rs_down63"] = roll_mean(np.where(down, ex, np.nan), 63, 5)
    F["rs_up63"] = roll_mean(np.where(up, ex, np.nan), 63, 5)
    F["rs_down21"] = roll_mean(np.where(down, ex, np.nan), 21, 3)
    # price location / trend structure
    hi252 = H.ffill(limit=5).rolling(252, min_periods=200).max()
    lo252 = L.ffill(limit=5).rolling(252, min_periods=200).min()
    F["d52h"] = C/hi252 - 1
    F["d52l"] = C/lo252 - 1
    sma50 = C.rolling(50, min_periods=40).mean(); sma200 = C.rolling(200, min_periods=150).mean()
    F["c_sma50"] = C/sma50 - 1
    F["c_sma200"] = C/sma200 - 1
    F["sma50_slope"] = sma50/sma50.shift(21) - 1
    tr = np.maximum(H - L, np.maximum((H - C.shift(1)).abs(), (L - C.shift(1)).abs()))
    atr10 = tr.rolling(10, min_periods=7).mean(); atr63 = tr.rolling(63, min_periods=40).mean()
    F["atr_contract"] = atr10/atr63
    rng = (H - L).replace(0, np.nan)
    F["clv5"] = (((C - L) - (H - C))/rng).rolling(5, min_periods=3).mean()
    F["clv21"] = (((C - L) - (H - C))/rng).rolling(21, min_periods=15).mean()
    gap = O/C.shift(1) - 1
    F["gap_max21"] = gap.rolling(21, min_periods=15).max()
    # volume / liquidity / accumulation
    F["log_turn60"] = np.log1p(turn60)
    F["turn_surge"] = T.rolling(21, min_periods=15).mean()/T.rolling(126, min_periods=80).mean()
    F["vol_surge5"] = V.rolling(5, min_periods=3).mean()/V.rolling(63, min_periods=40).mean()
    F["amihud63"] = (R.abs()/T.replace(0, np.nan)).rolling(63, min_periods=40).mean()
    upv = V.where(R > 0, 0); F["upvol21"] = upv.rolling(21, min_periods=15).sum()/V.rolling(21, min_periods=15).sum()
    # obv-style: volume-weighted direction over a quarter
    F["vwdir63"] = (np.sign(R)*T).rolling(63, min_periods=40).sum()/T.rolling(63, min_periods=40).sum()
    # delivery (institutional footprint) -- from MTO, available 2007+
    dp = (P["dlv"]/V).clip(0, 1)
    F["dlv21"] = dp.rolling(21, min_periods=10).mean()
    F["dlv_chg"] = dp.rolling(5, min_periods=3).mean() - dp.rolling(63, min_periods=30).mean()
    dlv_val = (P["dlv"]*C)
    F["dlv_turn_surge"] = dlv_val.rolling(21, min_periods=10).mean()/dlv_val.rolling(126, min_periods=60).mean()
    F["dlv_dir21"] = (np.sign(R)*dlv_val).rolling(21, min_periods=10).sum()/dlv_val.rolling(21, min_periods=10).sum()
    # trade size (2011+): bigger average ticket = more institutional
    qpt = V/P["trades"].replace(0, np.nan)
    F["qpt_chg"] = np.log(qpt.rolling(21, min_periods=10).mean()/qpt.rolling(126, min_periods=60).mean())
    # distress / age / price level
    F["t2t"] = P["t2t"].fillna(0)
    F["t2t_days63"] = P["t2t"].fillna(0).rolling(63, min_periods=1).sum()
    F["log_age"] = np.log1p(age)
    F["log_price"] = np.log(traded_price(P).ffill(limit=5))

    # sample on decision dates only (every `step` days) -> long format
    dates = C.index[260::step] if dates is None else pd.DatetimeIndex(dates)
    U = univ.reindex(dates)
    keys = U.stack()
    keys = keys[keys].index                       # (date, sym) pairs in the universe
    X = pd.DataFrame({k: v.reindex(dates).stack(future_stack=True).reindex(keys).astype("float32")
                      for k, v in F.items()})
    X.index = X.index.set_names(["date", "sym"])

    # market-state features (same value for every stock on a date)
    breadth = (C > sma200).where(univ).sum(axis=1)/univ.sum(axis=1).replace(0, np.nan)
    disp = F["r21"].where(univ).std(axis=1)
    M = pd.DataFrame({
        "m_r21": mkt_c/mkt_c.shift(21) - 1,
        "m_r63": mkt_c/mkt_c.shift(63) - 1,
        "m_vol21": mkt.rolling(21).std(),
        "m_breadth": breadth,
        "m_disp": disp,
        "m_d52h": mkt_c/mkt_c.rolling(252).max() - 1,
    }).reindex(dates).rename_axis("date")
    X = X.join(M, on="date").astype("float32")
    if use_cache and len(dates) > 1:
        X.to_parquet(cache)
    return X, C, univ, mkt


RANKED_EXCLUDE = {"t2t", "ear_days", "fo_member", "fund_age", "order_size63"}    # flags / clocks / sparse stay raw


SPARSE_RAW = set()          # --raw-sparse: insider/promoter flows (mostly zero) also stay raw


def rank_features(X):
    # sparse event counts (ann_*) stay raw too: ranking a mostly-zero column gives "0" a
    # different value every week (depends on how many OTHER stocks had the event) = noise
    feats = [c for c in X.columns if not c.startswith(("m_", "ann_")) and c not in RANKED_EXCLUDE | SPARSE_RAW]
    Xr = X.copy()
    Xr[feats] = X[feats].groupby(level="date").rank(pct=True).astype("float32")
    return Xr


# ---------------------------------------------------------------- labels
def forward_returns(C, dates, h):
    """Return from close[t+1] to close[t+1+h]; dead names exit at last price - haircut."""
    idx = C.index
    pos = idx.get_indexer(dates)
    last_valid = C.apply(lambda s: s.last_valid_index())
    Cf = C.ffill()
    out = {}
    for d, i in zip(dates, pos):
        if i + 1 >= len(idx): continue
        j = min(i + 1 + h, len(idx) - 1)
        if j == i + 1: continue
        p0 = Cf.iloc[i + 1]; p1 = Cf.iloc[j]
        r = p1/p0 - 1
        dead = last_valid < idx[j]
        # only names that truly stop trading forever (not just a gap) get the haircut
        dead &= last_valid < idx[-1] - pd.Timedelta(days=30)
        r = r.where(~dead, (1 + r)*(1 - DEAD_HAIRCUT) - 1)
        out[d] = r
    y = pd.DataFrame(out).T.stack().dropna().rename("fwd")
    # names must match X's index, else join() aligns on "sym" alone (cartesian, ~100GB)
    return y.rename_axis(["date", "sym"])


# ---------------------------------------------------------------- model
def walk_forward(X, y, h, step, cal, halflife=None, seeds=(7,), lambdarank=False, tmode="rank", train_dates=None,
                 vol=None, lgb_over=None, rounds=400):
    import lightgbm as lgb
    df = X.join(y, how="inner")
    df["target"] = df.groupby(level="date")["fwd"].rank(pct=True)
    if tmode == "volscaled":     # RISK-ADJUSTED label: rank of return per unit of trailing volatility (calmer winners)
        v_ = vol.reindex(df.index).clip(lower=0.005)
        df["target"] = (df["fwd"]/v_).groupby(level="date").rank(pct=True)
    if tmode == "topq":          # right-tail classifier: lands in the top 10% of the day
        df["target"] = (df["target"] >= 0.90).astype(float)
    elif tmode == "winsor":      # SIZE of the move, robustly scaled and capped (big winners count)
        g = df.groupby(level="date")["fwd"]
        med = g.transform("median"); mad = (df["fwd"] - med).abs().groupby(level="date").transform("median")
        df["target"] = ((df["fwd"] - med)/(1.4826*mad.replace(0, np.nan))).clip(-3, 3).fillna(0.0)
    feats = [c for c in X.columns]
    dates = df.index.get_level_values("date")
    preds = []; imps = []
    params = dict(objective="binary" if tmode == "topq" else "regression", learning_rate=0.03, num_leaves=31,
                  min_data_in_leaf=400, feature_fraction=0.7, bagging_fraction=0.7,
                  bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=8, seed=7)
    if lgb_over:
        params.update(lgb_over)
    for yr in range(TEST_FROM, min(dates.max().year, TEST_TO) + 1):
        start = pd.Timestamp(f"{yr}-01-01")
        # purge in TRADING days: a training row's label ends at cal[pos+1+h]; it must end
        # before the test year starts (BDay ignored NSE holidays -> a few days of leak)
        first = cal.searchsorted(start)
        purge = cal[max(first - h - 2, 0)]
        trm = dates < purge
        if train_dates is not None:       # daily predictions, but train on the same sparse grid as v1.1
            trm = trm & dates.isin(train_dates)
        tr = df[trm]; te = df[(dates >= start) & (dates < pd.Timestamp(f"{yr+1}-01-01"))]
        if len(te) == 0: continue
        # recency weighting: the edge decays as the market learns -- let recent years count more
        wt = None
        if halflife:
            age = (start - tr.index.get_level_values("date")).days.values/365.25
            wt = 0.5**(age/halflife)
        ps = []
        if lambdarank:                # optimise the TOP of each date's list (we only buy 20)
            tr = tr.sort_index(level="date")
            grp = tr.groupby(level="date", sort=False).size().values
            rel = np.minimum((tr["target"]*30).astype(int), 29)        # 30 relevance grades
        for sd in seeds:
            if lambdarank:
                m = lgb.train({**params, "seed": sd, "objective": "lambdarank", "lambdarank_truncation_level": 60,
                               "label_gain": list(range(30)), "eval_at": [20]},
                              lgb.Dataset(tr[feats], rel, group=grp), num_boost_round=rounds)
            else:
                m = lgb.train({**params, "seed": sd}, lgb.Dataset(tr[feats], tr["target"], weight=wt),
                              num_boost_round=rounds)
            ps.append(m.predict(te[feats]))
        p = pd.Series(np.mean(ps, axis=0), index=te.index, name="pred")
        preds.append(p)
        imps.append(pd.Series(m.feature_importance("gain"), index=feats, name=yr))
    return pd.concat(preds), pd.concat(imps, axis=1)


# ---------------------------------------------------------------- evaluation
def ic_report(score, y, label):
    d = pd.concat([score.rename("s"), y], axis=1, join="inner")
    ic = d.groupby(level="date").apply(lambda g: g["s"].corr(g["fwd"], method="spearman"))
    def q(g):
        k = max(len(g)//10, 5)
        g = g.sort_values("s")
        return g["fwd"].iloc[-k:].mean() - g["fwd"].mean()
    top = d.groupby(level="date").apply(q)
    yr = ic.groupby(ic.index.year).mean()
    print(f"  {label:<14} IC {ic.mean():+.4f} (t={ic.mean()/ic.std()*np.sqrt(len(ic)):+.1f})  "
          f"top-decile excess {top.mean()*100:+.2f}%/period (t={top.mean()/top.std()*np.sqrt(len(top)):+.1f})  "
          f"IC>0 in {(yr>0).sum()}/{len(yr)} yrs")
    return ic, top


FEES_1WAY = 0.0012     # STT 0.1% + exchange/SEBI/stamp/GST, delivery, per side


def cost_model(P, capital, top_n, k_impact=1.0):
    """One-way cost fraction per (date, sym), knowable at t: fees + half-spread + impact.
      half-spread  Abdi-Ranaldo (2017) close/high/low estimator, 63d, floored at 0.05%
      impact       k * daily vol * sqrt(order / 60d median turnover)
    Shifted one day: the cost of trading at t+1 uses estimates through t."""
    C, H, L = (np.log(P[k].ffill(limit=5)) for k in ("close", "high", "low"))
    eta = (H + L)/2
    s2 = 4*((C - eta)*(C - eta.shift(-1))).shift(1)          # uses t+1 mid -> lag one more day
    # average first, clip after: clipping daily products to 0 throws away the negative
    # half of the noise and inflates spreads several-fold
    spread = np.sqrt(s2.rolling(63, min_periods=30).mean().clip(lower=0))
    half = (spread/2).clip(lower=0.0005, upper=0.03)
    R = P["close"].ffill(limit=5).pct_change(fill_method=None)
    vol = R.rolling(63, min_periods=30).std()
    adv = P["turn"].rolling(60, min_periods=40).median()
    impact = k_impact*vol*np.sqrt((capital/top_n)/adv)
    return (FEES_1WAY + half + impact.clip(upper=0.05)).shift(1)


def simulate(score, C, univ, top_n, exit_rank, step, cost=None, log=None):
    """Daily NAV of an equal-weight book: decide at t, trade at t+1 close, hold through.
    Hysteresis: keep a holding while its rank <= exit_rank; fill free slots from the top.
    cost: optional (date x sym) one-way cost fraction (cost_model); else flat RT/2.
    log:  optional list; gets ("trade", date, sym, dweight, cost) and ("hold", date, sym, w, r) rows."""
    Cf = C.ffill(limit=5)
    R = Cf.pct_change(fill_method=None)
    last_valid = C.apply(lambda s: s.last_valid_index())
    idx = C.index
    # need a t+1 close to trade on
    sdates = [d for d in sorted(set(score.index.get_level_values("date"))) if idx.get_loc(d) + 1 < len(idx)]
    by_date = {d: g.droplevel(0).sort_values(ascending=False) for d, g in score.groupby(level="date")}
    nav = [1.0]; navd = [idx[idx.get_loc(sdates[0]) + 1]]
    w = pd.Series(dtype=float)                 # current (drifted) weights; 1-sum = cash
    turnover = 0.0; trades = 0
    for k, d in enumerate(sdates):
        s = by_date[d]
        rank = pd.Series(np.arange(1, len(s) + 1), index=s.index)
        hold = list(w.index)
        keep = [x for x in hold if x in rank.index and rank[x] <= exit_rank]
        new = [x for x in s.index if x not in keep][: top_n - len(keep)]
        target = pd.Series(1.0/top_n, index=keep + new)
        # cost on every unit of weight that changes hands (entries, exits AND re-weights)
        dwi = target.sub(w, fill_value=0).abs()
        dw = dwi.sum()
        i0 = idx.get_loc(d) + 1
        if cost is None:
            c = dw*RT/2
        else:
            c = (dwi*cost.iloc[i0].reindex(dwi.index).fillna(0.01)).sum()
        turnover += dw/2; trades += len(set(hold) ^ set(target.index))
        if log is not None:
            for x, dx in target.sub(w, fill_value=0).items():
                if abs(dx) > 1e-9:
                    cx = abs(dx)*(RT/2 if cost is None else (cost.iloc[i0].get(x, 0.01) if pd.notna(cost.iloc[i0].get(x, np.nan)) else 0.01))
                    log.append(("trade", idx[i0], x, dx, cx))
        w = target
        i1 = idx.get_loc(sdates[k + 1]) + 1 if k + 1 < len(sdates) else len(idx) - 1
        v = nav[-1]*(1 - c)
        for i in range(i0 + 1, i1 + 1):
            r = R.iloc[i].reindex(w.index)
            # a name that died: book it once at -haircut, then it is cash
            died = [x for x in w.index if pd.isna(r[x]) and last_valid[x] < idx[i]
                    and last_valid[x] < idx[-1] - pd.Timedelta(days=30)]
            r = r.fillna(0.0)
            for x in died:
                r[x] = -DEAD_HAIRCUT
            pr = (w*r).sum()
            if log is not None:
                for x in w.index:
                    log.append(("hold", idx[i], x, w[x], r[x]))
            v *= 1 + pr
            w = w*(1 + r)/(1 + pr) if (1 + pr) != 0 else w
            if died:
                w = w.drop(died)
            nav.append(v); navd.append(idx[i])
    nav = pd.Series(nav, index=navd)
    yrs = (nav.index[-1] - nav.index[0]).days/365.25
    return nav, turnover/yrs, trades


def stats(nav):
    r = nav.pct_change().dropna()
    yrs = (nav.index[-1] - nav.index[0]).days/365.25
    cagr = nav.iloc[-1]**(1/yrs) - 1
    dd = (nav/nav.cummax() - 1).min()
    sh = r.mean()/r.std()*np.sqrt(252)
    return cagr, dd, sh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=10)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--exit", type=int, default=60)
    ap.add_argument("--events", action="store_true", help="add earnings-reaction + insider features")
    ap.add_argument("--peers", action="store_true", help="add learned peer-group features")
    ap.add_argument("--fo", action="store_true", help="add F&O positioning features")
    ap.add_argument("--fund", action="store_true", help="add XBRL fundamental features (2019+)")
    ap.add_argument("--ann", action="store_true", help="add corporate-announcement event features")
    ap.add_argument("--raw-sparse", action="store_true", help="keep insider/promoter flows unranked")
    ap.add_argument("--no-insider", action="store_true",
                    help="drop insider/promoter features (NSE PIT feed empty since 2026-05)")
    ap.add_argument("--halflife", type=float, default=None, help="recency weight half-life, years")
    ap.add_argument("--seeds", type=int, default=1, help="average this many LightGBM seeds")
    ap.add_argument("--no-mkt", action="store_true", help="drop market-state features (m_*)")
    ap.add_argument("--target", default="rank", choices=["rank", "topq", "winsor", "volscaled"], help="training target")
    ap.add_argument("--lgb", default=None, help="LightGBM overrides, e.g. num_leaves=15,min_data_in_leaf=800,feature_fraction=0.5")
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--ptag", default=None, help="name suffix for the prediction file")
    ap.add_argument("--macro", action="store_true", help="add date-level macro/regime features (research_smart_macro.py)")
    ap.add_argument("--liq-max", type=int, default=None,
                    help="SPECIALIST universe: keep only the N most liquid stocks per date (ranks/targets are then within that set)")
    ap.add_argument("--predict-all", action="store_true",
                    help="features for EVERY trading day (step 1); train on the every-5th-day grid, predict all days")
    ap.add_argument("--preds-only", action="store_true", help="stop after saving predictions")
    ap.add_argument("--test-from", type=int, default=None, help="first walk-forward test year (default 2013)")
    ap.add_argument("--test-to", type=int, default=None, help="last walk-forward test year")
    ap.add_argument("--lambdarank", action="store_true", help="LambdaRank objective (top-of-list)")
    ap.add_argument("--shuffle", action="store_true",
                    help="AUDIT: permute labels across stocks within each date; any edge left = leak")
    a = ap.parse_args()
    global TEST_FROM, TEST_TO
    if a.test_from: TEST_FROM = a.test_from
    if a.test_to: TEST_TO = a.test_to
    t0 = time.time()
    P = load_panel()
    fstep = 1 if a.predict_all else a.step         # feature grid; training grid stays `a.step`
    X, C, univ, mkt = compute_features(P, fstep)
    if a.events:
        X = X.join(pd.read_parquet(BASE/f"data_store/smart_events_s{fstep}.parquet"))
    if a.peers:
        X = X.join(pd.read_parquet(BASE/f"data_store/smart_peers_s{fstep}.parquet"))
    if a.fo:
        X = X.join(pd.read_parquet(BASE/f"data_store/smart_fo_s{fstep}.parquet"))
    if a.fund:
        X = X.join(pd.read_parquet(BASE/f"data_store/smart_fund_s{fstep}.parquet"))
    if a.ann:
        X = X.join(pd.read_parquet(BASE/f"data_store/smart_ann_s{fstep}.parquet"))
    if a.macro:
        X = X.join(pd.read_parquet(BASE/"data_store/smart_macro.parquet"), on="date")
    if a.raw_sparse:
        SPARSE_RAW.update({"prom_net", "ins_net", "ins_buyers"})
    if a.no_insider:
        X = X.drop(columns=[c for c in ("prom_net", "ins_net", "ins_buyers") if c in X.columns])
    if a.no_mkt:
        X = X.drop(columns=[c for c in X.columns if c.startswith("m_")])
    if a.liq_max:
        _liq = X["log_turn60"].groupby(level="date").rank(ascending=False)
        X = X[_liq <= a.liq_max]
        print(f"specialist universe: top {a.liq_max} by liquidity, {len(X):,} rows, {len(X)//X.index.get_level_values('date').nunique()} per date")
    print(f"features {X.shape} in {time.time()-t0:.0f}s")
    vol_raw = X["vol63"] if "vol63" in X.columns else None
    lgb_over = {k: (int(v) if v.lstrip("-").isdigit() else float(v)) for k, v in (kv.split("=") for kv in a.lgb.split(","))} if a.lgb else None
    Xr = rank_features(X)
    y = forward_returns(C, Xr.index.get_level_values("date").unique(), a.h)
    if a.shuffle:
        rng = np.random.default_rng(0)
        y = y.groupby(level="date", group_keys=False).apply(
            lambda g: pd.Series(rng.permutation(g.values), index=g.index)).rename("fwd")
    grid = set(C.index[260::a.step]) if a.predict_all else None
    pred, imp = walk_forward(Xr, y, a.h, a.step, C.index, a.halflife, tuple(range(7, 7 + a.seeds)), a.lambdarank, a.target, grid,
                           vol=vol_raw, lgb_over=lgb_over, rounds=a.rounds)
    if a.preds_only:
        tg = f"h{a.h}_s{a.step}" + ("_ev" if a.events else "") + ("_pr" if a.peers else "") + ("_ni" if a.no_insider else "") + (f"_L{a.liq_max}" if a.liq_max else "") \
            + (f"_x{a.seeds}" if a.seeds > 1 else "") + ("_px_d1" if a.predict_all else "_px") + (f"_{a.ptag}" if a.ptag else "") \
            + (f"_tf{TEST_FROM}-{TEST_TO}" if (a.test_from or a.test_to) else "")
        pred.to_frame().to_parquet(BASE/f"data_store/smart_ml_pred_{tg}.parquet")
        print(f"saved daily predictions {tg}: {len(pred):,} rows, {pred.index.get_level_values('date').nunique()} dates, {time.time()-t0:.0f}s")
        return
    print(f"model done {time.time()-t0:.0f}s\n")
    yt = y[y.index.get_level_values("date").year >= TEST_FROM]
    print(f"== IC / top-decile, OOS {TEST_FROM}+, horizon {a.h}d (walk-forward) ==")
    ic_report(pred, yt, "ML model")
    singles = ["mom_riskadj", "r252_21", "d52h", "rs_down63", "dlv_chg", "turn_surge", "r21", "max5", "ivol63"]
    if a.events: singles += ["ear", "prom_net", "ins_net"]
    if a.peers: singles += ["peer_r21", "peer_mom", "rel_r63"]
    if a.fo: singles += ["oi_chg21", "oi_px5", "basis_ann", "pcr", "fut_spec"]
    if a.fund: singles += ["rev_yoy", "pat_yoy", "opm_chg", "sue", "rev_accel"]
    if a.ann: singles += ["ann_pledge", "ann_default", "ann_clarify", "ann_fund_raise"]
    for f in [f for f in singles if f in Xr.columns]:
        sgn = -1 if f in ("max5", "ivol63") else 1
        ic_report(sgn*Xr.loc[Xr.index.get_level_values("date").year >= TEST_FROM, f], yt, f + ("(-)" if sgn < 0 else ""))
    print("\n== feature importance (mean gain share, top 20) ==")
    gi = (imp/imp.sum()).mean(axis=1).sort_values(ascending=False)
    print("  " + ", ".join(f"{k} {v:.1%}" for k, v in gi.head(20).items()))

    print(f"\n== portfolio: top {a.top}, exit rank {a.exit}, every {a.step}d, RT {RT:.2%} ==")
    mom = Xr.loc[Xr.index.get_level_values("date").year >= TEST_FROM, "mom_riskadj"]
    mom = mom[mom.index.get_level_values("date").isin(pred.index.get_level_values("date"))]  # same decision dates
    res = {}
    for name, sc in (("ML model", pred), ("momentum (same mechanics)", mom)):
        nav, to, nt = simulate(sc, C, univ, a.top, a.exit, a.step)
        res[name] = nav
        cg, dd, sh = stats(nav)
        print(f"  {name:<28} CAGR {cg:+.1%}  MaxDD {dd:.1%}  Sharpe {sh:.2f}  turnover {to:.0%}/yr")
    ew = (1 + mkt.loc[res['ML model'].index[0]:].fillna(0)).cumprod()
    cg, dd, sh = stats(ew); print(f"  {'equal-wt universe':<28} CAGR {cg:+.1%}  MaxDD {dd:.1%}  Sharpe {sh:.2f}")
    out = pd.DataFrame(res); out["ew"] = ew
    yr = out.resample("YE").last().pct_change()
    yr.iloc[0] = out.resample("YE").last().iloc[0]/out.iloc[0] - 1
    print("\n== by calendar year ==")
    print((yr*100).round(1).rename(index=lambda d: d.year).to_string())
    tag = f"h{a.h}_s{a.step}" + ("_ev" if a.events else "") + ("_pr" if a.peers else "") + ("_fo" if a.fo else "") + ("_fd" if a.fund else "") + ("_an" if a.ann else "") + ("_rs" if a.raw_sparse else "") + ("_ni" if a.no_insider else "") + ("_nm" if a.no_mkt else "") + ("_lr" if a.lambdarank else "") + ("" if a.target == "rank" else f"_t{a.target}") + "_px" + ("_shuf" if a.shuffle else "") \
        + (f"_hl{a.halflife:g}" if a.halflife else "") + (f"_x{a.seeds}" if a.seeds > 1 else "")
    out.to_parquet(BASE/f"data_store/smart_ml_nav_{tag}.parquet")
    pred.to_frame().to_parquet(BASE/f"data_store/smart_ml_pred_{tag}.parquet")


if __name__ == "__main__":
    main()
