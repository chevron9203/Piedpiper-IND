"""
NSE-S1 five-gate breakout-momentum signal generator for Indian equities.

Design rationale:
  Pullback entries (RSI 40-65) underperform on NSE because stocks that pull
  back in a trending market often continue lower past the 2×ATR stop before
  recovering. Breakout-momentum entries (price making a fresh 20-day high with
  volume and RSI still not overbought) align with how NSE large/mid-caps actually
  trend — and give better hit rates on the follow-through.

Gate definitions:
  G1  Market regime  (HARD VETO — no trades if G1 fails):
      India VIX < 22  AND  Nifty 50 > EMA(200)

  G2  Weekly trend:
      Stock weekly close > EMA(20w)  AND  weekly ADX(14) > 20

  G3  Breakout + momentum:
      Close > 20-day rolling-max of close (shift 1, no lookahead)   ← fresh 20d high
      AND  63d return > 0                                            ← positive momentum
      AND  close > 1.05 × 252d rolling-low                          ← not near 52w low

  G4  Entry confirmation:
      RSI(14) in (50, 80)                                            ← trending, not over-extended
      AND  volume ≥ 1.5 × 20d avg                                    ← above-average conviction
      AND  (close−low)/(high−low) ≥ 0.5                              ← closed in upper half of range

  G5  Quality filter:
      1% ≤ ATR(14)/close ≤ 5%                                       ← enough room to move, not reckless
      AND  close ≥ ₹100

Score = G2+G3+G4+G5  (G1 is veto, not scored).
Enter if score ≥ min_score (default 3 of 4).
Score 4 → 1.5× risk boost.

Sizing reference (done in backtest / forward runner):
  stop_loss  = close − sl_atr_mult × ATR(14)   (default 2×)
  target     = close + tp_atr_mult × ATR(14)   (default 4×)
  risk       = 2% equity (3% if score=4)
  max hold   = 20 trading days
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from features.volatility import atr as _calc_atr
from features.momentum import rsi as _calc_rsi
from features.regime import adx as _calc_adx


# ── helpers ───────────────────────────────────────────────────────────────────

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("W-FRI").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna(subset=["close"])


def _ffill_to_index(s: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """Forward-fill s onto target idx — no look-ahead."""
    combined = s.reindex(idx.union(s.index)).ffill()
    return combined.reindex(idx)


# ── main ──────────────────────────────────────────────────────────────────────

def generate_signals(
    ohlcv_dict: dict[str, pd.DataFrame],
    nifty_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    universe_symbols: list[str] | None = None,
    # G1 — hard veto
    vix_threshold: float = 22.0,
    ema_long: int = 200,
    # G2 — weekly trend
    weekly_ema_span: int = 20,
    weekly_adx_min: float = 20.0,
    # G3 — breakout + momentum (63-day = 3-month high; fresh = only first day of breakout)
    breakout_window: int = 63,
    mom_window: int = 63,
    fresh_only: bool = True,         # signal only on the FIRST day of a new 63d high
    # G4 — entry confirmation
    rsi_window: int = 14,
    rsi_low: float = 55.0,           # raised from 50 → only strong momentum
    rsi_high: float = 80.0,
    vol_window: int = 20,
    vol_mult: float = 2.0,           # raised from 1.5× → high-conviction volume
    candle_body_min: float = 0.50,
    # G5 — quality
    atr_window: int = 14,
    min_atr_pct: float = 0.01,
    max_atr_pct: float = 0.05,
    min_price: float = 100.0,
    # exit reference levels (2:1 R:R with more room)
    sl_atr_mult: float = 2.5,
    tp_atr_mult: float = 5.0,
    # scoring: G1 is hard veto; G2+G3+G4+G5 scored
    min_score: int = 4,              # require ALL 4 non-veto gates
) -> pd.DataFrame:
    """
    Compute NSE-S1 signals for all symbols across all available dates.

    G1 is a hard market-regime veto — no signals on days when G1 fails.
    G2+G3+G4+G5 scored 0-4; enter when score >= min_score.
    Score 4 (all non-veto gates) → 1.5× risk boost.

    Returns
    -------
    pd.DataFrame with columns:
        date, symbol, score, close, atr, stop_loss, target
    Sorted: date asc, score desc.
    """
    if universe_symbols is None:
        universe_symbols = list(ohlcv_dict.keys())

    symbols = [s for s in universe_symbols if s in ohlcv_dict]
    if not symbols:
        logger.warning("NSE-S1: no symbols in ohlcv_dict")
        return _empty()

    logger.info("NSE-S1: computing indicators for {} symbols ...", len(symbols))

    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d.columns = [c.lower() for c in d.columns]
        d.index = pd.to_datetime(d.index).normalize()
        return d.sort_index()

    nifty = _prep(nifty_df)
    vix   = _prep(vix_df)

    # ── per-symbol indicators ─────────────────────────────────────────────────
    sym_data: dict[str, dict] = {}
    for sym in symbols:
        df = _prep(ohlcv_dict[sym])
        if len(df) < 150:
            continue
        idx = df.index

        atr_s    = _calc_atr(df, window=atr_window)
        rsi_s    = _calc_rsi(df, window=rsi_window)
        vol_avg  = df["volume"].rolling(vol_window, min_periods=vol_window // 2).mean()
        ret63    = df["close"].pct_change(mom_window)
        low252   = df["close"].rolling(252, min_periods=126).min()

        # G3: N-day high breakout.  shift(1) so we compare today's close to
        # the max of the PREVIOUS breakout_window days (no lookahead).
        min_bp   = max(breakout_window // 2, 20)
        high_bo  = df["close"].shift(1).rolling(breakout_window, min_periods=min_bp).max()

        # fresh_only: only fire on the FIRST day of a new N-day high.
        # Yesterday's close must have been BELOW the N-day max of the day before that.
        prev_high = df["close"].shift(2).rolling(breakout_window - 1, min_periods=min_bp - 1).max()
        already_broke = df["close"].shift(1) > prev_high   # yesterday was already at N-day high

        # G4: candle body quality  (close - low) / (high - low)
        hl_range = df["high"] - df["low"]
        cl_body  = (df["close"] - df["low"]) / (hl_range.replace(0, np.nan))

        # G2: weekly trend
        weekly      = _resample_weekly(df)
        w_ema_daily = _ffill_to_index(_ema(weekly["close"], weekly_ema_span), idx)
        w_adx_daily = _ffill_to_index(_calc_adx(weekly, window=14)["adx"], idx)

        sym_data[sym] = dict(
            close=df["close"], volume=df["volume"],
            atr=atr_s, rsi=rsi_s, vol_avg=vol_avg,
            ret63=ret63, low252=low252,
            high_bo=high_bo, already_broke=already_broke, cl_body=cl_body,
            w_ema=w_ema_daily, w_adx=w_adx_daily,
        )

    if not sym_data:
        return _empty()

    active = list(sym_data.keys())

    # ── wide DataFrames (dates × symbols) ─────────────────────────────────────
    def _wide(key: str) -> pd.DataFrame:
        return pd.DataFrame({s: sym_data[s][key] for s in active})

    close_df        = _wide("close")
    volume_df       = _wide("volume")
    atr_df          = _wide("atr")
    rsi_df          = _wide("rsi")
    vol_avg_df      = _wide("vol_avg")
    ret63_df        = _wide("ret63")
    low252_df       = _wide("low252")
    high_bo_df      = _wide("high_bo")
    already_broke_df= _wide("already_broke")
    cl_body_df      = _wide("cl_body")
    w_ema_df        = _wide("w_ema")
    w_adx_df        = _wide("w_adx")

    # ── G1: hard regime veto ──────────────────────────────────────────────────
    idx = close_df.index
    nifty_close  = nifty["close"].reindex(idx, method="ffill")
    nifty_ema200 = _ema(nifty["close"], ema_long).reindex(idx, method="ffill")
    vix_close    = vix["close"].reindex(idx, method="ffill")

    g1_series = (vix_close < vix_threshold) & (nifty_close > nifty_ema200)

    # Nifty 63d return — used to compute each stock's relative strength vs index
    nifty_ret63 = nifty["close"].pct_change(mom_window).reindex(idx, method="ffill")
    # Broadcast to full matrix
    g1_df = pd.DataFrame(
        np.tile(g1_series.values.reshape(-1, 1), (1, len(active))),
        index=idx, columns=active, dtype=bool,
    )

    # ── G2–G5: scored gates ───────────────────────────────────────────────────
    g2_df = (close_df > w_ema_df) & (w_adx_df > weekly_adx_min)

    # G3: 63-day high breakout (fresh = first day only) + momentum
    is_breakout = close_df > high_bo_df
    if fresh_only:
        # fire only when yesterday was NOT already above its N-day high
        is_breakout = is_breakout & (~already_broke_df.fillna(False).astype(bool))
    g3_df = (
        is_breakout
        & (ret63_df > 0)
        & (close_df > 1.05 * low252_df)
    )

    g4_df = (
        (rsi_df > rsi_low)
        & (rsi_df < rsi_high)
        & (volume_df >= vol_mult * vol_avg_df)
        & (cl_body_df >= candle_body_min)
    )

    atr_pct = atr_df / close_df.replace(0, np.nan)
    g5_df = (atr_pct >= min_atr_pct) & (atr_pct <= max_atr_pct) & (close_df >= min_price)

    # Score = G2 + G3 + G4 + G5  (G1 is not scored)
    score_df = (
        g2_df.astype(int) + g3_df.astype(int)
        + g4_df.astype(int) + g5_df.astype(int)
    )

    # ── extract signals (G1 hard veto + score ≥ min_score) ───────────────────
    valid = g1_df & (score_df >= min_score) & close_df.notna() & atr_df.notna()
    r_idx, c_idx = np.where(valid.values)

    if len(r_idx) == 0:
        logger.warning("NSE-S1: no signals generated")
        return _empty()

    dates_out = idx[r_idx]
    syms_out  = np.array(active)[c_idx]
    scores    = score_df.values[r_idx, c_idx].astype(int)
    closes    = close_df.values[r_idx, c_idx]
    atrs      = atr_df.values[r_idx, c_idx]

    # RS vs Nifty: stock's 63d return minus Nifty's 63d return on that date
    nifty_ret63_aligned = nifty_ret63.reindex(idx)
    nifty_ret_out = nifty_ret63_aligned.values[r_idx]
    stock_ret63   = ret63_df.values[r_idx, c_idx]
    rs_out = stock_ret63 - nifty_ret_out  # positive = outperforming Nifty

    result = pd.DataFrame({
        "date":      dates_out,
        "symbol":    syms_out,
        "score":     scores,
        "close":     np.round(closes, 2),
        "atr":       np.round(atrs, 2),
        "stop_loss": np.round(closes - sl_atr_mult * atrs, 2),
        "target":    np.round(closes + tp_atr_mult * atrs, 2),
        "rs_nifty":  np.round(rs_out, 4),  # relative strength vs Nifty (higher = better)
    })

    result = result.sort_values(["date", "score"], ascending=[True, False]).reset_index(drop=True)
    logger.info("NSE-S1: {} signals | {} signal-dates | avg {:.1f}/day",
                len(result), result["date"].nunique(),
                len(result) / max(result["date"].nunique(), 1))
    return result


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "symbol", "score", "close", "atr", "stop_loss", "target"])
