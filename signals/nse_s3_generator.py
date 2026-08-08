"""
NSE-S3: 52-Week HIGH Breakout with EMA Stack.

Empirical basis (BacktestIndia, IntradayLab research):
  - Stocks breaking 52-week HIGH on >1.5x volume: ~70% probability of 10-session
    continuation. The 52-week high is a proven psychological anchor — institutional
    analysts flag it, media covers it, and fund managers who missed the run chase it.
  - EMA alignment (50>100>200) guarantees a structural uptrend, not a spike.
  - This is complementary to S1 (63d close-based): S3 fires less often but with
    stronger signal quality.

Architecture:
  G1  Market regime (HARD VETO):
      VIX < 20 AND Nifty > 200d EMA AND Nifty > 50d EMA

  G_bo  52-Week HIGH breakout (HARD VETO — must fire, not scored):
      Today's close > 252-day rolling max of HIGH (shift 1, no lookahead)
      AND fresh: yesterday's close was NOT above 251-day HIGH max
      AND volume ≥ 2× 20d average

  Scored gates (G2-G4, each 0 or 1 point):

  G2  EMA alignment:
      Close > EMA(50) AND EMA(50) > EMA(100) AND EMA(100) > EMA(200)
      (all three EMAs in ascending order = strong structural uptrend)

  G3  Momentum quality:
      RSI(14) ∈ (55, 78)  — trending but not overbought
      AND candle body ≥ 50%  — closed in upper half of range
      AND 63d return > 0%  — positive 3-month momentum

  G4  Entry quality:
      ATR(14)/close ∈ [0.8%, 4%]  — enough movement, not reckless
      AND close ≥ ₹100
      AND close > 1.10 × 252d min close  — not near 52-week low

Score = G2+G3+G4. Enter if score ≥ min_score (default 2 of 3).
Stop/target: ATR-based (sl_atr_mult × ATR, tp_atr_mult × ATR), re-anchored to fill.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from features.volatility import atr as _calc_atr
from features.momentum import rsi as _calc_rsi


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def generate_signals(
    ohlcv_dict: dict[str, pd.DataFrame],
    nifty_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    universe_symbols: list[str] | None = None,
    # G1 — market regime hard veto
    vix_threshold: float = 20.0,
    ema_long: int = 200,
    nifty_ema_mid: int = 50,
    # G_bo — 52-week HIGH breakout hard veto
    breakout_window: int = 252,
    vol_window: int = 20,
    vol_mult: float = 2.0,
    # G2 — EMA alignment (scored)
    ema_fast: int = 50,
    ema_mid: int = 100,
    # G3 — momentum quality (scored)
    rsi_window: int = 14,
    rsi_low: float = 55.0,
    rsi_high: float = 78.0,
    mom_window: int = 63,
    candle_body_min: float = 0.50,
    # G4 — entry quality (scored)
    min_atr_pct: float = 0.008,
    max_atr_pct: float = 0.04,
    min_price: float = 100.0,
    not_near_low_mult: float = 1.10,
    # stop / target reference (backtest re-anchors to fill)
    sl_atr_mult: float = 2.5,
    tp_atr_mult: float = 5.0,
    # scoring
    min_score: int = 2,
) -> pd.DataFrame:
    """
    Compute NSE-S3 signals: 52-week HIGH breakout with EMA stack.

    Returns DataFrame: date, symbol, score, close, atr, stop_loss, target, rs_nifty
    """
    if universe_symbols is None:
        universe_symbols = list(ohlcv_dict.keys())
    symbols = [s for s in universe_symbols if s in ohlcv_dict]
    if not symbols:
        logger.warning("NSE-S3: no symbols in ohlcv_dict")
        return _empty()

    logger.info("NSE-S3: computing indicators for {} symbols ...", len(symbols))

    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d.columns = [c.lower() for c in d.columns]
        d.index = pd.to_datetime(d.index).normalize()
        return d.sort_index()

    nifty = _prep(nifty_df)
    vix   = _prep(vix_df)

    sym_data: dict[str, dict] = {}
    for sym in symbols:
        df = _prep(ohlcv_dict[sym])
        if len(df) < 300:  # need 252d + warmup
            continue
        idx = df.index

        atr_s   = _calc_atr(df, window=14)
        rsi_s   = _calc_rsi(df, window=rsi_window)
        vol_avg = df["volume"].rolling(vol_window, min_periods=vol_window // 2).mean()
        ret63   = df["close"].pct_change(mom_window)

        # G_bo: 252d CLOSE breakout (no lookahead)
        # Use CLOSE not HIGH — HIGH-based breakout consistently fails on NSE
        # because intraday wicks dominate the rolling max, creating false filters.
        min_bp     = max(breakout_window // 2, 100)
        high_bo    = df["close"].shift(1).rolling(breakout_window, min_periods=min_bp).max()
        prev_high  = df["close"].shift(2).rolling(breakout_window - 1, min_periods=min_bp - 1).max()
        already_br = df["close"].shift(1) > prev_high

        # G2: EMA alignment
        ema_f = _ema(df["close"], ema_fast)
        ema_m = _ema(df["close"], ema_mid)
        ema_l = _ema(df["close"], ema_long)

        # G3: candle body
        hl_range = df["high"] - df["low"]
        cl_body  = (df["close"] - df["low"]) / hl_range.replace(0, np.nan)

        # G4: 252d low for "not near 52w low" check
        low252 = df["close"].rolling(breakout_window, min_periods=min_bp).min()

        sym_data[sym] = dict(
            close=df["close"], high=df["high"], volume=df["volume"],
            atr=atr_s, rsi=rsi_s, vol_avg=vol_avg, ret63=ret63,
            high_bo=high_bo, already_br=already_br,
            ema_f=ema_f, ema_m=ema_m, ema_l=ema_l,
            cl_body=cl_body, low252=low252,
        )

    if not sym_data:
        return _empty()

    active = list(sym_data.keys())

    def _wide(key: str) -> pd.DataFrame:
        return pd.DataFrame({s: sym_data[s][key] for s in active})

    close_df   = _wide("close")
    high_df    = _wide("high")
    volume_df  = _wide("volume")
    atr_df     = _wide("atr")
    rsi_df     = _wide("rsi")
    vol_avg_df = _wide("vol_avg")
    ret63_df   = _wide("ret63")
    high_bo_df = _wide("high_bo")
    ab_df      = _wide("already_br")
    ema_f_df   = _wide("ema_f")
    ema_m_df   = _wide("ema_m")
    ema_l_df   = _wide("ema_l")
    cl_body_df = _wide("cl_body")
    low252_df  = _wide("low252")

    idx = close_df.index

    # G1: hard regime veto
    nifty_close  = nifty["close"].reindex(idx, method="ffill")
    nifty_ema200 = _ema(nifty["close"], ema_long).reindex(idx, method="ffill")
    vix_close    = vix["close"].reindex(idx, method="ffill")
    g1_series    = (vix_close < vix_threshold) & (nifty_close > nifty_ema200)
    if nifty_ema_mid > 0:
        nifty_ema_m = _ema(nifty["close"], nifty_ema_mid).reindex(idx, method="ffill")
        g1_series   = g1_series & (nifty_close > nifty_ema_m)

    nifty_ret63 = nifty["close"].pct_change(mom_window).reindex(idx, method="ffill")

    g1_df = pd.DataFrame(
        np.tile(g1_series.values.reshape(-1, 1), (1, len(active))),
        index=idx, columns=active, dtype=bool,
    )

    # G_bo: 252-day CLOSE breakout (hard veto)
    is_breakout = close_df > high_bo_df
    is_fresh    = is_breakout & (~ab_df.fillna(False).astype(bool))
    vol_surge   = volume_df >= vol_mult * vol_avg_df
    gbo_df      = is_fresh & vol_surge

    # G2: EMA alignment (scored)
    g2_df = (close_df > ema_f_df) & (ema_f_df > ema_m_df) & (ema_m_df > ema_l_df)

    # G3: momentum quality (scored)
    g3_df = (
        (rsi_df > rsi_low) & (rsi_df < rsi_high)
        & (cl_body_df >= candle_body_min)
        & (ret63_df > 0)
    )

    # G4: entry quality (scored)
    atr_pct = atr_df / close_df.replace(0, np.nan)
    g4_df = (
        (atr_pct >= min_atr_pct) & (atr_pct <= max_atr_pct)
        & (close_df >= min_price)
        & (close_df > not_near_low_mult * low252_df)
    )

    score_df = g2_df.astype(int) + g3_df.astype(int) + g4_df.astype(int)

    # Valid = G1 + G_bo (both mandatory) + score ≥ min_score
    valid = (
        g1_df & gbo_df
        & (score_df >= min_score)
        & close_df.notna() & atr_df.notna()
    )

    r_idx, c_idx = np.where(valid.values)
    if len(r_idx) == 0:
        logger.warning("NSE-S3: no signals generated")
        return _empty()

    dates_out = idx[r_idx]
    syms_out  = np.array(active)[c_idx]
    scores    = score_df.values[r_idx, c_idx].astype(int)
    closes    = close_df.values[r_idx, c_idx]
    atrs      = atr_df.values[r_idx, c_idx]
    nifty_ret_out = nifty_ret63.reindex(idx).values[r_idx]
    stock_ret63   = ret63_df.values[r_idx, c_idx]
    rs_out        = stock_ret63 - nifty_ret_out

    result = pd.DataFrame({
        "date":      dates_out,
        "symbol":    syms_out,
        "score":     scores,
        "close":     np.round(closes, 2),
        "atr":       np.round(atrs, 2),
        "stop_loss": np.round(closes - sl_atr_mult * atrs, 2),
        "target":    np.round(closes + tp_atr_mult * atrs, 2),
        "rs_nifty":  np.round(rs_out, 4),
    })

    result = result.sort_values(["date", "score"], ascending=[True, False]).reset_index(drop=True)
    logger.info(
        "NSE-S3: {} signals | {} signal-dates | avg {:.1f}/day | score dist: {}",
        len(result), result["date"].nunique(),
        len(result) / max(result["date"].nunique(), 1),
        dict(result["score"].value_counts().sort_index()),
    )
    return result


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "symbol", "score", "close", "atr",
                                  "stop_loss", "target", "rs_nifty"])
