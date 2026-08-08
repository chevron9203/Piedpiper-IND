"""
NSE-S2: HIGH-based breakout + VWAP + multi-timeframe trend.

Fixes vs failed v1:
  Problem 1 — breakout must be MANDATORY (hard gate), not scored.
    With min_score=3/4, 99.4% of v1 signals had NO breakout (G4 failing
    while G2+G3+G5 passing). Fixed: HIGH breakout is now a hard veto
    alongside G1. Only stocks with a fresh HIGH breakout can signal.

  Problem 2 — box floor stop was 6.9% avg, target 17.24%.
    Impossible to reach 17% gain in 25 trading days. Fixed: use ATR-based
    stops (2.5×ATR, 5×ATR) — same as proven S1 approach.

  Problem 3 — VWAP bundled with breakout gave only 129 signals over 6yr.
    Fixed: VWAP is now a separate SCORED gate (can miss it without losing
    the entire signal), not hard-gated with breakout.

Architecture:
  G1  Market regime (HARD VETO):
      VIX < threshold AND Nifty > 200d EMA AND Nifty > 50d EMA

  G_bo  HIGH-based breakout (HARD VETO — must fire, not scored):
      Today's close > 63-day max of HIGH (shift 1, no lookahead)
      AND fresh: yesterday was NOT already above that level
      AND volume ≥ vol_mult × 20d avg

  Scored gates (G2–G5, each 0 or 1):
  G2  Monthly trend:
      Close > 126d EMA (6-month alignment)

  G3  Weekly trend:
      Weekly close > 20w EMA AND weekly ADX(14) > 20

  G4  VWAP confirmation:
      Close > 20d rolling VWAP
      AND VWAP > VWAP 5 days ago (upward slope)

  G5  Entry quality:
      RSI(14) ∈ [55, 80]
      AND (close−low)/(high−low) ≥ 0.50
      AND ATR/close ∈ [1%, 5%]
      AND close ≥ ₹100

Score = G2+G3+G4+G5. Enter if score ≥ min_score (default 3 of 4).
All signals get SCORE5_MULT (1.5×) in backtest when score ≥ score_boost_min.

Stop/target: anchored to ATR in the signal, applied to actual fill price.
  stop_loss = close − sl_atr_mult × ATR(14)
  target    = close + tp_atr_mult × ATR(14)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from features.volatility import atr as _calc_atr
from features.momentum import rsi as _calc_rsi
from features.regime import adx as _calc_adx


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
    combined = s.reindex(idx.union(s.index)).ffill()
    return combined.reindex(idx)


def generate_signals(
    ohlcv_dict: dict[str, pd.DataFrame],
    nifty_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    universe_symbols: list[str] | None = None,
    # G1 — market regime hard veto
    vix_threshold: float = 20.0,
    ema_long: int = 200,
    nifty_ema_mid: int = 50,        # 0 = disabled
    # G_bo — HIGH breakout hard veto
    breakout_window: int = 63,
    vol_window: int = 20,
    vol_mult: float = 2.0,          # volume surge on breakout day
    # G2 — monthly trend (scored)
    monthly_ema_span: int = 126,
    # G3 — weekly trend (scored)
    weekly_ema_span: int = 20,
    weekly_adx_min: float = 20.0,
    # G4 — VWAP (scored)
    vwap_window: int = 20,
    vwap_lookback: int = 5,         # VWAP slope lookback days
    # G5 — entry quality (scored)
    rsi_window: int = 14,
    rsi_low: float = 55.0,
    rsi_high: float = 80.0,
    candle_body_min: float = 0.50,
    min_atr_pct: float = 0.01,
    max_atr_pct: float = 0.05,
    min_price: float = 100.0,
    # stop / target reference (used in forward test display; backtest re-anchors to fill)
    sl_atr_mult: float = 2.5,
    tp_atr_mult: float = 5.0,
    # scoring
    min_score: int = 3,             # min of 4 scored gates
) -> pd.DataFrame:
    """
    Compute NSE-S2 signals.

    Returns DataFrame with columns:
        date, symbol, score, close, atr, stop_loss, target, rs_nifty
    (no stop_dist/tp_dist — backtest uses ATR-based stops like S1)
    """
    if universe_symbols is None:
        universe_symbols = list(ohlcv_dict.keys())
    symbols = [s for s in universe_symbols if s in ohlcv_dict]
    if not symbols:
        logger.warning("NSE-S2: no symbols in ohlcv_dict")
        return _empty()

    logger.info("NSE-S2: computing indicators for {} symbols ...", len(symbols))

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
        if len(df) < 200:
            continue
        idx = df.index

        atr_s   = _calc_atr(df, window=14)
        rsi_s   = _calc_rsi(df, window=rsi_window)
        vol_avg = df["volume"].rolling(vol_window, min_periods=vol_window // 2).mean()

        # ── G_bo: HIGH-based breakout ─────────────────────────────────────────
        min_bp  = max(breakout_window // 2, 20)
        # Box top = rolling max of HIGH over past N days (shift 1 = no lookahead)
        high_bo = df["high"].shift(1).rolling(breakout_window, min_periods=min_bp).max()
        # Fresh: yesterday's close was NOT above yesterday's (N-1)-day HIGH max
        prev_high_bo = df["high"].shift(2).rolling(breakout_window - 1, min_periods=min_bp - 1).max()
        already_broke = df["close"].shift(1) > prev_high_bo

        # ── G4: rolling VWAP ──────────────────────────────────────────────────
        typical  = (df["high"] + df["low"] + df["close"]) / 3
        vwap_num = (typical * df["volume"]).rolling(vwap_window, min_periods=vwap_window // 2).sum()
        vwap_den = df["volume"].rolling(vwap_window, min_periods=vwap_window // 2).sum()
        vwap     = vwap_num / vwap_den.replace(0, np.nan)

        # ── G2: monthly trend ─────────────────────────────────────────────────
        ema_monthly = _ema(df["close"], monthly_ema_span)

        # ── G5: candle body ───────────────────────────────────────────────────
        hl_range = df["high"] - df["low"]
        cl_body  = (df["close"] - df["low"]) / hl_range.replace(0, np.nan)

        # ── G3: weekly trend ─────────────────────────────────────────────────
        weekly = _resample_weekly(df)
        w_ema  = _ffill_to_index(_ema(weekly["close"], weekly_ema_span), idx)
        w_adx  = _ffill_to_index(_calc_adx(weekly, window=14)["adx"], idx)

        # ── 63d return for RS ────────────────────────────────────────────────
        ret63 = df["close"].pct_change(63)

        sym_data[sym] = dict(
            close=df["close"], high=df["high"],
            volume=df["volume"], atr=atr_s, rsi=rsi_s, vol_avg=vol_avg,
            high_bo=high_bo, already_broke=already_broke,
            vwap=vwap, ema_monthly=ema_monthly, cl_body=cl_body,
            w_ema=w_ema, w_adx=w_adx, ret63=ret63,
        )

    if not sym_data:
        return _empty()

    active = list(sym_data.keys())

    def _wide(key: str) -> pd.DataFrame:
        return pd.DataFrame({s: sym_data[s][key] for s in active})

    close_df    = _wide("close")
    high_df     = _wide("high")
    volume_df   = _wide("volume")
    atr_df      = _wide("atr")
    rsi_df      = _wide("rsi")
    vol_avg_df  = _wide("vol_avg")
    high_bo_df  = _wide("high_bo")
    ab_df       = _wide("already_broke")
    vwap_df     = _wide("vwap")
    ema_m_df    = _wide("ema_monthly")
    cl_body_df  = _wide("cl_body")
    w_ema_df    = _wide("w_ema")
    w_adx_df    = _wide("w_adx")
    ret63_df    = _wide("ret63")

    idx = close_df.index

    # ── G1: hard regime veto ──────────────────────────────────────────────────
    nifty_close  = nifty["close"].reindex(idx, method="ffill")
    nifty_ema200 = _ema(nifty["close"], ema_long).reindex(idx, method="ffill")
    vix_close    = vix["close"].reindex(idx, method="ffill")
    g1_series    = (vix_close < vix_threshold) & (nifty_close > nifty_ema200)
    if nifty_ema_mid > 0:
        nifty_ema_m = _ema(nifty["close"], nifty_ema_mid).reindex(idx, method="ffill")
        g1_series   = g1_series & (nifty_close > nifty_ema_m)

    nifty_ret63 = nifty["close"].pct_change(63).reindex(idx, method="ffill")

    g1_df = pd.DataFrame(
        np.tile(g1_series.values.reshape(-1, 1), (1, len(active))),
        index=idx, columns=active, dtype=bool,
    )

    # ── G_bo: HIGH-based breakout (hard veto, not scored) ────────────────────
    is_breakout = close_df > high_bo_df
    is_fresh    = is_breakout & (~ab_df.fillna(False).astype(bool))
    vol_surge   = volume_df >= vol_mult * vol_avg_df
    gbo_df      = is_fresh & vol_surge    # MUST be true for any signal

    # ── G2: monthly trend (scored) ────────────────────────────────────────────
    g2_df = close_df > ema_m_df

    # ── G3: weekly trend (scored) ─────────────────────────────────────────────
    g3_df = (close_df > w_ema_df) & (w_adx_df > weekly_adx_min)

    # ── G4: VWAP (scored) — close above VWAP AND VWAP trending up ────────────
    above_vwap  = close_df > vwap_df
    vwap_rising = vwap_df > vwap_df.shift(vwap_lookback)
    g4_df       = above_vwap & vwap_rising

    # ── G5: RSI + entry quality (scored) ─────────────────────────────────────
    atr_pct = atr_df / close_df.replace(0, np.nan)
    g5_df = (
        (rsi_df > rsi_low) & (rsi_df < rsi_high)
        & (cl_body_df >= candle_body_min)
        & (atr_pct >= min_atr_pct) & (atr_pct <= max_atr_pct)
        & (close_df >= min_price)
    )

    # ── Score: G2+G3+G4+G5 (G1 and G_bo are hard vetoes) ─────────────────────
    score_df = (
        g2_df.astype(int) + g3_df.astype(int)
        + g4_df.astype(int) + g5_df.astype(int)
    )

    # Valid = G1 passes + G_bo (HIGH breakout) passes + score ≥ min_score
    valid = (
        g1_df & gbo_df
        & (score_df >= min_score)
        & close_df.notna() & atr_df.notna()
    )

    r_idx, c_idx = np.where(valid.values)
    if len(r_idx) == 0:
        logger.warning("NSE-S2: no signals generated")
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
        "NSE-S2: {} signals | {} signal-dates | avg {:.1f}/day | score dist: {}",
        len(result), result["date"].nunique(),
        len(result) / max(result["date"].nunique(), 1),
        dict(result["score"].value_counts().sort_index()),
    )
    return result


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "symbol", "score", "close", "atr",
                                  "stop_loss", "target", "rs_nifty"])
