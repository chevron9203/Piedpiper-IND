import numpy as np
import pandas as pd
from loguru import logger


def adx(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    high = df["high"]
    low = df["low"]
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = df["close"].shift(1)

    # Directional movement
    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    # True range
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    alpha = 1.0 / window

    # Wilder's smoothing via EWM
    atr_wilder = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=alpha, adjust=False).mean()

    plus_di = 100 * plus_dm_smooth / (atr_wilder + 1e-9)
    minus_di = 100 * minus_dm_smooth / (atr_wilder + 1e-9)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    adx_val = dx.ewm(alpha=alpha, adjust=False).mean()

    return pd.DataFrame(
        {"adx": adx_val, "plus_di": plus_di, "minus_di": minus_di},
        index=df.index,
    )


def trend_regime(df: pd.DataFrame, adx_window: int = 14, adx_threshold: float = 25) -> pd.Series:
    adx_df = adx(df, window=adx_window)
    return (adx_df["adx"] > adx_threshold).astype(float).rename("trend_regime")
