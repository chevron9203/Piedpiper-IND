import numpy as np
import pandas as pd
from loguru import logger


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window, min_periods=max(1, window // 2)).mean().rename("atr")


def realized_vol(df: pd.DataFrame, windows: list[int] = [5, 10, 20, 60]) -> pd.DataFrame:
    log_ret = np.log(df["close"] / df["close"].shift(1))
    results = {}
    for n in windows:
        mp = max(1, n // 2)
        results[f"rvol_{n}d"] = log_ret.rolling(n, min_periods=mp).std() * np.sqrt(252)
    return pd.DataFrame(results, index=df.index)


def vol_of_vol(df: pd.DataFrame, outer_window: int = 20, inner_window: int = 5) -> pd.Series:
    log_ret = np.log(df["close"] / df["close"].shift(1))
    mp_inner = max(1, inner_window // 2)
    inner_vol = log_ret.rolling(inner_window, min_periods=mp_inner).std() * np.sqrt(252)
    mp_outer = max(1, outer_window // 2)
    return inner_vol.rolling(outer_window, min_periods=mp_outer).std().rename("vol_of_vol")
