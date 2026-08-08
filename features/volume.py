import numpy as np
import pandas as pd
from loguru import logger


def volume_ratio(df: pd.DataFrame, windows: list[int] = [5, 20]) -> pd.DataFrame:
    results = {}
    for n in windows:
        mp = max(1, n // 2)
        roll_mean = df["volume"].rolling(n, min_periods=mp).mean()
        results[f"vol_ratio_{n}d"] = df["volume"] / roll_mean
    return pd.DataFrame(results, index=df.index)


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff())
    # on the first bar direction is NaN — treat as 0 so OBV starts at 0
    direction = direction.fillna(0)
    return (direction * df["volume"]).cumsum().rename("obv")


def price_volume_divergence(df: pd.DataFrame, window: int = 10) -> pd.Series:
    price_chg = df["close"].pct_change()
    vol_chg = df["volume"].pct_change()
    mp = max(1, window // 2)
    return price_chg.rolling(window, min_periods=mp).corr(vol_chg).rename("price_vol_divergence")


def vwap_deviation(df: pd.DataFrame, window: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mp = max(1, window // 2)
    rolling_vwap = (
        (tp * df["volume"]).rolling(window, min_periods=mp).sum()
        / df["volume"].rolling(window, min_periods=mp).sum()
    )
    return ((df["close"] - rolling_vwap) / rolling_vwap).rename("vwap_deviation")
