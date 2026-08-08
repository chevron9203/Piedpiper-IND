import numpy as np
import pandas as pd
from loguru import logger


def multi_horizon_returns(df: pd.DataFrame, horizons: list[int] = [1, 5, 10, 20, 60]) -> pd.DataFrame:
    results = {}
    for n in horizons:
        results[f"ret_{n}d"] = df["close"].pct_change(n)
    return pd.DataFrame(results, index=df.index)


def gap_open(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return ((df["open"] - prev_close) / prev_close).rename("gap_open")


def body_ratio(df: pd.DataFrame) -> pd.Series:
    return (
        (df["close"] - df["open"]).abs() / (df["high"] - df["low"] + 1e-9)
    ).rename("body_ratio")


def upper_wick_ratio(df: pd.DataFrame) -> pd.Series:
    upper_body = df[["open", "close"]].max(axis=1)
    return ((df["high"] - upper_body) / (df["high"] - df["low"] + 1e-9)).rename("upper_wick_ratio")


def lower_wick_ratio(df: pd.DataFrame) -> pd.Series:
    lower_body = df[["open", "close"]].min(axis=1)
    return ((lower_body - df["low"]) / (df["high"] - df["low"] + 1e-9)).rename("lower_wick_ratio")


def rolling_high_low_position(df: pd.DataFrame, windows: list[int] = [10, 20, 60]) -> pd.DataFrame:
    results = {}
    for n in windows:
        mp = max(1, n // 2)
        roll_low = df["low"].rolling(n, min_periods=mp).min()
        roll_high = df["high"].rolling(n, min_periods=mp).max()
        results[f"hl_pos_{n}d"] = (df["close"] - roll_low) / (roll_high - roll_low + 1e-9)
    return pd.DataFrame(results, index=df.index)


def high52w_ratio(df: pd.DataFrame) -> pd.Series:
    """Ratio of today's close to the 52-week (252-day) rolling high.
    Near 1.0 → near highs (momentum signal). Near 0 → deep drawdown.
    Proven factor in cross-sectional momentum literature."""
    high_252 = df["high"].rolling(252, min_periods=126).max()
    return (df["close"] / (high_252 + 1e-9)).rename("high52w_ratio")
