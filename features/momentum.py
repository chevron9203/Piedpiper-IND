import numpy as np
import pandas as pd
from loguru import logger


def rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    # Wilder's smoothing: equivalent to EWM with alpha = 1/window, adjust=False
    alpha = 1.0 / window
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return (100 - 100 / (1 + rs)).rename("rsi")


def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "macd_signal": signal_line, "macd_hist": hist},
        index=df.index,
    )


def bollinger_pct_b(df: pd.DataFrame, window: int = 20, num_std: float = 2) -> pd.Series:
    mp = max(1, window // 2)
    sma = df["close"].rolling(window, min_periods=mp).mean()
    std = df["close"].rolling(window, min_periods=mp).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return ((df["close"] - lower) / (upper - lower + 1e-9)).rename("bb_pct_b")


def distance_from_ma(df: pd.DataFrame, windows: list[int] = [20, 50, 200]) -> pd.DataFrame:
    results = {}
    for n in windows:
        mp = max(1, n // 2)
        sma = df["close"].rolling(n, min_periods=mp).mean()
        results[f"dist_ma_{n}d"] = (df["close"] - sma) / (sma + 1e-9)
    return pd.DataFrame(results, index=df.index)
