"""
Baseline strategies for benchmarking the NSE trading signal system.

Two baselines:
  1. buy_and_hold_equity  — invest full starting capital in the index on day 1.
  2. ma_crossover_signals — simple moving-average crossover on each stock.

These form the minimum hurdle: the ML system must beat both on risk-adjusted
returns to justify its complexity cost.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import STARTING_VIRTUAL_CAPITAL


def buy_and_hold_equity(
    ohlcv: pd.DataFrame,
    start_capital: float = STARTING_VIRTUAL_CAPITAL,
) -> pd.Series:
    """
    Simulate buying the Nifty index (or any OHLCV proxy) at the first open
    and holding to the end.

    Parameters
    ----------
    ohlcv         : DataFrame with columns [open, high, low, close, volume]
                    and a DatetimeIndex. Typically the Nifty 50 index OHLCV.
    start_capital : starting cash (default ₹1,00,000)

    Returns
    -------
    pd.Series : daily equity curve (DatetimeIndex, float values in ₹)
    """
    if ohlcv.empty:
        logger.warning("buy_and_hold_equity: empty OHLCV passed, returning empty series")
        return pd.Series(dtype=float)

    df = ohlcv.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.columns = [c.lower() for c in df.columns]

    if "open" not in df.columns:
        raise ValueError("OHLCV must contain an 'open' column")

    entry_price = df["open"].iloc[0]
    if entry_price <= 0:
        raise ValueError(f"Invalid entry price {entry_price} on {df.index[0].date()}")

    # fractional units allowed (index tracking, not actual shares)
    units = start_capital / entry_price

    # equity each day = units * close price
    equity = units * df["close"]
    equity.name = "buy_and_hold"

    logger.info(
        "buy_and_hold_equity: entry={:.2f} on {} | final={:.2f} on {} | return={:.1%}",
        entry_price,
        df.index[0].date(),
        equity.iloc[-1],
        df.index[-1].date(),
        (equity.iloc[-1] / start_capital) - 1,
    )
    return equity


def ma_crossover_signals(
    ohlcv_dict: dict[str, pd.DataFrame],
    fast: int = 20,
    slow: int = 50,
) -> pd.DataFrame:
    """
    Simple moving-average crossover signal for each stock in the universe.

    Logic
    -----
    - Signal = 1 (long) when fast SMA > slow SMA.
    - Signal = 0 (flat) otherwise.
    - First ``slow`` bars are NaN (insufficient history).
    - Signals are computed on close prices to represent end-of-day decisions.
      The BacktestEngine will shift these by 1 bar so execution happens at the
      *next* bar's open (no lookahead bias).

    Parameters
    ----------
    ohlcv_dict : {symbol: OHLCV DataFrame} with DatetimeIndex
    fast       : fast MA window in trading days (default 20)
    slow       : slow MA window in trading days (default 50)

    Returns
    -------
    pd.DataFrame : symbol × date signal matrix, values 0 or 1 (float to allow NaN).
                   Index = DatetimeIndex, columns = symbol strings.
    """
    if not ohlcv_dict:
        logger.warning("ma_crossover_signals: empty ohlcv_dict")
        return pd.DataFrame()

    signal_frames: dict[str, pd.Series] = {}

    for symbol, df in ohlcv_dict.items():
        try:
            _df = df.copy()
            _df.index = pd.to_datetime(_df.index)
            _df = _df.sort_index()
            _df.columns = [c.lower() for c in _df.columns]

            if "close" not in _df.columns:
                logger.warning("ma_crossover_signals: no 'close' for {}, skipping", symbol)
                continue

            close = _df["close"]
            fast_ma = close.rolling(fast, min_periods=fast).mean()
            slow_ma = close.rolling(slow, min_periods=slow).mean()

            signal = (fast_ma > slow_ma).astype(float)
            # mask first `slow` rows as NaN (no valid slow MA yet)
            signal.iloc[: slow - 1] = np.nan
            signal.name = symbol
            signal_frames[symbol] = signal

        except Exception as exc:
            logger.error("ma_crossover_signals: error for {}: {}", symbol, exc)

    if not signal_frames:
        return pd.DataFrame()

    result = pd.DataFrame(signal_frames)
    result.index = pd.to_datetime(result.index)
    result = result.sort_index()

    logger.info(
        "ma_crossover_signals: {} symbols, {} dates, fast={}, slow={}",
        len(result.columns),
        len(result),
        fast,
        slow,
    )
    return result
