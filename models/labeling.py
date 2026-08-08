"""
Triple-barrier labeling for NSE equity signals.

Label logic (long-only, net-of-cost barriers):
  - Entry at next day's open (open[t+1]) — strictly no lookahead.
  - TP barrier: entry_price * (1 + tp_pct) net of full round-trip cost.
  - SL barrier: entry_price * (1 - sl_pct) net of full round-trip cost.
  - Forward-scan up to max_hold trading days.
  - First-touch wins: 1 = TP, -1 = SL, 0 = time barrier.
  - Long-only V1 mapping: label 1 → "long", {-1, 0} → "flat" (binary target = 1 / 0).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import (
    BARRIER_MAX_HOLDING_DAYS,
    BARRIER_STOP_LOSS_PCT,
    BARRIER_TAKE_PROFIT_PCT,
)
from config.cost_model import round_trip_cost, net_return


# ---------------------------------------------------------------------------
# Reference quantity for cost estimation in barrier computation.
# Cost rates for a ₹1-lakh notional @ typical midcap price are used.
# Using 100 shares is consistent with cost_model.cost_adjusted_barrier().
# ---------------------------------------------------------------------------
_REF_QUANTITY = 100


def _net_of_cost_barriers(
    entry_price: float,
    tp_pct: float,
    sl_pct: float,
    symbol_tier: str,
    quantity: int = _REF_QUANTITY,
) -> tuple[float, float]:
    """
    Compute the net-of-cost take-profit and stop-loss price levels.

    The TP level is the price at which the round-trip net return equals tp_pct.
    The SL level is the price at which the round-trip net return equals -sl_pct.

    We back-solve from the net_return formula:
        net_return = (gross - costs) / entry_value

    Because costs themselves depend on exit_price, we iterate once (costs are
    a small fraction, so one iteration converges to <1 bp error).
    """
    entry_value = quantity * entry_price

    # --- TP level (net) ---
    # Gross TP exit price (without costs):
    gross_tp = entry_price * (1.0 + tp_pct)
    # Round-trip cost at that exit:
    cost_tp = round_trip_cost(quantity, entry_price, gross_tp, symbol_tier)["total"]
    # Required gross pnl to achieve tp_pct net: tp_pct * entry_value + costs
    required_gross_pnl_tp = tp_pct * entry_value + cost_tp
    tp_level = entry_price + required_gross_pnl_tp / quantity

    # --- SL level (net) ---
    gross_sl = entry_price * (1.0 - sl_pct)
    cost_sl = round_trip_cost(quantity, entry_price, gross_sl, symbol_tier)["total"]
    # Required gross loss (absolute) such that net loss = sl_pct * entry_value
    # net_pnl = gross_pnl - costs  =>  -sl_pct * entry_value = gross_pnl - cost_sl
    required_gross_pnl_sl = -(sl_pct * entry_value) + cost_sl  # negative
    sl_level = entry_price + required_gross_pnl_sl / quantity

    return tp_level, sl_level


def _net_of_cost_barriers_short(
    entry_price: float,
    tp_pct: float,
    sl_pct: float,
    symbol_tier: str,
    quantity: int = _REF_QUANTITY,
) -> tuple[float, float]:
    """
    Net-of-cost price barriers for a short position.

    Short TP: price must fall far enough that (entry - exit)*qty - costs = tp_pct * entry_value.
    Short SL: price rises until (exit - entry)*qty + costs = sl_pct * entry_value.
    Costs make TP level lower (must fall further) and SL level slightly tighter than gross.
    """
    entry_value = quantity * entry_price

    # Short TP level: price falls to here → net profit = tp_pct
    gross_tp_exit = entry_price * (1.0 - tp_pct)
    cost_tp = round_trip_cost(quantity, entry_price, gross_tp_exit, symbol_tier)["total"]
    required_gross_profit = tp_pct * entry_value + cost_tp
    tp_level = entry_price - required_gross_profit / quantity

    # Short SL level: price rises to here → net loss = sl_pct
    gross_sl_exit = entry_price * (1.0 + sl_pct)
    cost_sl = round_trip_cost(quantity, entry_price, gross_sl_exit, symbol_tier)["total"]
    required_gross_loss = sl_pct * entry_value - cost_sl
    sl_level = entry_price + max(required_gross_loss, 0) / quantity

    return tp_level, sl_level


def compute_labels(
    df: pd.DataFrame,
    symbol: str,
    symbol_tier: str,
    tp_pct: float = BARRIER_TAKE_PROFIT_PCT,
    sl_pct: float = BARRIER_STOP_LOSS_PCT,
    max_hold: int = BARRIER_MAX_HOLDING_DAYS,
) -> pd.DataFrame:
    """
    Compute triple-barrier labels for a single symbol.

    Parameters
    ----------
    df : pd.DataFrame
        Adjusted OHLCV with DatetimeIndex. Required columns: open, high, low, close.
    symbol : str
        Ticker symbol (used for logging).
    symbol_tier : str
        "nifty100" or "midcap100" — passed to cost model.
    tp_pct : float
        Gross take-profit fraction (net-of-cost target will be slightly higher entry).
    sl_pct : float
        Gross stop-loss fraction (net-of-cost barrier will be slightly tighter).
    max_hold : int
        Maximum holding period in trading days (time barrier).

    Returns
    -------
    pd.DataFrame
        Indexed by the signal date t (the date we decide to enter).
        Columns:
          entry_price   – open price at t+1 (actual fill price)
          tp_level      – net-of-cost take-profit price
          sl_level      – net-of-cost stop-loss price
          label         – raw triple-barrier: 1 (TP), -1 (SL), 0 (time)
          binary_label  – 1 if label==1 else 0  (long-only V1 target)
          holding_days  – number of days held
          net_return    – realised net return (signed, net of all costs)
    """
    df = df.copy().sort_index()
    required_cols = {"open", "high", "low", "close"}
    missing = required_cols - set(df.columns.str.lower())
    if missing:
        raise ValueError(f"[{symbol}] Missing columns: {missing}")

    # Normalise column names
    df.columns = df.columns.str.lower()

    n = len(df)
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values

    records = []

    # Signal date t runs from index 0 … n-2 (we need open[t+1] as entry)
    for i in range(n - 1):
        entry_idx = i + 1          # index of entry bar (t+1)
        entry_price = opens[entry_idx]

        if entry_price <= 0 or np.isnan(entry_price):
            continue

        tp_level, sl_level = _net_of_cost_barriers(
            entry_price, tp_pct, sl_pct, symbol_tier
        )

        label = 0           # default: time barrier
        holding_days = 0
        exit_price = df["close"].values[min(entry_idx + max_hold - 1, n - 1)]

        # Forward scan: bars entry_idx+1 … entry_idx+max_hold (inclusive)
        for k in range(1, max_hold + 1):
            bar = entry_idx + k
            if bar >= n:
                # Ran out of data — use last available close as exit
                exit_price = df["close"].values[n - 1]
                holding_days = n - 1 - entry_idx
                break

            high_k = highs[bar]
            low_k = lows[bar]

            tp_hit = high_k >= tp_level
            sl_hit = low_k <= sl_level

            if tp_hit and sl_hit:
                # Both hit in same bar — assume worst case for long: SL first
                # (conservative: favours label=-1 when ambiguous)
                label = -1
                exit_price = sl_level
                holding_days = k
                break
            elif tp_hit:
                label = 1
                exit_price = tp_level
                holding_days = k
                break
            elif sl_hit:
                label = -1
                exit_price = sl_level
                holding_days = k
                break
        else:
            # Time barrier exhausted — use close of last bar
            last_bar = min(entry_idx + max_hold, n - 1)
            exit_price = df["close"].values[last_bar]
            holding_days = last_bar - entry_idx

        realised_net = net_return(
            _REF_QUANTITY, entry_price, exit_price, symbol_tier
        )

        records.append(
            {
                "dt": df.index[i],           # signal date (t)
                "entry_price": entry_price,
                "tp_level": tp_level,
                "sl_level": sl_level,
                "label": label,
                "binary_label": 1 if label == 1 else 0,
                "holding_days": holding_days,
                "net_return": realised_net,
            }
        )

    if not records:
        logger.warning(f"[{symbol}] No label records generated — data too short?")
        return pd.DataFrame(
            columns=[
                "dt", "entry_price", "tp_level", "sl_level",
                "label", "binary_label", "holding_days", "net_return",
            ]
        )

    result = pd.DataFrame(records).set_index("dt")
    result.index = pd.DatetimeIndex(result.index)

    tp_rate = (result["label"] == 1).mean()
    sl_rate = (result["label"] == -1).mean()
    logger.info(
        f"[{symbol}] Labeled {len(result)} samples | "
        f"TP={tp_rate:.1%}  SL={sl_rate:.1%}  "
        f"Time={(1-tp_rate-sl_rate):.1%}"
    )

    return result


def compute_short_labels(
    df: pd.DataFrame,
    symbol: str,
    symbol_tier: str,
    tp_pct: float = BARRIER_TAKE_PROFIT_PCT,
    sl_pct: float = BARRIER_STOP_LOSS_PCT,
    max_hold: int = BARRIER_MAX_HOLDING_DAYS,
) -> pd.DataFrame:
    """
    Compute short triple-barrier labels for a single symbol.

    Mirror of compute_labels() with inverted barriers:
      - TP: price falls to tp_level (low <= tp_level) → label = 1 (short wins)
      - SL: price rises to sl_level (high >= sl_level) → label = -1 (short loses)
      - Both same bar: SL wins (conservative worst-case for short).
      - binary_label = 1 if label == 1 else 0.
    """
    df = df.copy().sort_index()
    required_cols = {"open", "high", "low", "close"}
    missing = required_cols - set(df.columns.str.lower())
    if missing:
        raise ValueError(f"[{symbol}] Missing columns: {missing}")

    df.columns = df.columns.str.lower()

    n = len(df)
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values

    records = []

    for i in range(n - 1):
        entry_idx = i + 1
        entry_price = opens[entry_idx]

        if entry_price <= 0 or np.isnan(entry_price):
            continue

        tp_level, sl_level = _net_of_cost_barriers_short(
            entry_price, tp_pct, sl_pct, symbol_tier
        )

        label = 0
        holding_days = 0
        exit_price = df["close"].values[min(entry_idx + max_hold - 1, n - 1)]

        for k in range(1, max_hold + 1):
            bar = entry_idx + k
            if bar >= n:
                exit_price = df["close"].values[n - 1]
                holding_days = n - 1 - entry_idx
                break

            high_k = highs[bar]
            low_k = lows[bar]

            tp_hit = low_k <= tp_level    # short TP: low reaches down to target
            sl_hit = high_k >= sl_level   # short SL: high reaches up to stop

            if tp_hit and sl_hit:
                # Both hit same bar — conservative for short: SL wins
                label = -1
                exit_price = sl_level
                holding_days = k
                break
            elif tp_hit:
                label = 1
                exit_price = tp_level
                holding_days = k
                break
            elif sl_hit:
                label = -1
                exit_price = sl_level
                holding_days = k
                break
        else:
            last_bar = min(entry_idx + max_hold, n - 1)
            exit_price = df["close"].values[last_bar]
            holding_days = last_bar - entry_idx

        # Short P&L: (entry - exit) * qty - costs
        gross_pnl = _REF_QUANTITY * (entry_price - exit_price)
        costs = round_trip_cost(_REF_QUANTITY, entry_price, exit_price, symbol_tier)["total"]
        realised_net = (gross_pnl - costs) / (_REF_QUANTITY * entry_price)

        records.append(
            {
                "dt": df.index[i],
                "entry_price": entry_price,
                "tp_level": tp_level,
                "sl_level": sl_level,
                "label": label,
                "binary_label": 1 if label == 1 else 0,
                "holding_days": holding_days,
                "net_return": realised_net,
            }
        )

    if not records:
        logger.warning(f"[{symbol}] No short label records generated — data too short?")
        return pd.DataFrame(
            columns=["dt", "entry_price", "tp_level", "sl_level",
                     "label", "binary_label", "holding_days", "net_return"]
        )

    result = pd.DataFrame(records).set_index("dt")
    result.index = pd.DatetimeIndex(result.index)

    tp_rate = (result["label"] == 1).mean()
    sl_rate = (result["label"] == -1).mean()
    logger.info(
        f"[{symbol}] Short labels: {len(result)} | "
        f"TP={tp_rate:.1%}  SL={sl_rate:.1%}  "
        f"Time={(1-tp_rate-sl_rate):.1%}"
    )

    return result


# ---------------------------------------------------------------------------
# Sample weights
# ---------------------------------------------------------------------------

def get_sample_weights(labels: pd.Series, returns: pd.Series) -> pd.Series:
    """
    Compute sample weights proportional to absolute net return magnitude.

    Samples with larger absolute returns (hits that matter more financially)
    receive higher weight during training. Follows the standard MLB weighting
    scheme from Advances in Financial Machine Learning (de Prado, Ch. 4).

    Parameters
    ----------
    labels : pd.Series
        Triple-barrier labels (1, -1, or 0) or binary labels.
    returns : pd.Series
        Net returns aligned with labels (same index).

    Returns
    -------
    pd.Series
        Non-negative weights summing to 1.0, indexed like labels.
    """
    aligned = returns.reindex(labels.index).fillna(0.0)
    weights = aligned.abs()

    # Floor at a small epsilon so every sample has non-zero weight
    weights = weights.clip(lower=1e-6)

    total = weights.sum()
    if total == 0:
        logger.warning("All returns are zero — using uniform weights.")
        weights = pd.Series(1.0 / len(labels), index=labels.index)
    else:
        weights = weights / total

    return weights
