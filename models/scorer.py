"""
Cross-sectional scorer and position-size calculator.

rank_universe():
  - Filters universe to symbols with predicted_prob >= confidence_threshold.
  - Ranks survivors by score descending.
  - Returns at most max_signals rows.

compute_position_size():
  - Tiered sizing: top-third 1.5x, middle-third 1.0x, bottom-third 0.75x.
  - Hard cap: position size <= (max_deployed_pct * total_capital) / max_positions.
  - Does not exceed remaining capital headroom implied by n_open_positions.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from config.settings import (
    CONFIDENCE_THRESHOLD,
    MAX_CAPITAL_DEPLOYED_PCT,
    MAX_CONCURRENT_POSITIONS,
)


def rank_universe(
    predictions: pd.Series,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    max_signals: int = MAX_CONCURRENT_POSITIONS,
) -> pd.DataFrame:
    """
    Cross-sectional ranking of signal probabilities.

    Parameters
    ----------
    predictions : pd.Series
        Mapping {symbol: predicted_prob} for a single trading day.
        Index must be symbol names.
    confidence_threshold : float
        Minimum probability for a signal to qualify. Signals below this are
        silently dropped (zero signals is a valid and correct output).
    max_signals : int
        Maximum number of qualifying signals to return (top-N by score).

    Returns
    -------
    pd.DataFrame
        Columns: symbol, score, rank.
        Sorted by score descending. May be empty.
    """
    if predictions.empty:
        logger.debug("rank_universe: received empty predictions Series.")
        return pd.DataFrame(columns=["symbol", "score", "rank"])

    # Filter by confidence threshold
    qualified = predictions[predictions >= confidence_threshold]

    if qualified.empty:
        logger.debug(
            f"rank_universe: no symbols above threshold={confidence_threshold:.2f} "
            f"(universe size={len(predictions)})."
        )
        return pd.DataFrame(columns=["symbol", "score", "rank"])

    # Sort descending, take top-N
    ranked = qualified.sort_values(ascending=False).head(max_signals)

    result = pd.DataFrame(
        {
            "symbol": ranked.index,
            "score": ranked.values,
        }
    )
    result["rank"] = range(1, len(result) + 1)
    result = result.reset_index(drop=True)

    logger.debug(
        f"rank_universe: {len(result)} signal(s) | "
        f"scores {result['score'].min():.3f}–{result['score'].max():.3f}"
    )
    return result


def compute_position_size(
    score: float,
    total_capital: float,
    n_open_positions: int,
    max_positions: int = MAX_CONCURRENT_POSITIONS,
    max_deployed_pct: float = MAX_CAPITAL_DEPLOYED_PCT,
) -> float:
    """
    Confidence-weighted position size for a single new signal.

    Sizing logic
    ------------
    1. Compute base allocation = (max_deployed_pct * total_capital) / max_positions.
    2. Determine tier multiplier from the signal's rank within the qualifying set:
         top-third scores    → 1.5× base
         middle-third scores → 1.0× base
         bottom-third scores → 0.75× base
       NOTE: because this function is called per-signal (not on the full ranked
       set), we infer tier from the absolute score using tertile cut-offs derived
       from (CONFIDENCE_THRESHOLD, 1.0):
         score >= 2/3*(1-thresh) + thresh  → top tier
         score >= 1/3*(1-thresh) + thresh  → mid tier
         else                              → bottom tier
    3. Hard cap: position size is never above base * 1.5 (the max-tier allocation).
    4. Respect remaining capital: if (n_open_positions + 1) * base > max_deployed,
       the residual headroom is the effective ceiling.

    Parameters
    ----------
    score : float
        Predicted probability for this signal (>= CONFIDENCE_THRESHOLD).
    total_capital : float
        Total portfolio capital in INR.
    n_open_positions : int
        Number of currently open positions (before this new one).
    max_positions : int
        Maximum concurrent positions allowed.
    max_deployed_pct : float
        Maximum fraction of total_capital that may be deployed.

    Returns
    -------
    float
        Position size in INR. Returns 0.0 if no room for new positions.
    """
    if n_open_positions >= max_positions:
        logger.debug(
            f"compute_position_size: already at max_positions={max_positions}, returning 0."
        )
        return 0.0

    max_deployed = max_deployed_pct * total_capital
    base_size = max_deployed / max_positions

    # Tier boundaries within (CONFIDENCE_THRESHOLD, 1.0)
    t = CONFIDENCE_THRESHOLD
    band = 1.0 - t                    # width of qualifying range
    top_cut = t + (2.0 / 3.0) * band  # top-third starts here
    mid_cut = t + (1.0 / 3.0) * band  # middle-third starts here

    if score >= top_cut:
        multiplier = 1.5
    elif score >= mid_cut:
        multiplier = 1.0
    else:
        multiplier = 0.75

    sized = base_size * multiplier

    # Cap: never exceed per-slot base * 1.5
    sized = min(sized, base_size * 1.5)

    # Remaining capital headroom
    slots_remaining = max_positions - n_open_positions
    already_deployed_estimate = n_open_positions * base_size
    headroom = max(0.0, max_deployed - already_deployed_estimate)
    per_slot_headroom = headroom / slots_remaining if slots_remaining > 0 else 0.0

    final_size = min(sized, per_slot_headroom)

    logger.debug(
        f"compute_position_size: score={score:.3f} tier={'top' if score>=top_cut else 'mid' if score>=mid_cut else 'bot'} "
        f"multiplier={multiplier} base={base_size:,.0f} sized={sized:,.0f} "
        f"headroom={per_slot_headroom:,.0f} → {final_size:,.0f}"
    )

    return max(0.0, final_size)
