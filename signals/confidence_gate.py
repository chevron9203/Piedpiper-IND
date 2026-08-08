"""
Confidence gate — filters ranked signals by probability threshold.

Zero signals passing is a valid and expected output on low-conviction days.
Never treat an empty result as an error; let the caller decide what to do.
"""
from __future__ import annotations

import pandas as pd
from loguru import logger

from config.settings import CONFIDENCE_THRESHOLD


def apply_gate(
    ranked: pd.DataFrame,
    threshold: float = CONFIDENCE_THRESHOLD,
) -> pd.DataFrame:
    """
    Filter ranked signals by confidence threshold.

    Parameters
    ----------
    ranked : pd.DataFrame
        Output of models.scorer.rank_universe(). Expected columns:
        symbol, score, rank.
    threshold : float
        Minimum score required to pass the gate. Defaults to
        CONFIDENCE_THRESHOLD from settings.

    Returns
    -------
    pd.DataFrame
        Subset of ``ranked`` where score >= threshold, preserving original
        column order and dtypes. May be empty — that is correct output,
        not a bug.
    """
    if ranked.empty:
        logger.info(
            "confidence_gate: received empty ranked DataFrame — "
            "0/0 signals passed (threshold={:.2f})",
            threshold,
        )
        return ranked.copy()

    total = len(ranked)
    passed = ranked[ranked["score"] >= threshold].copy()
    n_passed = len(passed)

    logger.info(
        "confidence_gate: {}/{} signals passed threshold={:.2f} "
        "(rejected: {})",
        n_passed,
        total,
        threshold,
        total - n_passed,
    )

    if n_passed == 0:
        logger.info(
            "confidence_gate: no signals cleared the bar today — "
            "this is valid output, not an error"
        )

    return passed.reset_index(drop=True)


def gate_is_empty(signals: pd.DataFrame) -> bool:
    """Return True if no signals passed the confidence gate."""
    return signals.empty
