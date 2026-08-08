"""
Ledger A vs Ledger B reconciliation.

Answers three questions:
  1. What did Ledger A trade that Ledger B declined?
     (Missed opportunities — did B leave money on the table?)

  2. What did both A and B trade?
     (Matched trades — how did real fills compare to virtual?)

  3. Overall: did Feron add or subtract value vs the autonomous system?

All comparisons are on a per-symbol × signal_date basis.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from loguru import logger

from portfolio.ledger_a import LedgerA
from portfolio.ledger_b import LedgerB


def compare_ledgers(ledger_a: LedgerA, ledger_b: LedgerB) -> dict:
    """
    Side-by-side comparison of virtual (A) vs real (B) portfolio performance.

    Parameters
    ----------
    ledger_a : LedgerA instance (must have closed_trades populated)
    ledger_b : LedgerB instance (reads from DuckDB)

    Returns
    -------
    dict with keys:
      a_only          : pd.DataFrame — signals taken by A but declined/missed by B
      both            : pd.DataFrame — signals taken by both, with fill diff columns
      b_only          : pd.DataFrame — trades in B not reflected in A (edge case)
      declined_by_b   : pd.DataFrame — signals in B's declined log that A traded
      summary         : dict — scalar summary metrics
    """
    # ── Load data ─────────────────────────────────────────────────────────────
    a_trades = ledger_a.to_dataframe()
    b_closed = ledger_b.get_closed_trades()
    b_open = ledger_b.get_open_trades()
    b_declined = ledger_b.get_declined_signals()

    b_all_trades = pd.concat([b_closed, b_open], ignore_index=True) if not b_closed.empty or not b_open.empty else pd.DataFrame()

    # Normalise join keys
    if not a_trades.empty:
        a_trades = _normalise(a_trades, ["symbol", "signal_date", "entry_date",
                                          "exit_date", "net_pnl", "gross_pnl",
                                          "entry_price", "quantity"])
    if not b_all_trades.empty:
        b_all_trades = _normalise(b_all_trades, ["symbol", "signal_date", "entry_date",
                                                   "exit_date", "net_pnl", "gross_pnl",
                                                   "entry_price", "quantity"])
    if not b_declined.empty:
        b_declined = _normalise(b_declined, ["symbol", "signal_date"])

    # ── Match key: symbol + signal_date ───────────────────────────────────────
    if a_trades.empty and b_all_trades.empty:
        return _empty_result()

    a_keys = set() if a_trades.empty else set(
        zip(a_trades["symbol"], a_trades["signal_date"].dt.date)
    )
    b_keys = set() if b_all_trades.empty else set(
        zip(b_all_trades["symbol"], pd.to_datetime(b_all_trades["signal_date"]).dt.date)
    )
    declined_keys = set() if b_declined.empty else set(
        zip(b_declined["symbol"], pd.to_datetime(b_declined["signal_date"]).dt.date)
    )

    # ── A only (A traded, B neither traded nor logged declined) ───────────────
    a_only_keys = a_keys - b_keys
    if not a_trades.empty:
        a_only_mask = a_trades.apply(
            lambda r: (r["symbol"], r["signal_date"].date()) in a_only_keys, axis=1
        )
        a_only = a_trades[a_only_mask].copy()
    else:
        a_only = pd.DataFrame()

    # ── Declined by B but traded by A ─────────────────────────────────────────
    declined_by_b_keys = a_keys & declined_keys
    if not a_trades.empty and declined_by_b_keys:
        declined_mask = a_trades.apply(
            lambda r: (r["symbol"], r["signal_date"].date()) in declined_by_b_keys,
            axis=1
        )
        declined_by_b = a_trades[declined_mask].copy()
        # Merge in B's decline reason
        if not b_declined.empty:
            b_dec_sub = b_declined[["symbol", "signal_date", "notes"]].rename(
                columns={"notes": "decline_reason"}
            )
            declined_by_b = declined_by_b.merge(
                b_dec_sub, on=["symbol", "signal_date"], how="left"
            )
    else:
        declined_by_b = pd.DataFrame()

    # ── Both traded ───────────────────────────────────────────────────────────
    both_keys = a_keys & b_keys
    if both_keys and not a_trades.empty and not b_all_trades.empty:
        a_both_mask = a_trades.apply(
            lambda r: (r["symbol"], r["signal_date"].date()) in both_keys, axis=1
        )
        b_both_mask = b_all_trades.apply(
            lambda r: (r["symbol"], pd.Timestamp(r["signal_date"]).date()) in both_keys,
            axis=1
        )
        a_both = a_trades[a_both_mask].copy()
        b_both = b_all_trades[b_both_mask].copy()

        both = a_both.merge(
            b_both[["symbol", "signal_date", "entry_date", "entry_price",
                     "net_pnl", "gross_pnl", "exit_date", "quantity"]],
            on=["symbol", "signal_date"],
            suffixes=("_a", "_b"),
            how="outer",
        )

        # Entry timing difference: days between A's fill and B's actual fill
        if "entry_date_a" in both.columns and "entry_date_b" in both.columns:
            both["entry_timing_diff_days"] = (
                pd.to_datetime(both["entry_date_b"])
                - pd.to_datetime(both["entry_date_a"])
            ).dt.days

        # Fill price difference
        if "entry_price_a" in both.columns and "entry_price_b" in both.columns:
            both["entry_price_diff"] = both["entry_price_b"] - both["entry_price_a"]
            both["entry_price_diff_pct"] = (
                both["entry_price_diff"] / both["entry_price_a"]
            )

        # Net P&L difference
        if "net_pnl_a" in both.columns and "net_pnl_b" in both.columns:
            both["net_pnl_diff"] = both["net_pnl_b"] - both["net_pnl_a"]
    else:
        both = pd.DataFrame()

    # ── B only (B traded something A didn't signal) ───────────────────────────
    b_only_keys = b_keys - a_keys
    if not b_all_trades.empty and b_only_keys:
        b_only_mask = b_all_trades.apply(
            lambda r: (r["symbol"], pd.Timestamp(r["signal_date"]).date()) in b_only_keys,
            axis=1
        )
        b_only = b_all_trades[b_only_mask].copy()
    else:
        b_only = pd.DataFrame()

    # ── Summary metrics ───────────────────────────────────────────────────────
    summary = _build_summary(a_trades, b_closed, both, a_only, declined_by_b)

    logger.info(
        "Reconciliation | A_trades={} | B_trades={} | both={} | a_only={} | declined_by_b={}",
        len(a_trades), len(b_all_trades), len(both),
        len(a_only), len(declined_by_b)
    )

    return {
        "a_only":        a_only,
        "both":          both,
        "b_only":        b_only,
        "declined_by_b": declined_by_b,
        "summary":       summary,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_summary(
    a_trades: pd.DataFrame,
    b_closed: pd.DataFrame,
    both: pd.DataFrame,
    a_only: pd.DataFrame,
    declined_by_b: pd.DataFrame,
) -> dict:
    """Compute scalar summary values for the reconciliation report."""
    s: dict = {}

    # A performance
    if not a_trades.empty and "net_pnl" in a_trades.columns:
        s["a_total_net_pnl"] = float(a_trades["net_pnl"].sum())
        s["a_n_trades"] = len(a_trades)
        s["a_hit_rate"] = float((a_trades["net_pnl"] > 0).mean())
    else:
        s.update({"a_total_net_pnl": 0.0, "a_n_trades": 0, "a_hit_rate": 0.0})

    # B performance (closed trades only)
    if not b_closed.empty and "net_pnl" in b_closed.columns:
        s["b_total_net_pnl"] = float(b_closed["net_pnl"].sum())
        s["b_n_trades"] = len(b_closed)
        s["b_hit_rate"] = float((b_closed["net_pnl"] > 0).mean())
    else:
        s.update({"b_total_net_pnl": 0.0, "b_n_trades": 0, "b_hit_rate": 0.0})

    # Opportunity cost: P&L A made on trades B declined
    if not declined_by_b.empty and "net_pnl" in declined_by_b.columns:
        s["missed_pnl_by_b"] = float(declined_by_b["net_pnl"].sum())
        s["n_declined_by_b"] = len(declined_by_b)
    else:
        s.update({"missed_pnl_by_b": 0.0, "n_declined_by_b": 0})

    # On matched trades: did B's real fills do better or worse than A's virtual?
    if not both.empty:
        if "net_pnl_diff" in both.columns:
            net_diff = both["net_pnl_diff"].dropna()
            s["matched_total_pnl_diff_b_vs_a"] = float(net_diff.sum())
            s["matched_avg_pnl_diff_per_trade"] = float(net_diff.mean()) if len(net_diff) else 0.0
        if "entry_timing_diff_days" in both.columns:
            s["avg_entry_timing_diff_days"] = float(
                both["entry_timing_diff_days"].dropna().mean()
            ) if not both["entry_timing_diff_days"].dropna().empty else 0.0
        if "entry_price_diff_pct" in both.columns:
            s["avg_entry_price_diff_pct"] = float(
                both["entry_price_diff_pct"].dropna().mean()
            ) if not both["entry_price_diff_pct"].dropna().empty else 0.0
    else:
        s.update({
            "matched_total_pnl_diff_b_vs_a": 0.0,
            "matched_avg_pnl_diff_per_trade": 0.0,
            "avg_entry_timing_diff_days": 0.0,
            "avg_entry_price_diff_pct": 0.0,
        })

    # Overall verdict
    s["b_added_value_vs_a"] = s["b_total_net_pnl"] > s["a_total_net_pnl"]
    s["net_alpha_b_over_a"] = s["b_total_net_pnl"] - s["a_total_net_pnl"]

    return s


def _normalise(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Keep only expected columns that exist, parse dates."""
    existing = [c for c in columns if c in df.columns]
    df = df[existing].copy()
    for col in ["signal_date", "entry_date", "exit_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _empty_result() -> dict:
    return {
        "a_only":        pd.DataFrame(),
        "both":          pd.DataFrame(),
        "b_only":        pd.DataFrame(),
        "declined_by_b": pd.DataFrame(),
        "summary":       {
            "a_total_net_pnl": 0.0, "a_n_trades": 0, "a_hit_rate": 0.0,
            "b_total_net_pnl": 0.0, "b_n_trades": 0, "b_hit_rate": 0.0,
            "missed_pnl_by_b": 0.0, "n_declined_by_b": 0,
            "matched_total_pnl_diff_b_vs_a": 0.0,
            "matched_avg_pnl_diff_per_trade": 0.0,
            "avg_entry_timing_diff_days": 0.0,
            "avg_entry_price_diff_pct": 0.0,
            "b_added_value_vs_a": False,
            "net_alpha_b_over_a": 0.0,
        },
    }
