"""
Daily signal report generator.

Produces a human-readable Markdown report covering:
  - Today's qualifying signals with full entry/exit/size details
  - Ledger A performance summary
  - Audit trail (confidence threshold, zero-signal validation note)

The report is saved to logs/reports/YYYY-MM-DD.md and also returned
as a string so the Notifier can use it directly.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from loguru import logger

from config.settings import CONFIDENCE_THRESHOLD, LOG_DIR

_REPORTS_DIR = LOG_DIR / "reports"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(
    signals: pd.DataFrame,
    date: date,
    ledger_a_metrics: dict,
    universe_size: int,
) -> str:
    """
    Generate the daily signal report as a Markdown string.

    Parameters
    ----------
    signals : pd.DataFrame
        Qualifying signals for today. May be empty (zero signals is valid).
        Expected columns: symbol, direction, entry_low, entry_high,
        stop_loss, target, confidence, position_size, gap_status (optional).
    date : date
        The trading date this report covers.
    ledger_a_metrics : dict
        Output of LedgerA.get_metrics(). May be empty dict on first run.
    universe_size : int
        Total number of symbols that were scored today.

    Returns
    -------
    str
        Full report as a Markdown string.
    """
    n_signals = len(signals)
    sections: list[str] = []

    # ── 1. Header ─────────────────────────────────────────────────────────────
    sections.append(_header(date, universe_size, n_signals))

    # ── 2. Signals table ──────────────────────────────────────────────────────
    sections.append(_signals_section(signals))

    # ── 3. Ledger A summary ───────────────────────────────────────────────────
    sections.append(_ledger_section(ledger_a_metrics))

    # ── 4. Footer ─────────────────────────────────────────────────────────────
    sections.append(_footer())

    report = "\n\n".join(sections)
    logger.info(
        "Report generated: date={} signals={} universe={}",
        date,
        n_signals,
        universe_size,
    )
    return report


def save_report(report_md: str, date: date) -> Path:
    """
    Save report to logs/reports/YYYY-MM-DD.md.

    Parameters
    ----------
    report_md : str
        Markdown report string from generate_report().
    date : date
        The trading date (used for the filename).

    Returns
    -------
    Path
        Absolute path to the saved file.
    """
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _REPORTS_DIR / f"{date.isoformat()}.md"
    path.write_text(report_md, encoding="utf-8")
    logger.info("Report saved to {}", path)
    return path


# ---------------------------------------------------------------------------
# Private section builders
# ---------------------------------------------------------------------------

def _header(date: date, universe_size: int, n_signals: int) -> str:
    signal_word = "signal" if n_signals == 1 else "signals"
    return (
        f"# Piedpiper Daily Report — {date.strftime('%A, %d %B %Y')}\n\n"
        f"| Field              | Value                       |\n"
        f"|--------------------|-----------------------------|\n"
        f"| Report date        | {date.isoformat()}          |\n"
        f"| Universe scored    | {universe_size} stocks      |\n"
        f"| Signals generated  | {n_signals} {signal_word}   |\n"
        f"| Confidence gate    | {CONFIDENCE_THRESHOLD:.0%}  |"
    )


def _signals_section(signals: pd.DataFrame) -> str:
    header = "## Qualifying Signals"

    if signals.empty:
        return (
            f"{header}\n\n"
            "_No signals cleared the confidence bar today. "
            "This is a valid output — sit on your hands._"
        )

    # Column display order and formatting
    col_map = {
        "symbol":        ("Symbol",        lambda v: str(v)),
        "direction":     ("Dir",           lambda v: str(v)),
        "entry_low":     ("Entry Low",     lambda v: f"₹{v:,.2f}"),
        "entry_high":    ("Entry High",    lambda v: f"₹{v:,.2f}"),
        "stop_loss":     ("Stop",          lambda v: f"₹{v:,.2f}"),
        "target":        ("Target",        lambda v: f"₹{v:,.2f}"),
        "confidence":    ("Conf",          lambda v: f"{v:.1%}"),
        "position_size": ("Size (₹)",      lambda v: f"₹{v:,.0f}"),
        "gap_status":    ("Gap Status",    lambda v: str(v) if pd.notna(v) else "—"),
    }

    # Build table header
    visible_cols = [col for col in col_map if col in signals.columns]
    col_headers = [col_map[col][0] for col in visible_cols]
    dividers = ["---"] * len(col_headers)

    rows = [
        "| " + " | ".join(col_headers) + " |",
        "| " + " | ".join(dividers) + " |",
    ]

    for _, row in signals.iterrows():
        cells = [col_map[col][1](row[col]) for col in visible_cols]
        rows.append("| " + " | ".join(cells) + " |")

    return header + "\n\n" + "\n".join(rows)


def _ledger_section(metrics: dict) -> str:
    header = "## Ledger A — Paper Portfolio Summary"

    if not metrics or "error" in metrics:
        reason = metrics.get("error", "no data") if metrics else "not yet initialised"
        return f"{header}\n\n_Ledger A metrics unavailable: {reason}_"

    equity     = metrics.get("current_equity", 0.0)
    sharpe     = metrics.get("sharpe", float("nan"))
    max_dd     = metrics.get("max_drawdown", float("nan"))
    hit_rate   = metrics.get("hit_rate", float("nan"))
    n_trades   = metrics.get("n_trades", 0)
    net_pnl    = metrics.get("net_pnl", 0.0)
    open_pos   = metrics.get("n_open_positions", "—")

    def _fmt_pct(v) -> str:
        try:
            return f"{v:.1%}"
        except (TypeError, ValueError):
            return "—"

    def _fmt_f(v, fmt=".2f") -> str:
        try:
            return format(v, fmt)
        except (TypeError, ValueError):
            return "—"

    return (
        f"{header}\n\n"
        f"| Metric             | Value                |\n"
        f"|--------------------|----------------------|\n"
        f"| Current equity     | ₹{equity:,.0f}       |\n"
        f"| Net P&L (all time) | ₹{net_pnl:,.0f}      |\n"
        f"| Sharpe ratio       | {_fmt_f(sharpe)}     |\n"
        f"| Max drawdown       | {_fmt_pct(max_dd)}   |\n"
        f"| Hit rate           | {_fmt_pct(hit_rate)} |\n"
        f"| Closed trades      | {n_trades}           |\n"
        f"| Open positions     | {open_pos}           |"
    )


def _footer() -> str:
    return (
        "---\n\n"
        "_Confidence threshold: "
        f"{CONFIDENCE_THRESHOLD:.0%} — tuned during backtesting; do not adjust post-deployment._\n\n"
        "_Zero signals is a **valid** output. The system held back because no stock "
        "cleared the probability bar — that is the gate working as designed._"
    )
