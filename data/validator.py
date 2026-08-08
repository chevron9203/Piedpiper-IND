"""
Data quality validation.

Checks raw and adjusted OHLCV for:
- Gaps (missing trading days)
- Duplicate timestamps
- Zero-volume days
- Price jumps inconsistent with known corporate actions
- EOD2 vs SmartAPI drift (Phase 0 cross-check)
- EOD2 adjustment validation against known split/bonus events
"""
from datetime import date, timedelta
from typing import Optional
import pandas as pd
import numpy as np
from loguru import logger
from data.holiday_calendar import trading_days_between


# ── Per-series checks ─────────────────────────────────────────────────────────

def check_gaps(df: pd.DataFrame, symbol: str,
               from_date: date, to_date: date) -> list[date]:
    """Return list of trading days missing from df."""
    expected = set(trading_days_between(from_date, to_date))
    if df.empty:
        return sorted(expected)

    actual = set(d.date() if hasattr(d, "date") else d for d in df.index)
    missing = sorted(expected - actual)
    if missing:
        logger.warning("{}: {} missing trading days (first: {})", symbol, len(missing), missing[0])
    return missing


def check_duplicates(df: pd.DataFrame, symbol: str) -> int:
    """Return count of duplicate index entries."""
    dups = df.index.duplicated().sum()
    if dups:
        logger.warning("{}: {} duplicate timestamps", symbol, dups)
    return int(dups)


def check_zero_volume(df: pd.DataFrame, symbol: str,
                       max_consecutive: int = 3) -> list[date]:
    """Return dates with zero or null volume. Flag if >max_consecutive consecutive."""
    if "volume" not in df.columns:
        return []
    zero = df[df["volume"].fillna(0) == 0]
    dates = [d.date() if hasattr(d, "date") else d for d in zero.index]

    # Check consecutive zero-volume runs
    consecutive = 0
    prev = None
    for d in dates:
        if prev and (d - prev).days <= 2:
            consecutive += 1
            if consecutive >= max_consecutive:
                logger.warning("{}: {} consecutive zero-volume days near {}", symbol, consecutive + 1, d)
                break
        else:
            consecutive = 0
        prev = d

    if dates:
        logger.debug("{}: {} zero-volume days", symbol, len(dates))
    return dates


def check_price_jumps(df: pd.DataFrame, symbol: str,
                       threshold_pct: float = 0.20) -> list[dict]:
    """
    Return dates where close-to-close return exceeds threshold_pct
    (default 20%) — likely unadjusted corporate action or bad data.
    """
    if df.empty or "close" not in df.columns:
        return []
    returns = df["close"].pct_change().abs()
    jumps = returns[returns > threshold_pct]
    result = []
    for dt, r in jumps.items():
        d = dt.date() if hasattr(dt, "date") else dt
        result.append({"date": d, "return_pct": round(r * 100, 2)})
        logger.warning("{}: price jump {:.1f}% on {}", symbol, r * 100, d)
    return result


def check_ohlcv_consistency(df: pd.DataFrame, symbol: str) -> int:
    """Check that High >= max(Open,Close) and Low <= min(Open,Close). Returns error count."""
    errors = 0
    if not all(c in df.columns for c in ["open", "high", "low", "close"]):
        return 0
    bad_high = df[df["high"] < df[["open", "close"]].max(axis=1)]
    bad_low = df[df["low"] > df[["open", "close"]].min(axis=1)]
    errors = len(bad_high) + len(bad_low)
    if errors:
        logger.warning("{}: {} OHLC consistency violations", symbol, errors)
    return errors


def validate_series(df: pd.DataFrame, symbol: str,
                    from_date: date, to_date: date) -> dict:
    """Run all checks on a single OHLCV series. Returns summary dict."""
    return {
        "symbol": symbol,
        "rows": len(df),
        "missing_days": len(check_gaps(df, symbol, from_date, to_date)),
        "duplicates": check_duplicates(df, symbol),
        "zero_volume_days": len(check_zero_volume(df, symbol)),
        "price_jumps": len(check_price_jumps(df, symbol)),
        "ohlc_errors": check_ohlcv_consistency(df, symbol),
    }


# ── Cross-source validation ───────────────────────────────────────────────────

def cross_check_closes(raw_df: pd.DataFrame, reference_df: pd.DataFrame,
                        symbol: str, tolerance_pct: float = 0.005) -> dict:
    """
    Compare closing prices between two sources on the same dates.
    Used to validate SmartAPI raw vs bhavcopy.
    tolerance_pct: 0.5% — price normalisation differences are expected but >0.5% is suspicious.
    """
    raw_df = _normalise_index(raw_df)
    reference_df = _normalise_index(reference_df)
    common = raw_df.index.intersection(reference_df.index)
    if len(common) < 5:
        return {"status": "insufficient_overlap", "common_dates": len(common)}

    raw_close = raw_df.loc[common, "close"]
    ref_close = reference_df.loc[common, "close"]
    diff_pct = ((raw_close - ref_close) / ref_close).abs()
    bad = diff_pct[diff_pct > tolerance_pct]

    result = {
        "status": "ok" if bad.empty else "discrepancies_found",
        "symbol": symbol,
        "common_dates": len(common),
        "discrepant_dates": int(len(bad)),
        "max_diff_pct": float(diff_pct.max()) if len(diff_pct) > 0 else 0.0,
        "sample_bad_dates": bad.index[:5].strftime("%Y-%m-%d").tolist(),
    }
    if bad.empty:
        logger.info("{}: cross-check OK ({} common dates)", symbol, len(common))
    else:
        logger.warning("{}: {} discrepant dates (max {:.2f}%)", symbol, len(bad), diff_pct.max() * 100)
    return result


def _normalise_index(df: pd.DataFrame) -> pd.DataFrame:
    """Strip timezone info and normalise index to date-only for cross-source comparison."""
    df = df.copy()
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = pd.to_datetime(df.index).normalize()
    return df


def validate_eod2_adjustment(symbol: str, smartapi_df: pd.DataFrame,
                              eod2_df: pd.DataFrame,
                              known_ca_dates: list[date],
                              tolerance_pct: float = 0.005) -> dict:
    """
    Phase 0 check: verify EOD2 adjusted data is consistent with SmartAPI.

    Both SmartAPI and EOD2 return backward-adjusted data, so the correct
    validation is cross-source agreement (ratio ≈ 1.0 everywhere).
    Disagreement > tolerance_pct at/around a CA date indicates one source
    missed or mis-applied the adjustment.

    Additionally checks EOD2 for suspicious >40% price jumps (unadjusted CA).
    """
    if smartapi_df.empty or eod2_df.empty:
        return {"symbol": symbol, "overall": "no_data", "ca_checks": []}

    smartapi_df = _normalise_index(smartapi_df)
    eod2_df = _normalise_index(eod2_df)

    common = smartapi_df.index.intersection(eod2_df.index)
    if len(common) < 5:
        return {"symbol": symbol, "overall": "insufficient_overlap",
                "common_dates": len(common), "ca_checks": []}

    api_close = smartapi_df.loc[common, "close"]
    eod_close = eod2_df.loc[common, "close"]
    diff_pct = ((eod_close - api_close) / api_close).abs()

    results = []
    for ca_date in known_ca_dates:
        ca_dt = pd.Timestamp(ca_date)
        # Check agreement in a ±20-day window around the CA
        window_start = ca_dt - pd.Timedelta(days=30)
        window_end = ca_dt + pd.Timedelta(days=30)
        window_idx = common[(common >= window_start) & (common <= window_end)]

        if len(window_idx) < 3:
            results.append({"ca_date": str(ca_date), "status": "insufficient_window_data"})
            continue

        window_diff = diff_pct.loc[window_idx]
        max_diff = float(window_diff.max())
        if max_diff <= tolerance_pct:
            results.append({
                "ca_date": str(ca_date),
                "status": "ok",
                "max_diff_pct": round(max_diff * 100, 3),
                "note": "EOD2 and SmartAPI agree around CA date",
            })
            logger.info("{}: CA window agreement OK (max diff {:.3f}%)", symbol, max_diff * 100)
        else:
            bad = window_diff[window_diff > tolerance_pct]
            results.append({
                "ca_date": str(ca_date),
                "status": "disagreement_near_ca",
                "max_diff_pct": round(max_diff * 100, 3),
                "discrepant_dates": bad.index.strftime("%Y-%m-%d").tolist()[:5],
            })
            logger.warning("{}: EOD2/SmartAPI disagree near CA {} (max diff {:.2f}%)",
                           symbol, ca_date, max_diff * 100)

    # Also check for large price jumps in EOD2 (>40% = unadjusted CA)
    eod2_returns = eod2_df["close"].pct_change().abs()
    big_jumps = eod2_returns[eod2_returns > 0.40].index.strftime("%Y-%m-%d").tolist()
    if big_jumps:
        logger.warning("{}: EOD2 has >40% price jumps (possible missing adjustment): {}",
                       symbol, big_jumps[:3])

    overall = "ok" if all(r.get("status") == "ok" for r in results) else "issues_found"
    return {
        "symbol": symbol,
        "overall": overall,
        "common_dates": len(common),
        "overall_max_diff_pct": round(float(diff_pct.max()), 4) * 100,
        "unadjusted_jumps_in_eod2": big_jumps[:5],
        "ca_checks": results,
    }
