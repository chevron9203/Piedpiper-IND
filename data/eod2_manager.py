"""
EOD2 integration for adjusted OHLCV data.

EOD2 (github.com/BennyThadikaran/eod2) maintains daily NSE OHLCV
pre-adjusted for splits and bonuses. We use it as our primary source
for adjusted data, validated against SmartAPI raw + bhavcopy in Phase 0.

EOD2 stores data as CSV files per symbol in a flat directory.
This module wraps that interface and exposes DataFrames.

Drift-check schedule: quarterly — run after any known corporate action.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from datetime import date
import pandas as pd
from loguru import logger
from config import settings

EOD2_DIR = settings.EOD2_DATA_DIR
EOD2_DAILY_DIR = EOD2_DIR / "src" / "eod2_data" / "daily"


def _eod2_csv_path(symbol: str) -> Path:
    """Resolve the path to EOD2's CSV for a given symbol.

    EOD2 writes LOWERCASE filenames (reliance.csv, no -EQ suffix). This used to
    build an uppercase name, which still resolved on macOS (case-insensitive FS)
    but matched nothing on the Linux box -- so every universe symbol silently
    "went missing" and adjusted_ohlcv froze. Try lowercase first, then uppercase,
    so it works on either filesystem regardless of how EOD2 names things."""
    sym = symbol.replace("-EQ", "").replace("-eq", "")
    lower = EOD2_DAILY_DIR / f"{sym.lower()}.csv"
    if lower.exists():
        return lower
    upper = EOD2_DAILY_DIR / f"{sym.upper()}.csv"
    return upper if upper.exists() else lower


def is_eod2_available() -> bool:
    """Check if EOD2 data directory and at least one CSV exist."""
    return EOD2_DAILY_DIR.exists() and any(EOD2_DAILY_DIR.glob("*.csv"))


def setup_eod2() -> None:
    """
    Clone and initialise EOD2 if not already present.

    EOD2 repo structure (actual, verified):
      eod2/
        setup_data.py      — downloads eod2_data zip (historical CSVs)
        src/
          init.py          — initialises metadata, runs first sync
          dget.py          — daily update (run after init)
          eod2_data/daily/ — one CSV per symbol (created by setup_data.py)

    This is a one-time setup; subsequent updates use update_eod2().
    """
    if is_eod2_available():
        logger.info("EOD2 already initialised at {}", EOD2_DAILY_DIR)
        return

    EOD2_DIR.mkdir(parents=True, exist_ok=True)

    # Clone if not already cloned
    if not (EOD2_DIR / "setup_data.py").exists():
        logger.info("Cloning EOD2 repository...")
        result = subprocess.run(
            ["git", "clone", "https://github.com/BennyThadikaran/eod2.git", str(EOD2_DIR)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"EOD2 clone failed:\n{result.stderr}")

        logger.info("Installing EOD2 dependencies...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(EOD2_DIR / "requirements.txt")],
            check=True, capture_output=True
        )

    # Step 1: download historical data zip (eod2_data submodule/zip)
    logger.info("Downloading EOD2 historical data (this may take several minutes)...")
    result = subprocess.run(
        [sys.executable, str(EOD2_DIR / "setup_data.py")],
        capture_output=True, text=True, cwd=str(EOD2_DIR)
    )
    if result.returncode != 0:
        raise RuntimeError(f"EOD2 setup_data.py failed:\n{result.stderr}\n{result.stdout}")
    logger.info("Historical data downloaded")

    # Step 2: run init to bring data up to date
    logger.info("Running EOD2 init to sync to current date...")
    result = subprocess.run(
        [sys.executable, str(EOD2_DIR / "src" / "init.py")],
        capture_output=True, text=True, cwd=str(EOD2_DIR / "src")
    )
    # init.py may return non-zero on first run if no config yet — check for data instead
    logger.info("EOD2 init stdout: {}", result.stdout[:500] if result.stdout else "(empty)")
    if result.stderr:
        logger.warning("EOD2 init stderr: {}", result.stderr[:500])

    if is_eod2_available():
        logger.info("EOD2 initialised successfully — {} CSVs available", len(list_available_symbols()))
    else:
        raise RuntimeError(
            f"EOD2 init completed but no daily CSVs found at {EOD2_DAILY_DIR}. "
            f"init stdout: {result.stdout[:1000]}"
        )


def update_eod2() -> None:
    """Run EOD2 daily update (init.py) to fetch latest adjusted data from NSE."""
    if not is_eod2_available():
        raise RuntimeError("EOD2 not initialised. Run setup_eod2() first.")

    logger.info("Updating EOD2 data...")
    result = subprocess.run(
        [sys.executable, str(EOD2_DIR / "src" / "init.py")],
        capture_output=True, text=True, cwd=str(EOD2_DIR / "src")
    )
    if result.returncode != 0:
        logger.error("EOD2 update failed:\n{}", result.stderr)
        raise RuntimeError(f"EOD2 update failed: {result.stderr}")
    logger.info("EOD2 update complete")


def load_adjusted_ohlcv(symbol: str, from_date: date | None = None,
                         to_date: date | None = None) -> pd.DataFrame:
    """
    Load EOD2-adjusted OHLCV for a symbol.

    Returns a DataFrame with columns: open, high, low, close, volume
    indexed by date. Raises FileNotFoundError if symbol is not in EOD2.
    """
    path = _eod2_csv_path(symbol)
    if not path.exists():
        raise FileNotFoundError(f"EOD2 data not found for {symbol!r} at {path}")

    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    df.index = pd.to_datetime(df.index).normalize()
    df.columns = [c.lower() for c in df.columns]

    # Standardise column names (EOD2 uses Open/High/Low/Close/Volume)
    rename = {"open": "open", "high": "high", "low": "low",
               "close": "close", "volume": "volume"}
    df = df.rename(columns={c.capitalize(): c for c in rename})
    df = df[["open", "high", "low", "close", "volume"]].dropna()

    if from_date:
        df = df[df.index.date >= from_date]
    if to_date:
        df = df[df.index.date <= to_date]

    return df.sort_index()


def list_available_symbols() -> list[str]:
    """Return all symbols available in EOD2."""
    if not EOD2_DAILY_DIR.exists():
        return []
    return [p.stem for p in EOD2_DAILY_DIR.glob("*.csv")]


def drift_check(symbol: str, raw_df: pd.DataFrame, tolerance_pct: float = 0.02) -> dict:
    """
    Compare EOD2 adjusted close vs raw close for dates without known CAs.
    Returns a dict with check result and any anomalous dates.

    tolerance_pct: max allowed ratio between adjusted and raw (e.g. 0.02 = 2%).
    This detects adjustment drift, not normal price differences on CA dates.
    """
    try:
        adj_df = load_adjusted_ohlcv(symbol)
    except FileNotFoundError:
        return {"status": "skip", "reason": f"{symbol} not in EOD2"}

    common_dates = raw_df.index.intersection(adj_df.index)
    if len(common_dates) < 10:
        return {"status": "insufficient_data", "common_dates": len(common_dates)}

    raw_close = raw_df.loc[common_dates, "close"]
    adj_close = adj_df.loc[common_dates, "close"]

    # Ratio should be roughly constant between CA events (adjustment factor)
    ratio = adj_close / raw_close
    # Use rolling median to establish "expected" ratio in each window
    rolling_median = ratio.rolling(20, min_periods=5).median()
    deviation = (ratio - rolling_median).abs() / rolling_median

    anomalies = deviation[deviation > tolerance_pct]

    return {
        "status": "ok" if len(anomalies) == 0 else "anomalies_found",
        "symbol": symbol,
        "common_dates": len(common_dates),
        "anomalous_dates": anomalies.index.strftime("%Y-%m-%d").tolist(),
        "max_deviation_pct": float(deviation.max()) if len(deviation) > 0 else 0.0,
    }


if __name__ == "__main__":
    logger.info("EOD2 available: {}", is_eod2_available())
    symbols = list_available_symbols()
    logger.info("Available symbols: {} (first 5: {})", len(symbols), symbols[:5])
    if symbols:
        df = load_adjusted_ohlcv(symbols[0])
        logger.info("Sample ({}):\n{}", symbols[0], df.tail())
