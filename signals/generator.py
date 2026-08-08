"""
Signal generator — orchestrates the daily signal production pipeline.

Pipeline (run once per trading day, after market close):
  1. Load adjusted OHLCV from store for all universe symbols
  2. Load Nifty 50 index and India VIX data
  3. Compute features via FeaturePipeline
  4. Load trained LightGBM model (specified path or latest in data_store/models/)
  5. Score the universe cross-sectionally
  6. Apply confidence gate
  7. Compute entry zone, cost-aware stop-loss and take-profit
  8. Compute suggested position size
  9. Save signals to DuckDB store
 10. Return signals DataFrame (empty DataFrame is valid output)

Design notes
------------
- All errors within per-symbol processing are caught and logged; the pipeline
  never crashes on a single bad symbol.
- Model loading falls back to the most recently written file in data_store/models/
  if no explicit path is given.
- Position sizes are computed against ``LedgerA.capital`` if a ledger is
  supplied, otherwise against settings.STARTING_VIRTUAL_CAPITAL.
"""
from __future__ import annotations

import glob
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from config.settings import (
    BARRIER_MAX_HOLDING_DAYS,
    CONFIDENCE_THRESHOLD,
    DATA_DIR,
    FEATURE_VERSION,
    NIFTY50_SYMBOL,
    INDIA_VIX_SYMBOL,
    STARTING_VIRTUAL_CAPITAL,
)
from config.cost_model import cost_adjusted_barrier
from features.pipeline import FeaturePipeline
from models.scorer import rank_universe, compute_position_size
from signals.confidence_gate import apply_gate, gate_is_empty
from storage.store import load_adjusted_ohlcv, save_signals


_MODELS_DIR = DATA_DIR / "models"


class SignalGenerator:
    """
    Orchestrates the daily signal generation pipeline.

    Parameters
    ----------
    model_path : str | Path | None
        Explicit path to a saved LightGBM `.txt` / `.pkl` model file.
        If None, the most recently modified file in data_store/models/ is used.
    """

    def __init__(self, model_path: str | Path | None = None):
        self.pipeline = FeaturePipeline(version=FEATURE_VERSION)
        self.model_path: Optional[Path] = Path(model_path) if model_path else None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        universe: pd.DataFrame,
        date: date | None = None,
        n_open_positions: int = 0,
        total_capital: float = STARTING_VIRTUAL_CAPITAL,
    ) -> pd.DataFrame:
        """
        Full daily pipeline for one trading date.

        Parameters
        ----------
        universe : pd.DataFrame
            Screened universe with at least 'symbol' and 'sector' columns.
            Produced by data.universe.get_universe().
        date : date | None
            Signal date. Defaults to today.
        n_open_positions : int
            Currently open positions in Ledger A (drives position sizing).
        total_capital : float
            Total portfolio capital in INR (drives position sizing).

        Returns
        -------
        pd.DataFrame
            Qualifying signals with all computed fields. Empty if none qualify.
        """
        run_date = date or pd.Timestamp.today().date()
        logger.info("SignalGenerator.run — date={} universe_size={}", run_date, len(universe))

        # ── Step 1: Load OHLCV ────────────────────────────────────────────────
        ohlcv_map: dict[str, pd.DataFrame] = {}
        for symbol in universe["symbol"]:
            try:
                df = load_adjusted_ohlcv(symbol)
                if df.empty or len(df) < 60:
                    logger.debug("Skipping {} — insufficient OHLCV rows ({})", symbol, len(df))
                    continue
                ohlcv_map[symbol] = df
            except Exception as exc:
                logger.warning("OHLCV load failed for {}: {}", symbol, exc)

        if not ohlcv_map:
            logger.error("No OHLCV data loaded — aborting signal run")
            return pd.DataFrame()

        logger.info("Loaded OHLCV for {}/{} symbols", len(ohlcv_map), len(universe))

        # ── Step 2: Load index & VIX data ─────────────────────────────────────
        index_df = self._load_index_data(NIFTY50_SYMBOL)
        vix_df = self._load_index_data(INDIA_VIX_SYMBOL)

        # ── Steps 3–4: Compute features and load model ─────────────────────────
        model = self._load_model()
        if model is None:
            logger.error("No trained model found — cannot generate signals")
            return pd.DataFrame()

        # ── Step 5: Score universe ─────────────────────────────────────────────
        predictions = self._score_universe(ohlcv_map, index_df, vix_df, model, run_date)
        if predictions.empty:
            logger.warning("Scorer returned empty predictions — no signals")
            return pd.DataFrame()

        # ── Step 6: Rank + gate ────────────────────────────────────────────────
        ranked = rank_universe(predictions, confidence_threshold=CONFIDENCE_THRESHOLD)
        gated = apply_gate(ranked, threshold=CONFIDENCE_THRESHOLD)

        if gate_is_empty(gated):
            logger.info("No signals passed confidence gate on {} — valid outcome", run_date)
            return pd.DataFrame()

        # ── Steps 7–8: Compute levels and position sizes ──────────────────────
        sector_map: dict[str, str] = (
            universe.set_index("symbol")["sector"].to_dict()
            if "sector" in universe.columns
            else {}
        )
        nifty100_symbols = self._get_nifty100_symbols()

        signal_rows: list[dict] = []
        for idx, row in gated.iterrows():
            symbol: str = row["symbol"]
            score: float = float(row["score"])

            # Previous close is the last available close in OHLCV
            ohlcv = ohlcv_map.get(symbol)
            if ohlcv is None or ohlcv.empty:
                logger.warning("No OHLCV available for gated signal {}; skipping", symbol)
                continue

            prev_close = float(ohlcv["close"].iloc[-1])
            entry_low, entry_high = self._compute_entry_zone(symbol, prev_close)
            entry_mid = (entry_low + entry_high) / 2.0

            symbol_tier = "nifty100" if symbol in nifty100_symbols else "midcap100"
            sl, tp = self._compute_levels(symbol, entry_mid, symbol_tier)

            pos_size = compute_position_size(
                score=score,
                total_capital=total_capital,
                n_open_positions=n_open_positions,
            )

            signal_rows.append({
                "dt":             run_date,
                "symbol":         symbol,
                "direction":      "LONG",
                "confidence":     score,
                "entry_low":      round(entry_low, 2),
                "entry_high":     round(entry_high, 2),
                "stop_loss":      round(sl, 2),
                "target":         round(tp, 2),
                "max_hold_days":  BARRIER_MAX_HOLDING_DAYS,
                "position_size":  round(pos_size, 2),
                "model_version":  self._model_version(),
                "sector":         sector_map.get(symbol, "Other"),
                "symbol_tier":    symbol_tier,
                "prev_close":     round(prev_close, 2),
            })

        if not signal_rows:
            logger.info("All gated signals failed level computation — returning empty")
            return pd.DataFrame()

        signals_df = pd.DataFrame(signal_rows)

        # ── Step 9: Persist ────────────────────────────────────────────────────
        self._save(signals_df, run_date)

        logger.info(
            "SignalGenerator complete: {} qualifying signals on {}",
            len(signals_df),
            run_date,
        )
        return signals_df

    # ------------------------------------------------------------------
    # Level computation helpers
    # ------------------------------------------------------------------

    def _compute_entry_zone(self, symbol: str, prev_close: float) -> tuple[float, float]:
        """
        Entry zone: tight band of ±0.5% around yesterday's close.
        The plan fill assumption is next-bar open; this zone guards against
        opening gaps pushing us into a worse fill.
        """
        return prev_close * 0.995, prev_close * 1.005

    def _compute_levels(
        self,
        symbol: str,
        entry_mid: float,
        symbol_tier: str,
    ) -> tuple[float, float]:
        """
        Cost-aware stop-loss and take-profit levels.

        Uses config.cost_model.cost_adjusted_barrier so that quoted targets
        and stops are net-of-cost (i.e. what you actually pocket / lose).
        """
        from config.settings import BARRIER_STOP_LOSS_PCT, BARRIER_TAKE_PROFIT_PCT

        tp = cost_adjusted_barrier(
            entry_mid, BARRIER_TAKE_PROFIT_PCT, "UP", symbol_tier=symbol_tier
        )
        sl = cost_adjusted_barrier(
            entry_mid, BARRIER_STOP_LOSS_PCT, "DOWN", symbol_tier=symbol_tier
        )
        return sl, tp

    # ------------------------------------------------------------------
    # Internal pipeline steps
    # ------------------------------------------------------------------

    def _load_index_data(self, symbol: str) -> pd.DataFrame:
        """Load index / VIX OHLCV from the store. Returns empty DF on failure."""
        try:
            return load_adjusted_ohlcv(symbol)
        except Exception as exc:
            logger.warning("Could not load index data for {}: {}", symbol, exc)
            return pd.DataFrame()

    def _load_model(self):
        """
        Load LightGBM model. Uses self.model_path if set, otherwise the
        most recently modified .txt file in data_store/models/.
        Returns None if no model is found.
        """
        import lightgbm as lgb

        path = self.model_path
        if path is None:
            path = self._latest_model_path()

        if path is None:
            logger.error("No model file found in {}", _MODELS_DIR)
            return None

        try:
            model = lgb.Booster(model_file=str(path))
            logger.info("Loaded model from {}", path)
            self.model_path = path  # cache resolved path
            return model
        except Exception as exc:
            logger.error("Failed to load model from {}: {}", path, exc)
            return None

    def _latest_model_path(self) -> Optional[Path]:
        """Return the most recently modified .txt file under data_store/models/."""
        if not _MODELS_DIR.exists():
            return None
        candidates = sorted(
            _MODELS_DIR.glob("*.txt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None

    def _model_version(self) -> str:
        """Derive a model version string from the model file name."""
        if self.model_path:
            return self.model_path.stem
        return "unknown"

    def _score_universe(
        self,
        ohlcv_map: dict[str, pd.DataFrame],
        index_df: pd.DataFrame,
        vix_df: pd.DataFrame,
        model,
        run_date: date,
    ) -> pd.Series:
        """
        Compute features for every symbol and return a Series of
        {symbol: predicted_probability}.
        """
        symbol_probs: dict[str, float] = {}

        for symbol, ohlcv in ohlcv_map.items():
            try:
                features_df = self.pipeline.compute(ohlcv, index_df, vix_df, symbol)
                if features_df.empty:
                    continue

                # Use the last row (today's features — forward-looking leak-free)
                last_row = features_df.iloc[[-1]]
                feature_cols = [
                    c for c in last_row.columns
                    if c not in {"symbol", "dt", "label", "binary_label", "net_return", "weight"}
                ]
                X = last_row[feature_cols].fillna(0)

                prob = float(model.predict(X)[0])
                symbol_probs[symbol] = prob

            except Exception as exc:
                logger.warning("Scoring failed for {}: {}", symbol, exc)
                continue

        if not symbol_probs:
            return pd.Series(dtype=float)

        predictions = pd.Series(symbol_probs)
        logger.info(
            "Scored {} symbols | prob range [{:.3f}, {:.3f}]",
            len(predictions),
            predictions.min(),
            predictions.max(),
        )
        return predictions

    def _save(self, signals_df: pd.DataFrame, run_date: date) -> None:
        """Persist signals to DuckDB. Idempotent via INSERT OR REPLACE."""
        store_cols = [
            "dt", "symbol", "direction", "confidence",
            "entry_low", "entry_high", "stop_loss", "target",
            "max_hold_days", "position_size", "model_version",
        ]
        # Only pass columns that the store schema knows about
        rows = signals_df[[c for c in store_cols if c in signals_df.columns]].to_dict("records")
        try:
            save_signals(rows)
            logger.info("Saved {} signal records for {} to store", len(rows), run_date)
        except Exception as exc:
            logger.error("Failed to save signals for {}: {}", run_date, exc)

    def _get_nifty100_symbols(self) -> set[str]:
        """Load Nifty 100 symbol set for tier classification."""
        try:
            from data.universe import get_nifty100_symbols
            return get_nifty100_symbols()
        except Exception as exc:
            logger.warning("Could not load Nifty 100 symbols: {}; all treated as midcap100", exc)
            return set()
