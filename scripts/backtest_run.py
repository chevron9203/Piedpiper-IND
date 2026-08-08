"""
One-shot backtest runner. Run this to execute walk-forward backtesting
and save results for viewing in the dashboard's Backtest tab.

Usage:
    python scripts/backtest_run.py
    python scripts/backtest_run.py --start-year 2018 --end-year 2024
    python scripts/backtest_run.py --baseline-only

Steps:
    1. Load full universe and adjusted OHLCV from store
    2. Run feature pipeline for all symbols and dates (if not cached)
    3. Compute triple-barrier labels for all symbols
    4. Run walk-forward training and OOS prediction (WalkForwardTrainer)
    5. Convert OOS predictions → signal_df, run BacktestEngine
    6. Run baseline strategies through the same engine
    7. Print metrics table: real model vs baselines, before and after costs
    8. Save results to data_store/backtest_results.parquet
       and trades to data_store/backtest_trades.parquet
    9. Print deflated Sharpe warning if multiple variants were tested
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from storage import store
from features.pipeline import FeaturePipeline
from models.labeling import compute_labels, compute_short_labels
from models.trainer import WalkForwardTrainer
from models.meta_labeler import MetaLabeler
from backtest.engine import BacktestEngine
from backtest.baseline import buy_and_hold_equity, ma_crossover_signals
from backtest.metrics import compute_all_metrics


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_PATH = Path(settings.DATA_DIR) / "backtest_results.parquet"
TRADES_PATH  = Path(settings.DATA_DIR) / "backtest_trades.parquet"

_BANNER = "=" * 68


def _banner(title: str) -> None:
    print(f"\n{_BANNER}")
    print(f"  {title}")
    print(_BANNER)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_universe() -> pd.DataFrame:
    """Load the screened universe from cache, or abort with a clear message."""
    cache = Path(settings.DATA_DIR) / "universe_cache.parquet"
    if not cache.exists():
        logger.error(
            "Universe cache not found at {}. "
            "Run: python -c \"from data.universe import get_universe; "
            "get_universe(force_refresh=True)\"",
            cache,
        )
        sys.exit(1)
    df = pd.read_parquet(cache)
    logger.info("Universe: {} symbols", len(df))
    return df


def _load_ohlcv(
    symbols: list[str],
    from_date: date,
    to_date: date,
) -> dict[str, pd.DataFrame]:
    """Load adjusted OHLCV from DuckDB for each symbol in the list."""
    ohlcv: dict[str, pd.DataFrame] = {}
    missing = 0
    for sym in symbols:
        try:
            df = store.load_adjusted_ohlcv(sym, from_date=from_date, to_date=to_date)
            if df.empty or len(df) < 60:
                missing += 1
                continue
            ohlcv[sym] = df
        except Exception as exc:
            logger.warning("OHLCV load failed for {}: {}", sym, exc)
            missing += 1

    logger.info(
        "OHLCV loaded: {}/{} symbols ({} skipped — insufficient data)",
        len(ohlcv), len(symbols), missing,
    )
    return ohlcv


def _load_index(symbol: str, from_date: date, to_date: date) -> pd.DataFrame:
    try:
        return store.load_adjusted_ohlcv(symbol, from_date=from_date, to_date=to_date)
    except Exception as exc:
        logger.warning("Could not load index {}: {}", symbol, exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def _compute_features(
    ohlcv_dict: dict[str, pd.DataFrame],
    index_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    version: str = settings.FEATURE_VERSION,
) -> pd.DataFrame:
    """
    Run FeaturePipeline for every symbol. Returns a combined DataFrame
    with columns [symbol, dt, <feature_cols>].
    """
    pipeline = FeaturePipeline(version=version)
    frames: list[pd.DataFrame] = []

    for sym, ohlcv in ohlcv_dict.items():
        try:
            feat = pipeline.compute(ohlcv, index_df, vix_df, sym)
            if feat.empty:
                continue
            feat["symbol"] = sym
            # Preserve DatetimeIndex as "dt" column before concat resets it
            feat = feat.reset_index()
            feat.rename(columns={"index": "dt"}, inplace=True)
            frames.append(feat)
        except Exception as exc:
            logger.warning("Feature computation failed for {}: {}", sym, exc)

    if not frames:
        logger.error("Feature computation produced no output — aborting")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    combined["dt"] = pd.to_datetime(combined["dt"])
    logger.info(
        "Features computed: {} rows, {} symbols, {} feature cols",
        len(combined),
        combined["symbol"].nunique(),
        combined.shape[1] - 2,  # minus symbol, dt
    )
    return combined


# ---------------------------------------------------------------------------
# Label computation
# ---------------------------------------------------------------------------

def _compute_labels(
    ohlcv_dict: dict[str, pd.DataFrame],
    nifty100_symbols: set[str] | None = None,
) -> pd.DataFrame:
    """
    Compute triple-barrier labels for all symbols.
    Returns a DataFrame with [symbol, dt, label, binary_label, net_return, weight].
    """
    _nifty100 = nifty100_symbols or set()
    frames: list[pd.DataFrame] = []

    for sym, ohlcv in ohlcv_dict.items():
        try:
            tier = "nifty100" if sym in _nifty100 else "midcap100"
            label_df = compute_labels(ohlcv, symbol=sym, symbol_tier=tier)
            if label_df.empty:
                continue
            label_df["symbol"] = sym
            # Preserve DatetimeIndex as "dt" column before concat resets it
            label_df = label_df.reset_index()
            label_df.rename(columns={"index": "dt"}, inplace=True)
            frames.append(label_df)
        except Exception as exc:
            logger.warning("Label computation failed for {}: {}", sym, exc)

    if not frames:
        logger.error("Label computation produced no output — aborting")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    combined["dt"] = pd.to_datetime(combined["dt"])
    logger.info(
        "Labels computed: {} rows, {} symbols | positive rate={:.2%}",
        len(combined),
        combined["symbol"].nunique(),
        combined["binary_label"].mean() if "binary_label" in combined.columns else float("nan"),
    )
    return combined


def _compute_short_labels(
    ohlcv_dict: dict[str, pd.DataFrame],
    nifty100_symbols: set[str] | None = None,
) -> pd.DataFrame:
    """
    Compute short triple-barrier labels for all symbols.
    Returns a DataFrame with [symbol, dt, label, binary_label, net_return, weight].
    """
    _nifty100 = nifty100_symbols or set()
    frames: list[pd.DataFrame] = []

    for sym, ohlcv in ohlcv_dict.items():
        try:
            tier = "nifty100" if sym in _nifty100 else "midcap100"
            label_df = compute_short_labels(ohlcv, symbol=sym, symbol_tier=tier)
            if label_df.empty:
                continue
            label_df["symbol"] = sym
            label_df = label_df.reset_index()
            label_df.rename(columns={"index": "dt"}, inplace=True)
            frames.append(label_df)
        except Exception as exc:
            logger.warning("Short label computation failed for {}: {}", sym, exc)

    if not frames:
        logger.warning("Short label computation produced no output — running long-only")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["dt"] = pd.to_datetime(combined["dt"])
    logger.info(
        "Short labels computed: {} rows, {} symbols | short win rate={:.2%}",
        len(combined),
        combined["symbol"].nunique(),
        combined["binary_label"].mean() if "binary_label" in combined.columns else float("nan"),
    )
    return combined


# ---------------------------------------------------------------------------
# Meta-labeler: second-pass walk-forward confidence filter
# ---------------------------------------------------------------------------

_META_FILTER_THRESHOLD = 0.50  # meta P(win|long) must reach this to keep the signal


def _apply_meta_labeler(
    oos: pd.DataFrame,
    feature_df: pd.DataFrame,
    label_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Walk-forward meta-labeler second pass.

    For each fold i, trains MetaLabeler on prior-fold OOS predictions and
    applies it as a BINARY FILTER on fold i:
      - meta_prob >= _META_FILTER_THRESHOLD → keep original primary_prob
      - meta_prob <  _META_FILTER_THRESHOLD → zero out (signal filtered)

    Using meta_prob as a multiplier (combined = primary × meta) is wrong here
    because combined would almost always fall below the CONFIDENCE_THRESHOLD gate,
    leaving near-zero tradeable signals.  The binary approach keeps the primary
    confidence intact while removing low-meta-confidence entries.
    """
    if "split_id" not in oos.columns:
        logger.warning("MetaLabeler: OOS has no split_id column — skipping")
        return oos

    feat_cols = [c for c in feature_df.columns if c not in {"symbol", "dt"}]

    # Join OOS with features and binary label outcome
    merged = (
        oos.reset_index(drop=True)
        .merge(feature_df[["symbol", "dt"] + feat_cols], on=["symbol", "dt"], how="inner")
        .merge(label_df[["symbol", "dt", "binary_label"]], on=["symbol", "dt"], how="left")
        .reset_index(drop=True)
    )
    merged["binary_label"] = merged["binary_label"].fillna(0).astype(float)

    splits = sorted(merged["split_id"].unique())
    if len(splits) < 2:
        logger.warning("MetaLabeler: need ≥2 splits to train — skipping")
        return oos

    logger.info("MetaLabeler: second-pass over {} folds (meta_threshold={:.2f})",
                len(splits), _META_FILTER_THRESHOLD)
    result_frames: list[pd.DataFrame] = []

    for i, split_id in enumerate(splits):
        fold_df = merged[merged["split_id"] == split_id].copy().reset_index(drop=True)

        if i == 0:
            # No prior fold — pass primary_prob through unchanged
            fold_df["combined_prob"] = fold_df["predicted_prob"]
            result_frames.append(fold_df[["symbol", "dt", "combined_prob"]])
            continue

        train_df = merged[merged["split_id"].isin(splits[:i])].reset_index(drop=True)

        X_train = train_df[feat_cols].reset_index(drop=True)
        X_test  = fold_df[feat_cols].reset_index(drop=True)
        p_train = pd.Series(train_df["predicted_prob"].values)
        p_test  = pd.Series(fold_df["predicted_prob"].values)
        y_train = pd.Series(train_df["binary_label"].values)

        meta = MetaLabeler(threshold=settings.CONFIDENCE_THRESHOLD)
        try:
            meta.fit(
                feature_df=X_train,
                primary_predictions=p_train,
                actual_outcomes=y_train,
            )
            if meta.model is None:
                fold_df["combined_prob"] = fold_df["predicted_prob"]
            else:
                # Binary filter: meta_prob >= _META_FILTER_THRESHOLD keeps primary_prob
                cand_mask = p_test.values >= settings.CONFIDENCE_THRESHOLD
                result_probs = fold_df["predicted_prob"].values.copy()

                if cand_mask.any():
                    X_cand = X_test.loc[cand_mask].copy()
                    X_cand["primary_prob"] = p_test.values[cand_mask]
                    for mc in (meta.feature_cols or []):
                        if mc not in X_cand.columns:
                            X_cand[mc] = 0.0
                    X_aligned = (
                        X_cand[meta.feature_cols].values.astype(np.float32)
                    )
                    meta_probs = meta.model.predict(X_aligned)
                    cand_indices = np.where(cand_mask)[0]
                    n_blocked = 0
                    for j, cidx in enumerate(cand_indices):
                        if meta_probs[j] < _META_FILTER_THRESHOLD:
                            result_probs[cidx] = 0.0
                            n_blocked += 1

                    n_prim = int(cand_mask.sum())
                    n_pass = n_prim - n_blocked
                    logger.info(
                        "  fold {}/{} ({}): {} of {} primary signals pass meta filter",
                        i + 1, len(splits), split_id, n_pass, n_prim,
                    )

                fold_df["combined_prob"] = result_probs
        except Exception as exc:
            logger.warning("MetaLabeler fold {} failed: {} — using primary prob", split_id, exc)
            fold_df["combined_prob"] = fold_df["predicted_prob"]

        result_frames.append(fold_df[["symbol", "dt", "combined_prob"]])

    prob_updates = pd.concat(result_frames, ignore_index=True)

    # Merge combined_prob back into the original OOS index order
    result = oos.merge(prob_updates, on=["symbol", "dt"], how="left")
    result["predicted_prob"] = result["combined_prob"].fillna(result["predicted_prob"])
    result = result.drop(columns=["combined_prob"])

    n_new_filtered = int(
        ((result["predicted_prob"] < settings.CONFIDENCE_THRESHOLD) &
         (oos["predicted_prob"] >= settings.CONFIDENCE_THRESHOLD)).sum()
    )
    logger.info("MetaLabeler total: {} signals additionally filtered by meta", n_new_filtered)
    return result


# ---------------------------------------------------------------------------
# Walk-forward → BacktestEngine pipeline
# ---------------------------------------------------------------------------

def _run_model_backtest(
    feature_df: pd.DataFrame,
    label_df: pd.DataFrame,
    short_label_df: pd.DataFrame,
    ohlcv_dict: dict[str, pd.DataFrame],
    sector_map: dict[str, str],
    symbol_tiers: dict[str, str],
    vix_df: pd.DataFrame | None = None,
    start_capital: float = settings.STARTING_VIRTUAL_CAPITAL,
) -> dict:
    """
    Train walk-forward long and short models, build combined signal_df
    (positive = long confidence, negative = short confidence), run BacktestEngine.
    """
    threshold = settings.CONFIDENCE_THRESHOLD

    # ── Long model ─────────────────────────────────────────────────────────────
    trainer = WalkForwardTrainer()
    oos_long = trainer.run_walk_forward(feature_df, label_df)

    if oos_long.empty:
        logger.error("Long walk-forward produced no OOS predictions — check data coverage")
        return {}

    logger.info(
        "Long OOS predictions: {} rows across {} folds | prob range [{:.3f}, {:.3f}]",
        len(oos_long),
        oos_long["split_id"].nunique() if "split_id" in oos_long.columns else 0,
        oos_long["predicted_prob"].min(),
        oos_long["predicted_prob"].max(),
    )

    # Second-pass meta-labeler filter — disabled until sufficient per-fold data
    # oos_long = _apply_meta_labeler(oos_long, feature_df, label_df)

    long_signal_df = oos_long.pivot_table(
        index="dt", columns="symbol", values="predicted_prob", aggfunc="last"
    ).fillna(0.0)
    long_signal_df.index = pd.to_datetime(long_signal_df.index)
    long_signal_df = long_signal_df.sort_index()
    long_signal_df[long_signal_df < threshold] = 0.0

    # ── Short model ────────────────────────────────────────────────────────────
    short_signal_df = pd.DataFrame()
    if not short_label_df.empty:
        short_trainer = WalkForwardTrainer()
        try:
            oos_short = short_trainer.run_walk_forward(feature_df, short_label_df)
            if not oos_short.empty:
                logger.info(
                    "Short OOS predictions: {} rows across {} folds | prob range [{:.3f}, {:.3f}]",
                    len(oos_short),
                    oos_short["split_id"].nunique() if "split_id" in oos_short.columns else 0,
                    oos_short["predicted_prob"].min(),
                    oos_short["predicted_prob"].max(),
                )
                short_signal_df = oos_short.pivot_table(
                    index="dt", columns="symbol", values="predicted_prob", aggfunc="last"
                ).fillna(0.0)
                short_signal_df.index = pd.to_datetime(short_signal_df.index)
                short_signal_df = short_signal_df.sort_index()
                short_signal_df[short_signal_df < threshold] = 0.0
        except Exception as exc:
            logger.warning("Short walk-forward failed: {} — running long-only", exc)

    # ── Merge long + short into combined signal_df ─────────────────────────────
    # Positive values = long confidence, negative = short confidence.
    # Where only one side fires: use it. Where both fire for same symbol/date:
    # use the higher-confidence direction (long gets priority on a tie).
    if not short_signal_df.empty:
        all_cols = sorted(set(long_signal_df.columns) | set(short_signal_df.columns))
        all_idx  = long_signal_df.index.union(short_signal_df.index)
        L = long_signal_df.reindex( index=all_idx, columns=all_cols, fill_value=0.0)
        S = short_signal_df.reindex(index=all_idx, columns=all_cols, fill_value=0.0)

        signal_df = L.copy()
        only_short = (L == 0) & (S > 0)          # short fires, long does not
        signal_df[only_short] = -S[only_short]
        short_wins = (S > L) & (L > 0) & (S > 0) # both fire, short has higher confidence
        signal_df[short_wins] = -S[short_wins]

        n_long  = int((signal_df > 0).sum().sum())
        n_short = int((signal_df < 0).sum().sum())
        logger.info("Combined signal_df: {} long cells, {} short cells across {} dates",
                    n_long, n_short, len(signal_df))
    else:
        signal_df = long_signal_df
        logger.info("Running long-only (no short model output)")

    # ── BacktestEngine ─────────────────────────────────────────────────────────
    engine = BacktestEngine(start_capital=start_capital)
    result = engine.run(
        signal_df=signal_df,
        ohlcv_dict=ohlcv_dict,
        sector_map=sector_map,
        tp_pct=settings.BARRIER_TAKE_PROFIT_PCT,
        sl_pct=settings.BARRIER_STOP_LOSS_PCT,
        symbol_tiers=symbol_tiers,
        vix_df=vix_df,
    )

    # Attach split_id from the long OOS (same fold dates apply to both models)
    if "split_id" in oos_long.columns and not result.get("trades", pd.DataFrame()).empty:
        oos_splits = (
            oos_long[["symbol", "dt", "split_id"]]
            .rename(columns={"dt": "signal_date"})
        )
        oos_splits["signal_date"] = pd.to_datetime(oos_splits["signal_date"])
        trades = result["trades"].copy()
        trades["signal_date"] = pd.to_datetime(trades.get("signal_date"), errors="coerce")
        merged = trades.merge(oos_splits, on=["symbol", "signal_date"], how="left")
        result["trades"] = merged

    return result


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def _run_baselines(
    ohlcv_dict: dict[str, pd.DataFrame],
    nifty_ohlcv: pd.DataFrame,
    sector_map: dict[str, str],
    symbol_tiers: dict[str, str],
    start_capital: float = settings.STARTING_VIRTUAL_CAPITAL,
) -> dict[str, dict]:
    """
    Run buy-and-hold (Nifty index) and MA crossover through BacktestEngine.
    Returns {'buy_and_hold': result, 'ma_crossover': result}.
    """
    baselines: dict[str, dict] = {}

    # Buy-and-hold on Nifty 50
    if not nifty_ohlcv.empty:
        bah_equity = buy_and_hold_equity(nifty_ohlcv, start_capital=start_capital)
        bah_returns = bah_equity.pct_change().dropna()
        bah_trades = pd.DataFrame()  # no individual trades for buy-and-hold
        baselines["buy_and_hold"] = {
            "equity_curve":  bah_equity,
            "trades":        bah_trades,
            "metrics_gross": compute_all_metrics(bah_equity, bah_trades, label="bah_gross"),
            "metrics_net":   compute_all_metrics(bah_equity, bah_trades, label="bah_net"),
        }
    else:
        logger.warning("Nifty 50 OHLCV not available — skipping buy-and-hold baseline")

    # MA crossover
    ma_signals = ma_crossover_signals(ohlcv_dict, fast=20, slow=50)
    if not ma_signals.empty:
        engine = BacktestEngine(start_capital=start_capital)
        ma_result = engine.run(
            signal_df=ma_signals,
            ohlcv_dict=ohlcv_dict,
            sector_map=sector_map,
            tp_pct=settings.BARRIER_TAKE_PROFIT_PCT,
            sl_pct=settings.BARRIER_STOP_LOSS_PCT,
            symbol_tiers=symbol_tiers,
        )
        baselines["ma_crossover"] = ma_result

    return baselines


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------

def _save_results(
    model_result: dict,
    baselines: dict[str, dict],
) -> None:
    """
    Save metrics to backtest_results.parquet and trades to backtest_trades.parquet.

    Metrics format: rows of (label, metric, value) — easy to pivot in the dashboard.
    """
    rows: list[dict] = []

    def _add(label: str, metrics: dict) -> None:
        for metric, value in metrics.items():
            rows.append({"label": label, "metric": metric, "value": float(value) if value is not None else None})

    if model_result:
        _add("gross", model_result.get("metrics_gross", {}))
        _add("net",   model_result.get("metrics_net",   {}))

    for name, result in baselines.items():
        _add(f"{name}_gross", result.get("metrics_gross", {}))
        _add(f"{name}_net",   result.get("metrics_net",   {}))

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_parquet(RESULTS_PATH, index=False)
    logger.info("Metrics saved to {}", RESULTS_PATH)

    # Trades
    trades = model_result.get("trades", pd.DataFrame()) if model_result else pd.DataFrame()
    if not trades.empty:
        # Drop non-serialisable column if present
        if "cost_breakdown" in trades.columns:
            trades = trades.drop(columns=["cost_breakdown"])
        trades.to_parquet(TRADES_PATH, index=False)
        logger.info("Trades saved to {}", TRADES_PATH)


# ---------------------------------------------------------------------------
# Metrics printing
# ---------------------------------------------------------------------------

def _fmt(v, pct: bool = False) -> str:
    try:
        f = float(v)
        return f"{f:.1%}" if pct else f"{f:.2f}"
    except (TypeError, ValueError):
        return "—"


def _print_metrics_table(
    model_result: dict,
    baselines: dict[str, dict],
) -> None:
    """Pretty-print a comparison table to stdout."""
    _banner("BACKTEST RESULTS — WALK-FORWARD OOS")

    labels = []
    gross_metrics = []
    net_metrics = []

    if model_result:
        labels.append("ML Model")
        gross_metrics.append(model_result.get("metrics_gross", {}))
        net_metrics.append(model_result.get("metrics_net", {}))

    for name, result in baselines.items():
        labels.append(name.replace("_", " ").title())
        gross_metrics.append(result.get("metrics_gross", {}))
        net_metrics.append(result.get("metrics_net", {}))

    col_w = 16
    header = f"{'Metric':<22}" + "".join(f"{lbl:>{col_w}}" for lbl in labels)
    print(f"\n{'Before Costs (Gross)'}")
    print(header)
    print("-" * len(header))

    metrics_order = [
        ("sharpe",        "Sharpe Ratio",   False),
        ("sortino",       "Sortino Ratio",  False),
        ("max_drawdown",  "Max Drawdown",   True),
        ("cagr",          "CAGR",           True),
        ("hit_rate",      "Hit Rate",       True),
        ("win_loss_ratio","W/L Ratio",      False),
        ("n_trades",      "# Trades",       False),
        ("total_gross_pnl","Total P&L",     False),
    ]

    for key, label, is_pct in metrics_order:
        row = f"{label:<22}"
        for m in gross_metrics:
            val = m.get(key)
            row += f"{_fmt(val, is_pct):>{col_w}}"
        print(row)

    print(f"\n{'After Costs (Net)'}")
    print(header)
    print("-" * len(header))

    net_order = [
        ("sharpe",        "Sharpe Ratio",   False),
        ("sortino",       "Sortino Ratio",  False),
        ("max_drawdown",  "Max Drawdown",   True),
        ("cagr",          "CAGR",           True),
        ("hit_rate",      "Hit Rate",       True),
        ("win_loss_ratio","W/L Ratio",      False),
        ("n_trades",      "# Trades",       False),
        ("total_net_pnl", "Total P&L",      False),
    ]

    for key, label, is_pct in net_order:
        row = f"{label:<22}"
        for m in net_metrics:
            val = m.get(key)
            row += f"{_fmt(val, is_pct):>{col_w}}"
        print(row)

    print()


def _print_deflated_sharpe_warning(multiple_variants: bool) -> None:
    if not multiple_variants:
        return
    _banner("DEFLATED SHARPE WARNING")
    print(
        "\n  Multiple model variants or parameter sets were tested before arriving at\n"
        "  this configuration. The reported Sharpe ratio is optimistically biased.\n\n"
        "  Per Bailey & Lopez de Prado (2014), the Deflated Sharpe Ratio (DSR) should\n"
        "  be computed to correct for multiple-testing. Treat the gross Sharpe as an\n"
        "  upper bound, not an expectation.\n\n"
        "  Rule of thumb: with N independent trials, the expected max Sharpe of a\n"
        "  random strategy scales as sqrt(2 * log(N)), so a reported Sharpe of 1.5\n"
        "  with 20 variants needs a DSR correction of ~0.6 Sharpe units.\n"
    )


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk-forward backtest runner for the NSE trading signal system."
    )
    parser.add_argument(
        "--start-year", type=int, default=None,
        help="Start year for the backtest (default: all available data)",
    )
    parser.add_argument(
        "--end-year", type=int, default=None,
        help="End year for the backtest (default: today)",
    )
    parser.add_argument(
        "--baseline-only", action="store_true",
        help="Run only baseline strategies (no ML model training)",
    )
    parser.add_argument(
        "--multiple-variants", action="store_true",
        help="Print deflated Sharpe warning (use when you tested multiple configs)",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Do not write results to parquet files (dry run)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    _banner("PiedPiper — Walk-Forward Backtest Runner")
    print(f"  Start year : {args.start_year or 'all available'}")
    print(f"  End year   : {args.end_year or date.today().year}")
    print(f"  Mode       : {'baseline only' if args.baseline_only else 'ML model + baselines'}")
    print()

    # ── Ensure DB schema is ready ──────────────────────────────────────────
    store.init_schema()

    # ── 1. Load universe ───────────────────────────────────────────────────
    universe = _load_universe()
    symbols: list[str] = universe["symbol"].tolist()
    sector_map: dict[str, str] = (
        universe.set_index("symbol")["sector"].to_dict()
        if "sector" in universe.columns
        else {}
    )

    # Nifty 100 symbols for slippage tier
    nifty100_path = Path(settings.DATA_DIR) / "nifty100_symbols.txt"
    nifty100_symbols: set[str] = set()
    if nifty100_path.exists():
        nifty100_symbols = set(nifty100_path.read_text().splitlines())
    symbol_tiers: dict[str, str] = {
        sym: ("nifty100" if sym in nifty100_symbols else "midcap100")
        for sym in symbols
    }

    # Date range
    from_date = date(args.start_year, 1, 1) if args.start_year else date(2015, 1, 1)
    to_date   = date(args.end_year, 12, 31) if args.end_year else date.today()

    # ── 2. Load OHLCV ──────────────────────────────────────────────────────
    _banner("Step 1/5 — Loading OHLCV")
    ohlcv_dict = _load_ohlcv(symbols, from_date, to_date)

    if not ohlcv_dict:
        logger.error("No OHLCV data available — cannot run backtest. Populate the store first.")
        sys.exit(1)

    nifty_ohlcv = _load_index(settings.NIFTY50_SYMBOL, from_date, to_date)
    vix_ohlcv   = _load_index(settings.INDIA_VIX_SYMBOL, from_date, to_date)

    model_result: dict = {}
    baselines: dict[str, dict] = {}

    if not args.baseline_only:
        # ── 3. Feature pipeline ────────────────────────────────────────────
        _banner("Step 2/5 — Computing Features")
        feature_df = _compute_features(ohlcv_dict, nifty_ohlcv, vix_ohlcv)

        # ── 4. Triple-barrier labels ───────────────────────────────────────
        _banner("Step 3/5 — Computing Labels (Triple Barrier)")
        label_df = _compute_labels(ohlcv_dict, nifty100_symbols=nifty100_symbols)

        # Short model disabled: NSE equity cash segment does not allow overnight
        # short positions (must square off intraday). Positional shorts require
        # an F&O account (stock futures / index futures). Keep long-only for now.
        short_label_df: pd.DataFrame = pd.DataFrame()

        # ── 5. Walk-forward training + OOS backtest ────────────────────────
        _banner("Step 4/5 — Walk-Forward Training & OOS Backtest")
        model_result = _run_model_backtest(
            feature_df     = feature_df,
            label_df       = label_df,
            short_label_df = short_label_df,   # empty → long-only run
            ohlcv_dict     = ohlcv_dict,
            sector_map     = sector_map,
            symbol_tiers   = symbol_tiers,
            # vix_df: disabled — VIX > 20 blocks COVID recovery which is the
            # strategy's best period; LightGBM already uses VIX as a feature.
        )

    # ── 6. Baselines ───────────────────────────────────────────────────────
    _banner("Step 5/5 — Running Baselines")
    baselines = _run_baselines(ohlcv_dict, nifty_ohlcv, sector_map, symbol_tiers)

    # ── 7. Print results table ─────────────────────────────────────────────
    _print_metrics_table(model_result, baselines)

    # ── 8. Persist ─────────────────────────────────────────────────────────
    if not args.no_save:
        _banner("Saving Results")
        _save_results(model_result, baselines)
        print(f"  Metrics  → {RESULTS_PATH}")
        print(f"  Trades   → {TRADES_PATH}")
        print(f"\n  View in dashboard: http://127.0.0.1:5001/backtest\n")
    else:
        print("  --no-save flag set: results not written to disk.\n")

    # ── 9. Deflated Sharpe warning ─────────────────────────────────────────
    _print_deflated_sharpe_warning(args.multiple_variants)

    _banner("Done")
    print()


if __name__ == "__main__":
    main()
