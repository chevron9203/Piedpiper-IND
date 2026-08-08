import pandas as pd
import numpy as np
from loguru import logger

from features.pipeline import FeaturePipeline


def run_leakage_tests(
    pipeline: FeaturePipeline,
    stock_df: pd.DataFrame,
    index_df: pd.DataFrame,
    vix_df: pd.DataFrame,
) -> dict[str, str]:
    full_features = pipeline.compute(stock_df, index_df, vix_df, symbol="_leakage_test_full")
    full_features = full_features.drop(columns=["symbol"], errors="ignore")

    truncated_features = pipeline.compute(
        stock_df.iloc[:-1],
        index_df,
        vix_df,
        symbol="_leakage_test_trunc",
    )
    truncated_features = truncated_features.drop(columns=["symbol"], errors="ignore")

    overlapping_dates = full_features.index.intersection(truncated_features.index)

    results: dict[str, str] = {}
    feature_cols = full_features.columns.tolist()

    for col in feature_cols:
        if col not in truncated_features.columns:
            results[col] = "ok"
            continue

        full_vals = full_features.loc[overlapping_dates, col]
        trunc_vals = truncated_features.loc[overlapping_dates, col]

        both_nan = full_vals.isna() & trunc_vals.isna()
        # Where neither is NaN, values must match within floating-point tolerance
        comparable = ~full_vals.isna() & ~trunc_vals.isna()

        if comparable.any():
            diff = (full_vals[comparable] - trunc_vals[comparable]).abs()
            # Allow for tiny floating-point rounding but not meaningful divergence
            if (diff > 1e-8).any():
                results[col] = "LOOKAHEAD_SUSPECTED"
                continue

        results[col] = "ok"

    n_total = len(results)
    n_ok = sum(1 for v in results.values() if v == "ok")
    n_leaky = n_total - n_ok

    logger.info(f"Leakage test summary: {n_ok}/{n_total} features clean, {n_leaky} suspected leakage")

    if n_leaky > 0:
        leaky = [k for k, v in results.items() if v == "LOOKAHEAD_SUSPECTED"]
        logger.warning(f"Suspected lookahead features: {leaky}")

    return results


def assert_no_leakage(
    pipeline: FeaturePipeline,
    stock_df: pd.DataFrame,
    index_df: pd.DataFrame,
    vix_df: pd.DataFrame,
) -> None:
    results = run_leakage_tests(pipeline, stock_df, index_df, vix_df)
    leaky = [k for k, v in results.items() if v == "LOOKAHEAD_SUSPECTED"]
    if leaky:
        raise ValueError(
            f"Lookahead bias detected in {len(leaky)} feature(s): {leaky}"
        )
    logger.success("No lookahead bias detected in any feature.")
