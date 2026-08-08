"""
Walk-forward trainer for the NSE cross-sectional LightGBM ranking model.

CV scheme:
  - Expanding training window (growing from MIN_TRAIN_WINDOW_YEARS back).
  - Purged: training samples whose label window overlaps the test period are removed.
    A label at date t uses forward data up to t + BARRIER_MAX_HOLDING_DAYS,
    so samples where  label_date + max_hold_days >= test_start  are purged.
  - Embargo: test_start = train_end + EMBARGO_DAYS (trading-day count, not calendar).
  - Retrain cadence: every RETRAIN_CADENCE_TRADING_DAYS trading days.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import lightgbm as lgb
from loguru import logger

from config.settings import (
    BARRIER_MAX_HOLDING_DAYS,
    EMBARGO_DAYS,
    MIN_TRAIN_WINDOW_YEARS,
    RETRAIN_CADENCE_TRADING_DAYS,
)

if TYPE_CHECKING:
    pass  # lgb.Booster imported at runtime via lightgbm

# ---------------------------------------------------------------------------
# LightGBM hyper-parameters (locked per plan)
# ---------------------------------------------------------------------------
LGB_PARAMS: dict = {
    "objective": "binary",
    "metric": "auc",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 5,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
}

# Columns that must never be fed as model features
_NON_FEATURE_COLS = {"symbol", "dt", "label", "binary_label", "net_return", "weight"}


class WalkForwardTrainer:
    """
    Expanding-window walk-forward cross-validation trainer.

    Parameters
    ----------
    retrain_cadence : int
        Number of trading days between retraining events (test fold length).
    embargo_days : int
        Gap in trading days between end of training data and start of test fold.
    min_train_days : int
        Minimum number of trading days required before the first model is fitted.
    max_hold_days : int
        Triple-barrier max holding period — used to determine purge boundary.
    lgb_params : dict | None
        LightGBM parameters. Defaults to LGB_PARAMS.
    """

    def __init__(
        self,
        retrain_cadence: int = RETRAIN_CADENCE_TRADING_DAYS,
        embargo_days: int = EMBARGO_DAYS,
        min_train_days: int = MIN_TRAIN_WINDOW_YEARS * 252,
        max_hold_days: int = BARRIER_MAX_HOLDING_DAYS,
        lgb_params: dict | None = None,
    ) -> None:
        self.retrain_cadence = retrain_cadence
        self.embargo_days = embargo_days
        self.min_train_days = min_train_days
        self.max_hold_days = max_hold_days
        self.lgb_params = lgb_params or LGB_PARAMS.copy()

    # ------------------------------------------------------------------
    # Split generation
    # ------------------------------------------------------------------

    def get_splits(self, dates: pd.DatetimeIndex) -> list[dict]:
        """
        Generate expanding-window train/test splits over a sorted DatetimeIndex.

        Each split dict has:
          train_start  – first available date in training window
          train_end    – last date included in training data
          purge_start  – first label date excluded due to purge (= test_start - max_hold_days)
          test_start   – first date of out-of-sample test fold (train_end + embargo_days)
          test_end     – last date of test fold (test_start + retrain_cadence - 1)
          split_id     – integer index (0-based)

        Purge rule: training samples at label_date t are removed when
            t + max_hold_days  >=  test_start
        i.e. the label overlaps into the embargo or test period.
        """
        dates = dates.sort_values().unique()
        n = len(dates)

        splits = []
        split_id = 0

        # First possible test start: after min_train_days + embargo
        first_test_start_idx = self.min_train_days + self.embargo_days

        if first_test_start_idx >= n:
            logger.warning(
                f"Not enough data for even one split. "
                f"Need >{first_test_start_idx} trading days, got {n}."
            )
            return splits

        # Slide the test window in steps of retrain_cadence
        test_start_idx = first_test_start_idx
        while test_start_idx < n:
            test_end_idx = min(test_start_idx + self.retrain_cadence - 1, n - 1)

            # train_end is embargo_days before test_start
            train_end_idx = test_start_idx - self.embargo_days - 1

            if train_end_idx < 0:
                test_start_idx += self.retrain_cadence
                continue

            train_window_size = train_end_idx + 1  # all dates up to train_end
            if train_window_size < self.min_train_days:
                test_start_idx += self.retrain_cadence
                continue

            # Purge boundary: remove training samples where label window overlaps test
            # label_date + max_hold_days >= test_start  =>  label_date >= test_start - max_hold_days
            purge_start_idx = test_start_idx - self.max_hold_days
            purge_start_idx = max(purge_start_idx, 0)

            splits.append(
                {
                    "split_id": split_id,
                    "train_start": dates[0],
                    "train_end": dates[train_end_idx],
                    "purge_start": dates[purge_start_idx],
                    "test_start": dates[test_start_idx],
                    "test_end": dates[test_end_idx],
                }
            )
            split_id += 1
            test_start_idx += self.retrain_cadence

        logger.info(f"Generated {len(splits)} walk-forward splits.")
        return splits

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        feature_df: pd.DataFrame,
        label_df: pd.DataFrame,
        split: dict,
    ) -> lgb.Booster:
        """
        Fit a LightGBM model on the training fold of one walk-forward split.

        Parameters
        ----------
        feature_df : pd.DataFrame
            Multi-symbol feature matrix. Must contain columns 'symbol' and 'dt'.
            All other numeric columns are used as features.
        label_df : pd.DataFrame
            Must contain columns: symbol, dt, binary_label (or label), net_return, weight.
        split : dict
            One element from get_splits().

        Returns
        -------
        lgb.Booster
            Trained booster.
        """
        # --- Merge features and labels on (symbol, dt) ---
        merged = self._merge_feature_label(feature_df, label_df)

        # --- Apply training window + purge ---
        in_train = (merged["dt"] >= split["train_start"]) & (
            merged["dt"] <= split["train_end"]
        )
        # Purge: remove samples whose label window reaches into test period
        # label at date t is purged if t >= purge_start
        not_purged = merged["dt"] < split["purge_start"]

        train_mask = in_train & not_purged
        train_data = merged[train_mask].copy()

        if train_data.empty:
            raise ValueError(
                f"Split {split['split_id']}: no training samples after purge. "
                f"train window: {split['train_start']} – {split['train_end']}, "
                f"purge_start: {split['purge_start']}"
            )

        X_train, y_train, w_train = self._extract_Xyw(train_data)

        logger.info(
            f"Split {split['split_id']} | train={len(X_train)} samples | "
            f"features={X_train.shape[1]} | "
            f"pos_rate={y_train.mean():.2%} | "
            f"date range: {split['train_start'].date()} – {split['train_end'].date()}"
        )

        dataset = lgb.Dataset(X_train, label=y_train, weight=w_train, free_raw_data=False)

        # Separate fit-time params from constructor params
        fit_params = {k: v for k, v in self.lgb_params.items() if k != "n_estimators"}
        n_estimators = self.lgb_params.get("n_estimators", 300)

        booster = lgb.train(
            params=fit_params,
            train_set=dataset,
            num_boost_round=n_estimators,
        )

        return booster

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self, model: lgb.Booster, feature_df: pd.DataFrame
    ) -> pd.Series:
        """
        Score all symbols in feature_df for a given date (or date range).

        Parameters
        ----------
        model : lgb.Booster
        feature_df : pd.DataFrame
            Feature matrix with 'symbol' and 'dt' columns.

        Returns
        -------
        pd.Series
            Predicted probability indexed by symbol. Multi-date inputs return a
            MultiIndex (dt, symbol).
        """
        feat_cols = self._feature_cols(feature_df)
        X = feature_df[feat_cols].values
        probs = model.predict(X)

        if "dt" in feature_df.columns and feature_df["dt"].nunique() > 1:
            index = pd.MultiIndex.from_arrays(
                [feature_df["dt"].values, feature_df["symbol"].values],
                names=["dt", "symbol"],
            )
        else:
            index = pd.Index(feature_df["symbol"].values, name="symbol")

        return pd.Series(probs, index=index, name="predicted_prob")

    # ------------------------------------------------------------------
    # Full walk-forward run
    # ------------------------------------------------------------------

    def run_walk_forward(
        self,
        feature_df: pd.DataFrame,
        label_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Run all walk-forward splits and collect out-of-sample predictions.

        Returns
        -------
        pd.DataFrame
            Columns: symbol, dt, predicted_prob, actual_label, split_id.
            Only test-fold rows are included (no training-set predictions).
        """
        all_dates = pd.DatetimeIndex(
            sorted(
                set(feature_df["dt"].unique()) | set(label_df["dt"].unique())
            )
        )
        splits = self.get_splits(all_dates)

        if not splits:
            logger.error("No valid splits — cannot run walk-forward.")
            return pd.DataFrame(
                columns=["symbol", "dt", "predicted_prob", "actual_label", "split_id"]
            )

        merged = self._merge_feature_label(feature_df, label_df)
        oos_records: list[pd.DataFrame] = []

        for split in splits:
            logger.info(
                f"Running split {split['split_id']} | "
                f"test: {split['test_start'].date()} – {split['test_end'].date()}"
            )
            try:
                booster = self.train(feature_df, label_df, split)
            except ValueError as exc:
                logger.warning(f"Split {split['split_id']} skipped: {exc}")
                continue

            # --- Test fold ---
            test_mask = (merged["dt"] >= split["test_start"]) & (
                merged["dt"] <= split["test_end"]
            )
            test_data = merged[test_mask].copy()

            if test_data.empty:
                logger.warning(f"Split {split['split_id']}: empty test fold, skipping.")
                continue

            feat_cols = self._feature_cols(test_data)
            probs = booster.predict(test_data[feat_cols].values)

            target_col = "binary_label" if "binary_label" in test_data.columns else "label"

            oos = pd.DataFrame(
                {
                    "symbol": test_data["symbol"].values,
                    "dt": test_data["dt"].values,
                    "predicted_prob": probs,
                    "actual_label": test_data[target_col].values,
                    "split_id": split["split_id"],
                }
            )
            oos_records.append(oos)

        if not oos_records:
            logger.error("Walk-forward produced no out-of-sample predictions.")
            return pd.DataFrame(
                columns=["symbol", "dt", "predicted_prob", "actual_label", "split_id"]
            )

        result = pd.concat(oos_records, ignore_index=True)
        result["dt"] = pd.to_datetime(result["dt"])
        result = result.sort_values(["dt", "symbol"]).reset_index(drop=True)

        logger.info(
            f"Walk-forward complete: {len(result)} OOS predictions across "
            f"{result['split_id'].nunique()} splits."
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _feature_cols(self, df: pd.DataFrame) -> list[str]:
        """Return the model feature columns (all numeric cols except meta-cols)."""
        return [
            c for c in df.columns
            if c not in _NON_FEATURE_COLS
            and pd.api.types.is_numeric_dtype(df[c])
        ]

    def _merge_feature_label(
        self,
        feature_df: pd.DataFrame,
        label_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Inner-join features and labels on (symbol, dt).
        Adds a 'weight' column (uniform = 1.0 if not already present in label_df).
        """
        # Ensure dt is datetime in both
        feature_df = feature_df.copy()
        label_df = label_df.copy()
        feature_df["dt"] = pd.to_datetime(feature_df["dt"])
        label_df["dt"] = pd.to_datetime(label_df["dt"])

        if "weight" not in label_df.columns:
            label_df["weight"] = 1.0

        merged = feature_df.merge(
            label_df[["symbol", "dt", "label", "binary_label", "net_return", "weight"]
                      if "binary_label" in label_df.columns
                      else ["symbol", "dt", "label", "net_return", "weight"]],
            on=["symbol", "dt"],
            how="inner",
        )

        if merged.empty:
            raise ValueError(
                "Feature/label merge produced an empty DataFrame. "
                "Check that 'symbol' and 'dt' columns are aligned."
            )

        return merged

    @staticmethod
    def _extract_Xyw(
        df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract feature matrix X, target y, and sample weights w from merged df."""
        feat_cols = [
            c for c in df.columns
            if c not in _NON_FEATURE_COLS
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        X = df[feat_cols].values.astype(np.float32)

        target_col = "binary_label" if "binary_label" in df.columns else "label"
        y = df[target_col].values.astype(np.float32)

        w = df["weight"].values.astype(np.float32) if "weight" in df.columns else np.ones(len(df), dtype=np.float32)

        return X, y, w
