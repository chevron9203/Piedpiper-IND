"""
Meta-labeler: secondary LightGBM model that filters primary model signals.

Architecture
------------
Primary model  →  P(long)  →  if P >= threshold: candidate signal
Meta-labeler   →  P(win | primary said long)  →  combined confidence

Combined confidence = primary_prob * meta_prob.

Training set: only samples where the primary model predicted "long" (i.e., where
primary_prob >= CONFIDENCE_THRESHOLD). The meta target is 1 if the trade was
net-profitable, 0 otherwise.

This isolates the meta-labeler from the base-rate calibration of the primary model
and focuses it purely on "given we plan to enter, will this trade make money?"
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from loguru import logger

from config.settings import CONFIDENCE_THRESHOLD

# ---------------------------------------------------------------------------
# Meta-labeler LightGBM parameters (slightly more regularised than primary —
# smaller training set, higher risk of overfit).
# ---------------------------------------------------------------------------
META_LGB_PARAMS: dict = {
    "objective": "binary",
    "metric": "auc",
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 4,
    "num_leaves": 15,
    "min_child_samples": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.2,
    "reg_lambda": 0.2,
    "verbose": -1,
}

# Columns that are never model features
_META_NON_FEATURE_COLS = {
    "symbol", "dt", "label", "binary_label",
    "net_return", "weight", "primary_prob",
}


class MetaLabeler:
    """
    Secondary model: given primary model's signal, predict whether to act.

    Attributes
    ----------
    model : lgb.Booster | None
        Trained LightGBM booster (None until fit() is called).
    threshold : float
        Primary model confidence threshold; only predictions above this
        are candidates for the meta-labeler.
    feature_cols : list[str] | None
        Feature column names captured at fit time (used to align predict input).
    """

    def __init__(
        self,
        threshold: float = CONFIDENCE_THRESHOLD,
        lgb_params: dict | None = None,
    ) -> None:
        self.model: lgb.Booster | None = None
        self.threshold = threshold
        self.lgb_params = lgb_params or META_LGB_PARAMS.copy()
        self.feature_cols: list[str] | None = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        feature_df: pd.DataFrame,
        primary_predictions: pd.Series,
        actual_outcomes: pd.Series,
    ) -> None:
        """
        Train the meta-labeler on samples where the primary model said "long".

        Parameters
        ----------
        feature_df : pd.DataFrame
            Feature matrix for all candidate samples (same index as
            primary_predictions and actual_outcomes).
        primary_predictions : pd.Series
            Primary model's predicted probabilities, indexed identically to
            feature_df (by integer position or a shared index).
        actual_outcomes : pd.Series
            Binary outcome: 1 if trade was net-profitable, 0 otherwise.
            Must share the same index as primary_predictions.
        """
        # Align all inputs on the same index
        idx = feature_df.index
        preds = primary_predictions.reindex(idx)
        outcomes = actual_outcomes.reindex(idx)

        # Filter: only rows where primary model predicted "long"
        candidate_mask = preds >= self.threshold
        n_candidates = candidate_mask.sum()

        if n_candidates == 0:
            logger.warning(
                f"MetaLabeler.fit: no samples with primary_prob >= {self.threshold}. "
                "Meta-labeler not trained."
            )
            return

        X_meta = feature_df.loc[candidate_mask].copy()
        y_meta = outcomes.loc[candidate_mask]
        p_meta = preds.loc[candidate_mask]

        # Append primary_prob as an additional feature (it carries calibration signal)
        X_meta = X_meta.copy()
        X_meta["primary_prob"] = p_meta.values

        self.feature_cols = [
            c for c in X_meta.columns
            if c not in _META_NON_FEATURE_COLS
            and pd.api.types.is_numeric_dtype(X_meta[c])
        ]

        X = X_meta[self.feature_cols].values.astype(np.float32)
        y = y_meta.values.astype(np.float32)

        pos_rate = y.mean()
        logger.info(
            f"MetaLabeler.fit: {len(X)} samples | "
            f"positive_rate={pos_rate:.2%} | features={len(self.feature_cols)}"
        )

        if pos_rate == 0.0 or pos_rate == 1.0:
            logger.warning(
                f"MetaLabeler.fit: degenerate target (pos_rate={pos_rate:.2%}). "
                "Model may not be meaningful."
            )

        dataset = lgb.Dataset(X, label=y, free_raw_data=False)

        fit_params = {k: v for k, v in self.lgb_params.items() if k != "n_estimators"}
        n_estimators = self.lgb_params.get("n_estimators", 200)

        self.model = lgb.train(
            params=fit_params,
            train_set=dataset,
            num_boost_round=n_estimators,
        )

        logger.info("MetaLabeler.fit: training complete.")

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict_confidence(
        self,
        feature_df: pd.DataFrame,
        primary_predictions: pd.Series,
    ) -> pd.Series:
        """
        Compute combined confidence for each candidate signal.

        combined_confidence = primary_prob * meta_prob

        Only rows where primary_prob >= threshold are scored; the rest receive
        a combined confidence of 0.0.

        Parameters
        ----------
        feature_df : pd.DataFrame
            Feature matrix (same index as primary_predictions).
        primary_predictions : pd.Series
            Primary model's predicted probabilities.

        Returns
        -------
        pd.Series
            Combined confidence scores, same index as primary_predictions.
            Values in [0, 1]. Non-candidates get 0.0.
        """
        if self.model is None:
            raise RuntimeError(
                "MetaLabeler.predict_confidence called before fit(). "
                "Call fit() first."
            )

        idx = feature_df.index
        preds = primary_predictions.reindex(idx).fillna(0.0)
        candidate_mask = preds >= self.threshold

        combined = pd.Series(0.0, index=idx, name="combined_confidence")

        if candidate_mask.sum() == 0:
            logger.debug("MetaLabeler.predict_confidence: no candidates above threshold.")
            return combined

        X_meta = feature_df.loc[candidate_mask].copy()
        X_meta["primary_prob"] = preds.loc[candidate_mask].values

        if self.feature_cols is None:
            raise RuntimeError(
                "MetaLabeler has no feature_cols set — was fit() called successfully?"
            )

        # Align to training feature set; fill missing columns with 0
        missing_cols = set(self.feature_cols) - set(X_meta.columns)
        for col in missing_cols:
            logger.warning(
                f"MetaLabeler.predict_confidence: feature '{col}' missing at inference, filling 0."
            )
            X_meta[col] = 0.0

        X = X_meta[self.feature_cols].values.astype(np.float32)
        meta_probs = self.model.predict(X)

        p_primary = preds.loc[candidate_mask].values
        combined_scores = p_primary * meta_probs

        combined.loc[candidate_mask] = combined_scores

        logger.debug(
            f"MetaLabeler.predict_confidence: {candidate_mask.sum()} candidates | "
            f"meta_prob range [{meta_probs.min():.3f}, {meta_probs.max():.3f}] | "
            f"combined range [{combined_scores.min():.3f}, {combined_scores.max():.3f}]"
        )

        return combined

    # ------------------------------------------------------------------
    # Convenience: is_fitted
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Return True if the model has been trained."""
        return self.model is not None

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return (
            f"MetaLabeler(threshold={self.threshold}, status={status}, "
            f"n_features={len(self.feature_cols) if self.feature_cols else 0})"
        )
