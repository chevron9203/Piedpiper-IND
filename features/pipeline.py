import pandas as pd
from loguru import logger

from features.price_action import (
    multi_horizon_returns,
    gap_open,
    body_ratio,
    upper_wick_ratio,
    lower_wick_ratio,
    rolling_high_low_position,
    high52w_ratio,
)
from features.volatility import atr, realized_vol, vol_of_vol
from features.volume import volume_ratio, obv, price_volume_divergence, vwap_deviation
from features.momentum import rsi, macd, bollinger_pct_b, distance_from_ma
from features.regime import adx, trend_regime
from features.market_context import index_features


class FeaturePipeline:
    def __init__(self, version: str = "v1"):
        self.version = version

    def compute(
        self,
        stock_df: pd.DataFrame,
        index_df: pd.DataFrame,
        vix_df: pd.DataFrame,
        symbol: str,
    ) -> pd.DataFrame:
        logger.info(f"Computing features for {symbol} [{self.version}]")

        parts = [
            multi_horizon_returns(stock_df),
            gap_open(stock_df),
            body_ratio(stock_df),
            upper_wick_ratio(stock_df),
            lower_wick_ratio(stock_df),
            rolling_high_low_position(stock_df),
            atr(stock_df),
            realized_vol(stock_df),
            vol_of_vol(stock_df),
            volume_ratio(stock_df),
            obv(stock_df),
            price_volume_divergence(stock_df),
            vwap_deviation(stock_df),
            rsi(stock_df),
            macd(stock_df),
            bollinger_pct_b(stock_df),
            distance_from_ma(stock_df),
            adx(stock_df),
            trend_regime(stock_df),
            index_features(index_df, stock_df, vix_df),
        ]

        result = pd.concat(parts, axis=1)

        n_features = result.shape[1]
        # Drop rows where more than half the feature columns are NaN
        threshold = n_features * 0.5
        result = result.dropna(thresh=int(n_features - threshold))

        result["symbol"] = symbol
        logger.info(f"Feature matrix shape for {symbol}: {result.shape}")
        return result
