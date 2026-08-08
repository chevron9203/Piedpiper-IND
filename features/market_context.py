import numpy as np
import pandas as pd
from loguru import logger


def index_features(
    index_df: pd.DataFrame,
    stock_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    windows: list[int] = [20, 60],
) -> pd.DataFrame:
    target_index = stock_df.index

    # Align all inputs to the stock's trading calendar via reindex+ffill
    idx = index_df["close"].reindex(target_index, method="ffill")
    vix_close = vix_df["close"].reindex(target_index, method="ffill")
    stock_close = stock_df["close"]

    idx_ret_1d = idx.pct_change(1).rename("index_ret_1d")
    idx_ret_5d = idx.pct_change(5).rename("index_ret_5d")
    idx_ret_20d = idx.pct_change(20).rename("index_ret_20d")

    idx_log_ret = np.log(idx / idx.shift(1))
    mp_20 = max(1, 20 // 2)
    index_rvol_20d = (
        idx_log_ret.rolling(20, min_periods=mp_20).std() * np.sqrt(252)
    ).rename("index_rvol_20d")

    vix_level = vix_close.rename("vix_level")
    vix_change_1d = vix_close.pct_change(1).rename("vix_change_1d")

    stock_ret = stock_close.pct_change(1)
    idx_ret = idx.pct_change(1)

    beta_corr_parts = {}
    for w in windows:
        mp = max(1, w // 2)
        cov = stock_ret.rolling(w, min_periods=mp).cov(idx_ret)
        var_idx = idx_ret.rolling(w, min_periods=mp).var()
        beta_corr_parts[f"beta_{w}d"] = cov / (var_idx + 1e-9)
        beta_corr_parts[f"corr_{w}d"] = stock_ret.rolling(w, min_periods=mp).corr(idx_ret)

    result = pd.concat(
        [
            idx_ret_1d,
            idx_ret_5d,
            idx_ret_20d,
            index_rvol_20d,
            vix_level,
            vix_change_1d,
            pd.DataFrame(beta_corr_parts, index=target_index),
        ],
        axis=1,
    )
    return result
