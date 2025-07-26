import logging

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin

import config

logger = logging.getLogger(__name__)


def compute_probability_weighted_returns(
    clf: ClassifierMixin,
    X_test: pd.DataFrame,
    returns: pd.DataFrame,
    threshold: float = config.META_PROBA_THRESHOLD,
    tc: float = config.TRANSACTION_COSTS,
    long_only: bool = True,
    target_vol: float = 0.02,
    vol_span: int = 20,
    normalize: bool = True,
    max_leverage: float = 1.5,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Compute probability-weighted returns with volatility targeting and optional long/short and leverage cap.
    """
    valid_idx = X_test.index.intersection(
        returns.stack()[returns.stack().notna()].index
    )
    X_valid = X_test.loc[valid_idx]
    

    proba_long = pd.Series(clf.predict_proba(X_valid)[:, 1], index=valid_idx).unstack().reindex_like(returns)


    if long_only:
        weights_df = proba_long.where(proba_long >= threshold, 0)
    else:
        proba_short = pd.Series(clf.predict_proba(X_valid)[:, -1], index=valid_idx).unstack().reindex_like(returns)
        
        long_mask = proba_long >= threshold
        short_mask = proba_short >= threshold
        
        weights_df = pd.DataFrame(0, index=returns.index, columns=returns.columns)
        weights_df[long_mask] = proba_long[long_mask]
        weights_df[short_mask] = -proba_short[short_mask]



    if normalize:
        sum_abs = weights_df.abs().sum(axis=1).replace(0, np.nan)
        weights_df = weights_df.div(sum_abs, axis=0).fillna(0)


    raw_returns = (returns * weights_df).sum(axis=1)

    # Volatility targeting
    realized_vol = raw_returns.ewm(span=vol_span).std() * np.sqrt(252)
    scaling = target_vol / realized_vol
    scaled_weights_df = weights_df.mul(scaling, axis=0)

    # Leverage capping
    leverage = scaled_weights_df.abs().sum(axis=1)
    leverage = leverage.replace(0, np.nan)
    cap_ratio = (max_leverage / leverage).clip(upper=1).fillna(1)
    scaled_weights_df = scaled_weights_df.mul(cap_ratio, axis=0)
        
    scaled_returns = (returns * scaled_weights_df).sum(axis=1)
    turnover = scaled_weights_df.diff().abs().sum(axis=1)
    net_returns = scaled_returns - turnover * tc

    # Logging
    strategy_type = "Long-Only" if long_only else "Long/Short"
    logger.info(f"=== Probability-Weighted ({strategy_type}) Strategy ===")
    logger.info(f"Threshold: {threshold}, Normalize: {normalize}")
    logger.info(
        f"Average weight (non-zero): {weights_df[weights_df > 0].mean().mean():.4f}"
    )
    logger.info(f"Trades/day (avg): {weights_df.astype(bool).sum(axis=1).mean():.2f}")
    logger.info(
        f"Vol target: {target_vol:.2%}, Realized vol (avg): {realized_vol.mean():.2%}"
    )
    logger.info(f"Total turnover: {turnover.sum():.4f}")
    logger.info(f"Average leverage: {leverage.mean():.2f}, Max leverage: {leverage.max():.2f}")

    return scaled_returns, net_returns, scaled_weights_df, turnover
