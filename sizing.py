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
    target_vol: float = 0.02,
    vol_span: int = 20,
    normalize: bool = True,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Long-only strategy: compute thresholded, probability-weighted returns with optional volatility targeting.
    """
    valid_idx = X_test.index.intersection(
        returns.stack()[returns.stack().notna()].index
    )
    X_valid = X_test.loc[valid_idx]
    probs = clf.predict_proba(X_valid)[:, 1]

    proba_series = pd.Series(probs, index=valid_idx)
    proba_df = proba_series.unstack().reindex_like(returns)

    # Filter: only long signals above threshold
    weights_df = proba_df.where(proba_df >= threshold, 0)

    if normalize:
        sum_weights = weights_df.sum(axis=1).replace(0, np.nan)
        weights_df = weights_df.div(sum_weights, axis=0).fillna(0)

    raw_returns = (returns * weights_df).sum(axis=1)

    # Volatility targeting
    realized_vol = raw_returns.ewm(span=vol_span).std() * np.sqrt(252)
    scaling = target_vol / realized_vol
    scaled_returns = raw_returns * scaling
    scaled_weights_df = weights_df.mul(scaling, axis=0)

    turnover = scaled_weights_df.diff().abs().sum(axis=1)
    net_returns = scaled_returns - turnover * tc

    # Logging
    logging.info("=== Probability-Weighted (Long-Only) Strategy ===")
    logging.info(f"Threshold: {threshold}, Normalize: {normalize}")
    logging.info(
        f"Average weight (non-zero): {weights_df[weights_df > 0].mean().mean():.4f}"
    )
    logging.info(f"Trades/day (avg): {weights_df.astype(bool).sum(axis=1).mean():.2f}")
    logging.info(
        f"Vol target: {target_vol:.2%}, Realized vol (avg): {realized_vol.mean():.2%}"
    )
    logging.info(f"Total turnover: {turnover.sum():.4f}")

    return scaled_returns, net_returns, scaled_weights_df, proba_df, turnover
