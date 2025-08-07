import logging
from typing import Tuple,Literal

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin

import config
from utils import get_class_to_index

logger = logging.getLogger(__name__)


def compute_probability_weighted_returns(
    clf: ClassifierMixin,
    filtered_signals: pd.DataFrame,
    X_test: pd.DataFrame,
    returns: pd.DataFrame,
    logic: Literal["NORMAL", "INVERTED"] = config.LOGIC,
    prob_weighting: bool = config.PROB_WEIGHTING,
    target_vol: float = config.TARGET_VOL,
    leverage_cap: float = config.LEVERAGE_CAP,
) -> Tuple[pd.Series, pd.Series, pd.DataFrame, pd.Series, pd.Series]:
    

    weights = filtered_signals.astype(float).copy()
    # Normalize per row (day) so that weights sum to 1 (or -1, etc.)
    row_sums = weights.abs().sum(axis=1).replace(0, np.nan)
    weights = weights.div(row_sums, axis=0).fillna(0.0)
    
    
    if prob_weighting:
        proba_array = clf.predict_proba(X_test)  # shape (n, 3)
        success_idx = get_class_to_index(clf)[1]

        signal_values = filtered_signals.stack().reindex_like(X_test).values
        success_probs = proba_array[np.arange(len(signal_values)), success_idx]

        weighted = signal_values * success_probs
        weights = pd.Series(weighted, index=X_test.index).unstack().reindex_like(filtered_signals).fillna(0.0)

        """
        EXPERIMENTAL: softening a trade based on the model’s uncertainty
        RISKS: double penalization
        complicates interpretation
        confidence penalty from P(0) and P(-1)

        timeout_idx = get_class_to_index(clf)[0]
        fail_idx = get_class_to_index(clf)[-1]

        timeout_probs = proba_array[np.arange(len(signal_values)), timeout_idx]
        fail_probs = proba_array[np.arange(len(signal_values)), fail_idx]

        penalty = 1 - (timeout_probs + fail_probs)
        penalty = np.clip(penalty, 0.0, 1.0)

        weighted = signal_values * success_probs * penalty
        """

    if logic == "INVERTED":
        # Inverted logic: long what model thinks is bad, short what it thinks is good
        weights = -weights

    elif logic != "NORMAL":
        raise ValueError(f"Unsupported logic mode: {logic}. Use 'NORMAL' or 'INVERTED'.")
    
    

    # Vol targeting
    raw_returns = (returns * weights).sum(axis=1)
    realized_vol = raw_returns.ewm(span=config.VOL_SPAN).std() * np.sqrt(252)
    if target_vol != -1:
        scaling = target_vol / realized_vol
        weights = weights.mul(scaling, axis=0)

    # Leverage cap
    leverage = weights.abs().sum(axis=1).replace(0, np.nan)
    if leverage_cap != -1:
        cap_ratio = (leverage_cap / leverage).clip(upper=1).fillna(1)
        weights = weights.mul(cap_ratio, axis=0)

    port_returns = (returns * weights).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1)

    long_tc = config.LONG_SIDE_TC  # e.g., 0.001
    short_tc = config.SHORT_SIDE_TC  # e.g., 0.002 or 0.003

    diff = weights.diff().abs()

    costs = (
        diff[weights > 0].sum(axis=1) * long_tc
        + diff[weights < 0].sum(axis=1) * short_tc
    )

    net_returns = port_returns - costs

    # Logging
    strategy_type = "Long-Only" if config.LONG_ONLY else "Long/Short"
    logger.info(f"=== Probability-Weighted ({strategy_type}) Strategy ===")
    logger.info(f"Signal logic mode: {logic}")
    logger.info(f"Probability weighting: {prob_weighting}")
    logger.info(
        f"Average weight (non-zero): {weights[weights != 0].mean().mean():.4f}"
    )
    logger.info(f"Trades/day (avg): {weights.astype(bool).sum(axis=1).mean():.2f}")
    logger.info(f"Volatility targeting: {target_vol:.2%}")
    logger.info(
        f"Vol target: {target_vol:.2%}, Realized vol (avg): {realized_vol.mean():.2%}"
    )
    logger.info(f"Total turnover: {turnover.sum():.4f}")
    logger.info(
        f"Avg daily cost: {costs.mean():.5f}, Annualized: {(1 + costs.mean()) ** 252 - 1:.2%}"
    )
    logger.info (f"Leverage cap: {leverage_cap:.2f}")
    logger.info(
        f"Average leverage: {leverage.mean():.2f}, Max leverage: {leverage.max():.2f}"
    )

    return port_returns, net_returns, weights.fillna(0.0), turnover, costs
