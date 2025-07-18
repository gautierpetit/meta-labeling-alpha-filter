import logging

import pandas as pd
from sklearn.base import ClassifierMixin

import config

logger = logging.getLogger(__name__)


def generate_momentum_signals(
    momentum_df: pd.DataFrame, top_quantile: float = config.TOP_QUANTILE
) -> pd.DataFrame:
    """
    Generate momentum signals monthly based on top quantile threshold.

    Parameters:
        momentum_df (pd.DataFrame): DataFrame of momentum scores.
        top_quantile (float): Quantile to define top-performing assets.

    Returns:
        pd.DataFrame: Binary signals with 1 for top decile performers.
    """
    signals = pd.DataFrame(index=momentum_df.index, columns=momentum_df.columns, data=0)
    for date in momentum_df.index:
        momentums = momentum_df.loc[date]
        threshold = momentums.quantile(top_quantile)
        selected = momentums[momentums >= threshold].index
        signals.loc[date, selected] = 1
    return signals


def filter_signals_with_meta_model(
    daily_signals: pd.DataFrame,
    clf: ClassifierMixin,
    X_test: pd.DataFrame,
    threshold: float = config.META_PROBA_THRESHOLD,
) -> pd.DataFrame:
    """
    Filters daily trade signals using the predicted probabilities from a meta-model.
    Only retains signals where the model predicts success probability above the threshold.

    Parameters:
        daily_signals (pd.DataFrame): Binary signal matrix (1 = entry signal).
        clf (ClassifierMixin): Trained meta-model (must support predict_proba).
        X_test (pd.DataFrame): Meta-model features, indexed by (date, ticker).
        threshold (float): Minimum predicted probability for signal inclusion.

    Returns:
        pd.DataFrame: Filtered binary signal matrix.
    """
    # Initialize output with zeros
    filtered_signals = pd.DataFrame(
        0, index=daily_signals.index, columns=daily_signals.columns
    )

    # Get (date, ticker) pairs where a signal was issued
    signal_idx = daily_signals.stack()[daily_signals.stack() == 1].index

    # Restrict to cases where we have meta-features
    valid_idx = signal_idx.intersection(X_test.index)

    if len(valid_idx) == 0:
        logger.warning("No matching signals found for meta-model filtering.")
        return filtered_signals

    # Predict probabilities using the meta-model
    X_valid = X_test.loc[valid_idx]
    probs = clf.predict_proba(X_valid)[:, 1]

    # Keep only the signals above the threshold
    selected_idx = valid_idx[probs >= threshold]

    # Set filtered signals to 1 where accepted
    for date, ticker in selected_idx:
        filtered_signals.at[date, ticker] = 1

    return filtered_signals
