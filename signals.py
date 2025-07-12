import pandas as pd
from sklearn.base import ClassifierMixin

import config


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
    Filters signals using meta-model probability predictions.

    Parameters:
        daily_signals (pd.DataFrame): Initial binary signals.
        clf (ClassifierMixin): Trained classification model.
        X_test (pd.DataFrame): Meta features for predictions.
        threshold (float): Probability threshold for inclusion.

    Returns:
        pd.DataFrame: Filtered signals.
    """
    filtered_signals = pd.DataFrame(
        0, index=daily_signals.index, columns=daily_signals.columns
    )
    valid_idx = X_test.index.intersection(
        daily_signals.stack()[daily_signals.stack() == 1].index
    )
    X_valid = X_test.loc[valid_idx]
    probs = clf.predict_proba(X_valid)[:, 1]
    selected_idx = valid_idx[probs >= threshold]
    for date, ticker in selected_idx:
        filtered_signals.at[date, ticker] = 1
    return filtered_signals
