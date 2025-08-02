import logging

import pandas as pd
from sklearn.base import ClassifierMixin

import config
from utils import get_class_to_index

logger = logging.getLogger(__name__)


def filter_signals_with_meta_model(
    daily_signals: pd.DataFrame,
    clf: ClassifierMixin,
    X_test: pd.DataFrame,
    threshold: float = config.META_PROBA_THRESHOLD,
) -> pd.DataFrame:
    """
    Filters daily trade signals using the predicted probabilities from a meta-model.
    Retains long (+1) or short (-1) signals only if the model predicts success probability above threshold.
    Works with both sklearn-like models and KerasCalibrationCV wrapper.

    Args:
        daily_signals (pd.DataFrame): DataFrame of daily trade signals (+1, -1, or 0).
        clf (ClassifierMixin): Trained classifier for predicting probabilities.
        X_test (pd.DataFrame): Feature matrix for testing.
        threshold (float): Probability threshold for filtering signals (default: config.META_PROBA_THRESHOLD).

    Returns:
        pd.DataFrame: Filtered trade signals.

    Example:
        filtered_signals = filter_signals_with_meta_model(daily_signals, clf, X_test, threshold=0.6)
    """
    filtered_signals = pd.DataFrame(
        0, index=daily_signals.index, columns=daily_signals.columns
    )

    # Get non-zero signal entries
    stacked_signals = daily_signals.stack()
    signal_idx = stacked_signals[stacked_signals != 0].index
    valid_idx = signal_idx.intersection(X_test.index)

    if len(valid_idx) == 0:
        logger.warning("No matching signals found for meta-model filtering.")
        return filtered_signals

    X_valid = X_test.loc[valid_idx]

    # Get class-to-index mapping and probabilities
    class_to_index = get_class_to_index(clf)
    probs = clf.predict_proba(X_valid)

    # Longs → class +1
    long_mask = stacked_signals.loc[valid_idx] == 1
    long_probs = probs[long_mask.values, class_to_index[1]]
    long_idx = valid_idx[long_mask]
    passed_long = long_idx[long_probs >= threshold]

    # Shorts → class -1
    short_mask = stacked_signals.loc[valid_idx] == -1
    short_probs = probs[short_mask.values, class_to_index[-1]]
    short_idx = valid_idx[short_mask]
    passed_short = short_idx[short_probs >= threshold]

    # Write back signals
    for idx in passed_long:
        filtered_signals.at[idx] = 1
    for idx in passed_short:
        filtered_signals.at[idx] = -1

    return filtered_signals
