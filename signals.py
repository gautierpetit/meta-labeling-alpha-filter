import logging

import pandas as pd
from sklearn.base import ClassifierMixin

import config

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
    """
    filtered_signals = pd.DataFrame(0, index=daily_signals.index, columns=daily_signals.columns)

    # Get non-zero signal entries
    stacked_signals = daily_signals.stack()
    signal_idx = stacked_signals[stacked_signals != 0].index
    valid_idx = signal_idx.intersection(X_test.index)

    if len(valid_idx) == 0:
        logger.warning("No matching signals found for meta-model filtering.")
        return filtered_signals

    X_valid = X_test.loc[valid_idx]
    

    if hasattr(clf, "class_labels_"):
        class_to_index = {label: i for i, label in enumerate(clf.class_labels_)}
        probs = clf.predict_proba(X_valid)
    elif hasattr(clf, "classes_"):
        class_to_index = {cls: i for i, cls in enumerate(clf.classes_)}
        probs = clf.predict_proba(X_valid)
    else:
        raise ValueError("Classifier must define `class_labels_` or `classes_`.")


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

