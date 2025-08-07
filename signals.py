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
    min_gap: float = config.MIN_GAP,
) -> pd.DataFrame:
    """
    Filters trade signals using predicted success probability (label +1).
    Both longs and shorts are retained if the model predicts they will succeed.
    """

    filtered_signals = pd.DataFrame(0, index=daily_signals.index, columns=daily_signals.columns)

    stacked_signals = daily_signals.stack()
    signal_idx = stacked_signals[stacked_signals != 0].index
    valid_idx = signal_idx.intersection(X_test.index)

    if len(valid_idx) == 0:
        logger.warning("No matching signals found for meta-model filtering.")
        return filtered_signals

    X_valid = X_test.loc[valid_idx]
    class_to_index = get_class_to_index(clf)
    probs = clf.predict_proba(X_valid)

    for i, idx in enumerate(valid_idx):
        signal = stacked_signals.loc[idx]
        prob = probs[i]

        p = {
            -1: prob[class_to_index[-1]],
            0: prob[class_to_index[0]],
            1: prob[class_to_index[1]],
        }

         # Tier 1: Basic success threshold (always enforced)
        passes_threshold = p[1] >= threshold
        # Tier 2: Optional confidence margin check
        passes_gap = min_gap == -1 or (p[1] - max(p[0], p[-1])) >= min_gap

        if passes_threshold and passes_gap:
            filtered_signals.at[idx] = signal  # retain original direction (+1 or -1)

    logger.info(f"Valid coverage: {len(valid_idx)} / {len(signal_idx)} ({len(valid_idx)/len(signal_idx):.2%})")
    logger.info(f"Threshold: {threshold}, Min gap: {min_gap}")
    logger.info(f"Long side: Raw:{daily_signals.stack().value_counts().loc[1]} Filtered:{filtered_signals.stack().value_counts().loc[1]} ")
    logger.info(f"Short side: Raw:{daily_signals.stack().value_counts().loc[-1]} Filtered:{filtered_signals.stack().value_counts().loc[-1]} ") if not config.LONG_ONLY else None
    logger.info(f"Total signals: {filtered_signals.stack().value_counts()[1:].sum()}")
    logger.info(f"Filtered percentage: {(filtered_signals.stack().value_counts()[1:].sum() / daily_signals.stack().value_counts()[1:].sum()):.2%}")
    logger.info(f"daily_signals nonzero: {len(signal_idx)}")
    logger.info(f"X_test index: {len(X_test)}")
    logger.info(f"Valid meta-signal entries: {len(valid_idx)}")


    return filtered_signals
