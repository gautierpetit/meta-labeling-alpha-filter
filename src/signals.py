"""Signal filtering helpers.

This module contains utilities to apply a trained meta-model to a
matrix of daily candidate signals (MultiIndex [date, ticker]) and
produce a filtered signal matrix keeping only the top / confident
predictions.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin

import src.config as config
from src.utils import get_class_to_index

logger = logging.getLogger(__name__)


def filter_signals_with_meta_model(
    daily_signals: pd.DataFrame,
    clf: ClassifierMixin,
    X_test: pd.DataFrame,
    min_gap: float = config.MIN_GAP,
) -> pd.DataFrame:
    """Filter daily candidate signals using a trained meta-classifier.

    Parameters
    ----------
    daily_signals
        DataFrame indexed by a MultiIndex (date, ticker) or by date with
        tickers as columns. Values should be -1, 0 or 1 where non-zero
        entries are candidate signals.
    clf
        Trained classifier implementing `predict_proba` and compatible
        with `src.utils.get_class_to_index`.
    X_test
        Feature matrix aligned to the same index/columns as `daily_signals`.
    min_gap
        Minimum probability gap required between P(+1) and the best
        competing class to accept a signal. Use -1 to disable this check.

    Returns
    -------
    pd.DataFrame
        Filtered signals with the same shape as `daily_signals`.
    """

    th_long = float(getattr(config, "META_PROBA_THRESHOLD_LONG", 0.5))
    th_short = float(getattr(config, "META_PROBA_THRESHOLD_SHORT", 0.5))
    score_mode = str(getattr(config, "META_SCORE_MODE", "prob"))  # "prob" | "edge" | "logit_edge"

    filtered_signals = pd.DataFrame(0, index=daily_signals.index, columns=daily_signals.columns)

    idx_map: dict[int, int] = get_class_to_index(clf)

    stacked_signals: pd.Series = daily_signals.stack()
    # Boolean mask is a Series[bool]
    entry_mask: pd.Series = stacked_signals.ne(0)
    signal_idx = stacked_signals.loc[entry_mask].index

    # Ensure valid_idx is compatible with expected index types
    valid_idx = signal_idx.intersection(X_test.index)

    if len(valid_idx) == 0:
        logger.warning("No matching signals found for meta-model filtering.")
        return filtered_signals

    # predict_proba must be called with rows in the same order as valid_idx
    X_valid = X_test.loc[valid_idx]
    probs = np.asarray(clf.predict_proba(X_valid))

    for i, idx in enumerate(valid_idx):
        signal = int(stacked_signals.loc[idx])  # +1 or -1
        prob = probs[i]
        p = {-1: prob[idx_map[-1]], 0: prob[idx_map[0]], 1: prob[idx_map[1]]}

        # side-specific threshold
        th_side = th_long if signal > 0 else th_short
        passes_threshold = p[1] >= th_side
        passes_gap = (min_gap == -1) or ((p[1] - max(p[0], p[-1])) >= min_gap)

        if passes_threshold and passes_gap:
            filtered_signals.at[idx] = signal

    K = int(getattr(config, "TOP_K_PER_DAY", -1))
    if K != -1:
        # per-day top-K selection using configured scoring mode
        day_mask: pd.Series = filtered_signals.ne(0).any(axis=1)
        nz_days = filtered_signals.index[day_mask.to_numpy()]
        for dt in nz_days:
            row = filtered_signals.loc[dt]
            if row.abs().sum() == 0:
                continue
            cols = row.index[row != 0].tolist()
            if len(cols) <= K:
                continue

            # Select rows for this date and the subset of tickers (MultiIndex: Date, Ticker).
            try:
                x_dt = X_test.loc[pd.IndexSlice[dt, cols], :]
            except KeyError:
                # if some (date,ticker) combos are missing, skip this date
                logger.debug("Missing rows for date %s while scoring top-K; skipping.", dt)
                continue
            # Score is on P(+1)
            proba_dt = np.asarray(clf.predict_proba(x_dt))[:, idx_map[1]]
            side_dt = np.sign(filtered_signals.loc[dt, cols].to_numpy())
            th_vec = np.where(side_dt > 0, th_long, th_short)

            if score_mode == "edge":
                # Score is on the edge (P(+1) - threshold)
                score = proba_dt - th_vec
            elif score_mode == "logit_edge":
                # Score is on the logit edge (logit(P(+1)) - logit(threshold))
                eps = 1e-6
                logit = np.log(np.clip(proba_dt, eps, 1 - eps)) - np.log(
                    np.clip(1 - proba_dt, eps, 1 - eps)
                )
                th_logit = np.log(np.clip(th_vec, eps, 1 - eps)) - np.log(
                    np.clip(1 - th_vec, eps, 1 - eps)
                )
                score = logit - th_logit
            else:  # "prob"
                score = proba_dt

            top_cols = cols[np.argsort(score)[-K:]]
            drop_cols = [c for c in cols if c not in top_cols]
            filtered_signals.loc[dt, drop_cols] = 0

    # summary counts and logging
    vc_raw = daily_signals.stack().value_counts()
    vc_flt = filtered_signals.stack().value_counts()
    long_raw = int(vc_raw.get(1, 0))
    long_flt = int(vc_flt.get(1, 0))
    short_raw = int(vc_raw.get(-1, 0))
    short_flt = int(vc_flt.get(-1, 0))
    total_raw_nz = int((daily_signals != 0).sum().sum())
    total_flt_nz = int((filtered_signals != 0).sum().sum())

    coverage_pct = (len(valid_idx) / len(signal_idx)) if len(signal_idx) > 0 else 0.0
    logger.info(
        "Valid coverage: %d / %d (%.2f%%)", len(valid_idx), len(signal_idx), 100.0 * coverage_pct
    )
    logger.info(
        "Long Threshold: %.3f, Short Threshold: %.3f, Min gap: %s", th_long, th_short, str(min_gap)
    )
    logger.info("Picking the top %d signals per day, mode %s", K, score_mode)
    logger.info("Long side: Raw:%d Filtered:%d", long_raw, long_flt)
    if not getattr(config, "LONG_ONLY", False):
        logger.info("Short side: Raw:%d Filtered:%d", short_raw, short_flt)
    logger.info("Total signals: %d", total_flt_nz)
    filtered_pct = (total_flt_nz / total_raw_nz) if total_raw_nz > 0 else 0.0
    logger.info("Filtered percentage: %.2f%%", 100.0 * filtered_pct)

    return filtered_signals
