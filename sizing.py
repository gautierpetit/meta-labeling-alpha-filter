import logging

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin

import config
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

logger = logging.getLogger(__name__)


def compute_probability_weighted_returns(
    clf: ClassifierMixin,
    X_test: pd.DataFrame,
    returns: pd.DataFrame,
    threshold: float = config.META_PROBA_THRESHOLD,
    target_vol: float = 0.02,
    vol_span: int = 20,
    normalize: bool = True,
    max_leverage: float = 1.5,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.Series, pd.Series]:
    """
    Compute probability-weighted returns with volatility targeting and optional long/short and leverage cap.
    Supports both sklearn and custom KerasClassifier with label_map.
    """
    valid_idx = X_test.index.intersection(returns.stack()[returns.stack().notna()].index)
    X_valid = X_test.loc[valid_idx]

        # Handle label mapping for different model types
    if hasattr(clf, "class_labels_"):
        class_to_index = {label: i for i, label in enumerate(clf.class_labels_)}
        proba = clf.predict_proba(X_valid)
    elif hasattr(clf, "classes_"):
        class_to_index = {cls: i for i, cls in enumerate(clf.classes_)}
        proba = clf.predict_proba(X_valid)
    else:
        raise ValueError("Classifier must define `class_labels_` or `classes_`.")


    

    # Long
    proba_long = pd.Series(proba[:, class_to_index[1]], index=valid_idx).unstack().reindex_like(returns)

    
    
    # Short
    proba_short = pd.Series(proba[:, class_to_index[-1]], index=valid_idx).unstack().reindex_like(returns)

    # Apply thresholds
    long_mask = proba_long >= threshold
    short_mask = proba_short >= threshold

    
    weights_df = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)

    if config.LONG_ONLY:
        if config.INVERT_SIGNALS:
            # Use model's short confidence to go long (inverted logic), but still no shorts
            logic_mode = "INVERTED"
            weights_df[short_mask] = +proba_short[short_mask]
        else:
            # Standard: use model's long confidence to go long
            logic_mode = "NORMAL"
            weights_df[long_mask] = +proba_long[long_mask]
    else:
        if config.INVERT_SIGNALS:
            # Fully inverted logic: go short what model thinks is good, long what it thinks is bad
            logic_mode = "INVERTED"
            weights_df[long_mask] = -proba_long[long_mask]
            weights_df[short_mask] = +proba_short[short_mask]
        else:
            # Normal: long the strong, short the weak
            logic_mode = "NORMAL"
            weights_df[long_mask] = +proba_long[long_mask]
            weights_df[short_mask] = -proba_short[short_mask]



    if normalize:
        sum_abs = weights_df.abs().sum(axis=1).replace(0, np.nan)
        weights_df = weights_df.div(sum_abs, axis=0).fillna(0)

    raw_returns = (returns * weights_df).sum(axis=1)

    # Vol targeting
    realized_vol = raw_returns.ewm(span=vol_span).std() * np.sqrt(252)
    scaling = target_vol / realized_vol
    scaled_weights_df = weights_df.mul(scaling, axis=0)

    # Leverage cap
    leverage = scaled_weights_df.abs().sum(axis=1).replace(0, np.nan)
    cap_ratio = (max_leverage / leverage).clip(upper=1).fillna(1)
    scaled_weights_df = scaled_weights_df.mul(cap_ratio, axis=0)

    scaled_returns = (returns * scaled_weights_df).sum(axis=1)
    turnover = scaled_weights_df.diff().abs().sum(axis=1)

    long_tc = config.LONG_SIDE_TC  # e.g., 0.001
    short_tc = config.SHORT_SIDE_TC  # e.g., 0.002 or 0.003

    diff = scaled_weights_df.diff().abs()

    costs = (
        diff[scaled_weights_df > 0].sum(axis=1) * long_tc +
        diff[scaled_weights_df < 0].sum(axis=1) * short_tc
    )

    net_returns = scaled_returns - costs
            

    # Logging
    strategy_type = "Long-Only" if config.LONG_ONLY else "Long/Short"
    logger.info(f"=== Probability-Weighted ({strategy_type}) Strategy ===")  
    logger.info(f"Signal logic mode: {logic_mode}")
    logger.info(f"Threshold: {threshold}, Normalize: {normalize}")
    logger.info(f"Average weight (non-zero): {weights_df[weights_df != 0].mean().mean():.4f}")
    logger.info(f"Trades/day (avg): {weights_df.astype(bool).sum(axis=1).mean():.2f}")
    logger.info(f"Vol target: {target_vol:.2%}, Realized vol (avg): {realized_vol.mean():.2%}")
    logger.info(f"Total turnover: {turnover.sum():.4f}")
    logger.info(f"Avg daily cost: {costs.mean():.5f}, Annualized: {(1 + costs.mean()) ** 252 - 1:.2%}")
    logger.info(f"Average leverage: {leverage.mean():.2f}, Max leverage: {leverage.max():.2f}")

    return scaled_returns, net_returns, scaled_weights_df, turnover
