import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import src.config as config

# Configure logging
logger = logging.getLogger(__name__)


def get_daily_signals(
    prices: pd.DataFrame,
    monthly_prices: pd.DataFrame,
    long_only: bool = config.LONG_ONLY,
    hold_months: int = 3,
    skip_months: int = 1,
) -> pd.DataFrame:
    """
    Generate daily trading signals based on monthly momentum ranking.
    Can return long-only or long-short signals.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily price data (used to mask NaNs).
    monthly_prices : pd.DataFrame
        Monthly price data (used to compute momentum signal).
    long_only : bool, optional
        If True, generates long-only signals; else, long-short. Default is config.LONG_ONLY.
    hold_months : int, optional
        Number of months to hold a position. Default is 3.
    skip_months : int, optional
        Number of months to skip after the signal date. Default is 1.

    Returns
    -------
    pd.DataFrame
        Signal matrix (-1, 0, 1 values)
    """
    logger.info("Generating daily trading signals.")

    momentum = monthly_prices.pct_change(
        12, fill_method=None
    ) - monthly_prices.pct_change(1, fill_method=None)
    daily_signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)

    for date in momentum.index:
        momentums = momentum.loc[date]
        start = date + pd.offsets.MonthEnd(skip_months)
        end = start + pd.offsets.MonthEnd(hold_months - 1)

        if long_only:
            long_thresh = momentums.quantile(config.TOP_QUANTILE)
            longs = momentums[momentums >= long_thresh].index
            daily_signals.loc[start:end, longs] = 1
        else:
            long_thresh = momentums.quantile(config.TOP_QUANTILE)
            short_thresh = momentums.quantile(config.BOTTOM_QUANTILE)
            longs = momentums[momentums >= long_thresh].index
            shorts = momentums[momentums <= short_thresh].index
            daily_signals.loc[start:end, longs] = 1
            daily_signals.loc[start:end, shorts] = -1

    daily_signals = daily_signals.where(~prices.isna(), other=0)
    

    logger.info("Daily trading signals generated successfully.")
    return daily_signals


def compute_momentum(
    prices: pd.DataFrame,
    daily_signals: pd.DataFrame,
) -> pd.Series:
    """
    Compute the daily returns of a momentum strategy.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily price data.
    daily_signals : pd.DataFrame
        Signal matrix (-1, 0, 1 values).
    Returns
    -------
    pd.Series
        Daily strategy returns.
    """
    logger.info("Computing momentum strategy returns.")

    daily_returns = prices.pct_change(fill_method=None)
    active = daily_signals != 0
    valid  = daily_returns.notna()
    used   = active & valid

    strategy_returns = daily_returns.where(used, 0.0) * daily_signals.where(used, 0.0)
    den = used.sum(axis=1).replace(0, np.nan)
    mom_returns = strategy_returns.sum(axis=1) / den

    logger.info("Momentum strategy returns computed successfully.")
    return mom_returns
