import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

# Configure logging
logger = logging.getLogger(__name__)


def get_daily_signals(
    prices: pd.DataFrame,
    monthly_prices: pd.DataFrame,
    long_only: bool = config.LONG_ONLY,
    hold_months: int = 3,
    skip_months: int = 1,
) -> tuple[pd.DataFrame, pd.MultiIndex]:
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
    tuple[pd.DataFrame, pd.MultiIndex]
        Signal matrix (-1, 0, 1 values) and MultiIndex of (date, ticker) with active signals.
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
    signal_dates = daily_signals.stack()[daily_signals.stack() != 0].index

    logger.info("Daily trading signals generated successfully.")
    return daily_signals, signal_dates


def compute_momentum(
    prices: pd.DataFrame,
    daily_signals: pd.DataFrame,
    long_only: bool = config.LONG_ONLY,
    plot: bool = True,
) -> pd.Series:
    """
    Compute the daily returns of a momentum strategy.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily price data.
    daily_signals : pd.DataFrame
        Signal matrix (-1, 0, 1 values).
    long_only : bool, optional
        Whether this is a long-only strategy. Default is config.LONG_ONLY.
    plot : bool, optional
        Whether to plot the cumulative returns. Default is True.

    Returns
    -------
    pd.Series
        Daily strategy returns.
    """
    logger.info("Computing momentum strategy returns.")

    daily_returns = prices.pct_change(fill_method=None)
    strategy_returns = daily_returns * daily_signals

    if long_only:
        total_positions = (daily_signals != 0).sum(axis=1)
    else:
        long_count = (daily_signals == 1).sum(axis=1)
        short_count = (daily_signals == -1).sum(axis=1)
        total_positions = long_count + short_count

    total_positions = total_positions.replace(0, np.nan)
    mom_returns = strategy_returns.sum(axis=1) / total_positions

    if plot:
        logger.info("Plotting cumulative returns.")
        (1 + mom_returns.fillna(0)).cumprod().plot(
            title="Momentum Strategy Performance"
            if long_only
            else "Long/Short Momentum Strategy",
            figsize=(12, 6),
        )
        plt.close()

    logger.info("Momentum strategy returns computed successfully.")
    return mom_returns
