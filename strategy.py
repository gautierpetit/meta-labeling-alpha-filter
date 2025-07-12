import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from signals import generate_momentum_signals



def get_daily_signals(
    prices: pd.DataFrame, 
    monthly_prices: pd.DataFrame
) -> tuple[pd.DataFrame, pd.MultiIndex]:
    """
    Generate daily trading signals based on monthly momentum ranking.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily price data (used to mask NaNs).
    monthly_prices : pd.DataFrame
        Monthly price data (used to compute momentum signal).

    Returns
    -------
    daily_signals : pd.DataFrame
        Binary signal matrix (1 if a stock is held, 0 otherwise).
    signal_dates : pd.MultiIndex
        MultiIndex (date, ticker) of active signal dates.
    """
    # Compute 12-month momentum with 1-month gap
    momentum = monthly_prices.pct_change(12, fill_method=None) - monthly_prices.pct_change(1, fill_method=None)
    monthly_signals = generate_momentum_signals(momentum)

    # Create daily signals with 3-month holding period
    daily_signals = pd.DataFrame(index=prices.index, columns=prices.columns, data=0)
    for date in monthly_signals.index:
        start = date + pd.offsets.MonthEnd(1)  # skip 1-month gap
        end = start + pd.offsets.MonthEnd(2)   # hold for 3 months
        tickers = monthly_signals.columns[monthly_signals.loc[date] == 1]
        daily_signals.loc[start:end, tickers] = 1

    # Mask signals during periods with missing price data
    daily_signals = daily_signals.where(~prices.isna(), other=0)

    # Extract (date, ticker) pairs with active signals
    signal_dates = daily_signals[daily_signals == 1].stack().index

    return daily_signals, signal_dates


def compute_momentum(
    prices: pd.DataFrame, 
    daily_signals: pd.DataFrame, 
    plot: bool = True
) -> pd.Series:
    """
    Compute the daily returns of the momentum strategy.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily price data.
    daily_signals : pd.DataFrame
        Binary signal matrix indicating which stocks are held each day.
    plot : bool, optional
        Whether to plot the cumulative returns, by default True.

    Returns
    -------
    pd.Series
        Daily portfolio returns of the momentum strategy.
    """
    daily_returns = prices.pct_change(fill_method=None)
    strategy_returns = daily_returns * daily_signals
    n_positions = daily_signals.sum(axis=1).replace(0, np.nan)
    mom_returns = strategy_returns.sum(axis=1) / n_positions

    if plot:
        (1 + mom_returns.fillna(0)).cumprod().plot(
            title="Momentum Strategy Performance", figsize=(12, 6)
        )

    return mom_returns




