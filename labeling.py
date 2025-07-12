import numpy as np
import pandas as pd
from tqdm import tqdm

import config


def get_trade_outcomes(
    prices: pd.DataFrame,
    daily_signals: pd.DataFrame,
    tp: float = config.TARGET_TP_THRESHOLD,
    sl: float = config.TARGET_SL_THRESHOLD,
) -> pd.DataFrame:
    """
    Evaluate trade outcomes based on take profit (TP) and stop loss (SL).

    Parameters:
        prices (pd.DataFrame): Asset price data.
        daily_signals (pd.DataFrame): Trade entry signals.
        tp (float): Take profit threshold.
        sl (float): Stop loss threshold.

    Returns:
        pd.DataFrame: Trade outcomes with 1 (TP), 0 (SL), or NaN.
    """
    outcomes = pd.DataFrame(index=daily_signals.index, columns=daily_signals.columns)
    for date in tqdm(daily_signals.index):
        tickers = daily_signals.columns[daily_signals.loc[date] == 1]
        for ticker in tickers:
            entry_price = prices.at[date, ticker]
            if pd.isna(entry_price):
                continue
            future_dates = prices.index[prices.index > date][:63]
            future_prices = prices.loc[future_dates, ticker]
            if future_prices.empty:
                continue
            max_return = (future_prices / entry_price - 1).max()
            min_return = (future_prices / entry_price - 1).min()
            if max_return >= tp:
                outcomes.at[date, ticker] = 1
            elif min_return <= -sl:
                outcomes.at[date, ticker] = 0
            else:
                outcomes.at[date, ticker] = np.nan
    return outcomes
