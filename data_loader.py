import pandas as pd

import config


def load_prices() -> pd.DataFrame:
    """
    Load filtered daily price data.

    Returns:
        pd.DataFrame: DataFrame containing daily price data.
    """
    return pd.read_parquet(config.FILTERED_PRICES)


def load_monthly_prices() -> pd.DataFrame:
    """
    Return month-end resampled filtered prices.

    Returns:
        pd.DataFrame: DataFrame containing month-end prices.
    """
    daily = load_prices()
    return daily.resample("ME").last()


def load_returns() -> pd.DataFrame:
    """
    Load daily returns calculated from price data.

    Returns:
        pd.DataFrame: DataFrame containing daily returns.
    """
    prices = load_prices()
    return prices.pct_change(fill_method=None)


def load_volumes() -> pd.DataFrame:
    """
    Load filtered daily volume data.

    Returns:
        pd.DataFrame: DataFrame containing daily volume data.
    """
    return pd.read_parquet(config.FILTERED_VOLUMES)


def load_low_prices() -> pd.DataFrame:
    """
    Load daily low prices.

    Returns:
        pd.DataFrame: DataFrame containing daily low prices.
    """
    return pd.read_parquet(config.FILTERED_LOW)


def load_high_prices() -> pd.DataFrame:
    """
    Load daily high prices.

    Returns:
        pd.DataFrame: DataFrame containing daily high prices.
    """
    return pd.read_parquet(config.FILTERED_HIGH)


def load_vix() -> pd.Series:
    """
    Load and forward-fill the VIX index for volatility context.

    Returns:
        pd.Series: Series containing VIX index aligned with the trading calendar.
    """
    vix = pd.read_parquet(config.VIX)
    return vix.reindex(load_prices().index).ffill()


def load_spy_prices() -> pd.Series:
    """
    Load SPY ETF closing prices for use as a benchmark.

    Returns:
        pd.Series: Series containing SPY ETF closing prices.
    """
    return pd.read_parquet(config.SPY).squeeze()


def load_spy_returns() -> pd.Series:
    """
    Compute and return daily returns of the SPY ETF.

    Returns:
        pd.Series: Series containing daily SPY ETF returns.
    """
    spy_prices = load_spy_prices()
    return spy_prices.pct_change().squeeze()


def load_labels() -> pd.Series:
    """
    Load saved trade outcomes (Y labels).

    Returns:
        pd.Series: Series containing trade outcomes.
    """
    return pd.read_parquet(config.Y).squeeze()


def load_features() -> pd.DataFrame:
    """
    Load saved feature matrix (X features).

    Returns:
        pd.DataFrame: DataFrame containing feature matrix.
    """
    return pd.read_parquet(config.X)


def load_rates() -> pd.DataFrame:
    """
    Load the market yield on US Treasury 10 Year (DGS10).

    Returns:
        pd.DataFrame: DataFrame containing market yield data aligned with the trading calendar.
    """
    rates = pd.read_csv(config.DGS10)
    rates.index = pd.to_datetime(rates["observation_date"], format="%Y-%m-%d")
    rates = rates.reindex(load_prices().index).ffill()
    rates.drop(columns="observation_date", inplace=True)
    return rates
