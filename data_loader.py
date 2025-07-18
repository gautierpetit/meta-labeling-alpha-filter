import pandas as pd

import config


def load_prices() -> pd.DataFrame:
    """Load filtered daily price data."""
    return pd.read_parquet(config.FILTERED_PRICES)


def load_monthly_prices() -> pd.DataFrame:
    """
    Return month-end resampled filtered prices.
    """
    daily = load_prices()
    return daily.resample("ME").last()


def load_returns() -> pd.DataFrame:
    """Load daily returns"""
    prices = load_prices()
    return prices.pct_change(fill_method=None)


def load_volumes() -> pd.DataFrame:
    """Load filtered daily volume data."""
    return pd.read_parquet(config.FILTERED_VOLUMES)


def load_low_prices() -> pd.DataFrame:
    """Load daily low prices."""
    return pd.read_parquet(config.FILTERED_LOW)


def load_high_prices() -> pd.DataFrame:
    """Load daily high prices."""
    return pd.read_parquet(config.FILTERED_HIGH)


def load_vix() -> pd.Series:
    """
    Load and forward-fill the VIX index for volatility context.
    Returns a Series aligned with the trading calendar.
    """
    vix = pd.read_parquet(config.VIX)
    return vix.reindex(load_prices().index).ffill()


def load_spy_prices() -> pd.Series:
    """
    Load SPY ETF closing prices for use as a benchmark.
    """
    return pd.read_parquet(config.SPY).squeeze()


def load_spy_returns() -> pd.Series:
    """
    Compute and return daily returns of the SPY ETF.
    """
    spy_prices = load_spy_prices()
    return spy_prices.pct_change().squeeze()


def load_labels() -> pd.Series:
    """Load saved trade outcomes (Y labels)."""
    return pd.read_parquet(config.Y).squeeze()


def load_features() -> pd.DataFrame:
    """Load saved feature matrix (X features)."""
    return pd.read_parquet(config.X)


def load_rates() -> pd.DataFrame:
    """Load the market yield on US Treasury 10 Year (DGS10)."""
    rates = pd.read_csv(config.DGS10)
    rates.index = pd.to_datetime(rates["observation_date"], format="%Y-%m-%d")
    rates = rates.reindex(load_prices().index).ffill()
    rates.drop(columns="observation_date", inplace=True)
    return rates
