"""
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

Convenience data loaders for project assets (prices, volumes, features, labels).

These functions centralize file I/O and basic alignment logic so callers can
assume consistent indices and dtypes. They favor explicit typing and simple
error messages when files are missing.
"""

import logging
from pathlib import Path

import pandas as pd

import src.config as config

logger = logging.getLogger(__name__)


def load_prices() -> pd.DataFrame:
    """Load filtered daily price DataFrame from disk.

    Returns:
        pd.DataFrame: daily price matrix indexed by trading date with
        tickers as columns.

    Raises:
        FileNotFoundError: If the configured parquet file does not exist.
    """
    path = Path(config.FILTERED_PRICES)
    if not path.exists():
        logger.error("Filtered prices file not found: %s", path)
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def load_monthly_prices() -> pd.DataFrame:
    """Return month-end resampled filtered prices.

    Uses calendar month-end periods and takes the last available price in
    each month.
    """
    daily = load_prices()
    return daily.resample("M").last()


def load_returns() -> pd.DataFrame:
    """Compute daily returns from filtered prices.

    Returns a DataFrame of the same shape as prices where each value is the
    simple return from the previous trading day.
    """
    prices = load_prices()
    return prices.pct_change()


def load_volumes() -> pd.DataFrame:
    """Load filtered daily volumes from disk.

    Returns:
        pd.DataFrame: volume matrix indexed by trading date.
    """
    path = Path(config.FILTERED_VOLUMES)
    if not path.exists():
        logger.error("Filtered volumes file not found: %s", path)
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def load_low_prices() -> pd.DataFrame:
    """Load daily low prices.

    Returns:
        pd.DataFrame: low price matrix.
    """
    path = Path(config.FILTERED_LOW)
    if not path.exists():
        logger.error("Filtered low prices file not found: %s", path)
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def load_high_prices() -> pd.DataFrame:
    """Load daily high prices.

    Returns:
        pd.DataFrame: high price matrix.
    """
    path = Path(config.FILTERED_HIGH)
    if not path.exists():
        logger.error("Filtered high prices file not found: %s", path)
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def load_vix() -> pd.Series:
    """Load VIX closing series and align to the project trading calendar.

    Returns:
        pd.Series: VIX series indexed by the same dates as filtered prices.
    """
    path = Path(config.VIX)
    if not path.exists():
        logger.error("VIX file not found: %s", path)
        raise FileNotFoundError(path)
    vix = pd.read_parquet(path)
    # Ensure it's a Series
    if isinstance(vix, pd.DataFrame):
        vix = vix.squeeze()
    return vix.reindex(load_prices().index).ffill()


def load_spy_prices() -> pd.Series:
    """Load SPY ETF closing prices (benchmark).

    Returns a pandas Series of SPY close prices.
    """
    path = Path(config.SPY)
    if not path.exists():
        logger.error("SPY file not found: %s", path)
        raise FileNotFoundError(path)
    spy = pd.read_parquet(path)
    return spy.squeeze()


def load_spy_returns() -> pd.Series:
    """Return daily simple returns for SPY.

    Returns:
        pd.Series: SPY daily returns aligned to the trading calendar.
    """
    spy_prices = load_spy_prices()
    return spy_prices.pct_change().squeeze()


def load_labels() -> pd.Series:
    """Load saved outcome labels (Y).

    Returns:
        pd.Series: labels indexed by the sample index used in the project.
    """
    path = Path(config.Y)
    if not path.exists():
        logger.error("Labels file not found: %s", path)
        raise FileNotFoundError(path)
    return pd.read_parquet(path).squeeze()


def load_features() -> pd.DataFrame:
    """Load saved feature matrix (X features).

    Returns:
        pd.DataFrame: feature matrix used for modeling.
    """
    path = Path(config.X)
    if not path.exists():
        logger.error("Features file not found: %s", path)
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def load_rates() -> pd.DataFrame:
    """Load macro rate series (10Y and 10Y-3M) and align to trading calendar.

    Returns:
        pd.DataFrame: columns for the raw 10Y yield and the 10Y-3M spread (or
        the files provided by the configuration).
    """
    path_10y = Path(config.DGS10)
    path_spread = Path(config.T10Y3M)

    if not path_10y.exists() or not path_spread.exists():
        logger.error("Rate files missing: %s, %s", path_10y, path_spread)
        raise FileNotFoundError((path_10y, path_spread))

    ten_year = pd.read_csv(path_10y)
    ten_year.index = pd.to_datetime(ten_year["observation_date"], format="%Y-%m-%d")
    ten_year = ten_year.reindex(load_prices().index).ffill()
    ten_year = ten_year.drop(columns="observation_date")

    ten_year_minus = pd.read_csv(path_spread)
    ten_year_minus.index = pd.to_datetime(ten_year_minus["observation_date"], format="%Y-%m-%d")
    ten_year_minus = ten_year_minus.reindex(load_prices().index).ffill()
    ten_year_minus = ten_year_minus.drop(columns="observation_date")

    return pd.concat([ten_year, ten_year_minus], axis=1)
