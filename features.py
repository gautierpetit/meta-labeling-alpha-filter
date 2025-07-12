import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator

import config
from labeling import get_trade_outcomes
from strategy import get_daily_signals
from data_loader import (
    load_prices,
    load_monthly_prices,
    load_returns,
    load_volumes,
    load_low_prices,
    load_high_prices,
    load_spy_returns,
    load_vix,
)

logger = logging.getLogger(__name__)

def build_features() -> tuple[pd.DataFrame, pd.Series]:
    """
    Generate feature matrix X and binary outcome labels Y for meta-modeling.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix indexed by (date, ticker)
    Y : pd.Series
        Binary labels indicating trade success (1 = TP hit, 0 = SL hit)
    """

    # Load data
    prices = load_prices()
    monthly_prices = load_monthly_prices()
    returns = load_returns()
    volumes = load_volumes()
    low = load_low_prices()
    high = load_high_prices()
    spy_returns = load_spy_returns()
    vix = load_vix()

    # Signals & outcomes
    daily_signals, signal_dates = get_daily_signals(prices, monthly_prices)
    Y = get_trade_outcomes(prices, daily_signals)

    # Price-based features
    log_prices = np.log(prices)
    volatility_20d = prices.pct_change(fill_method=None).rolling(20).std()
    volatility_zscore = (volatility_20d - volatility_20d.mean()) / volatility_20d.std()
    momentum_12m_1m = prices.pct_change(252, fill_method=None) - prices.pct_change(21, fill_method=None)
    momentum_6m = prices.pct_change(126, fill_method=None)
    momentum_12m = prices.pct_change(252, fill_method=None)
    momentum_change = momentum_6m - momentum_12m
    vol_adj_momentum = momentum_12m / volatility_20d
    returns_1d = prices.pct_change(fill_method=None)
    returns_5d = prices.pct_change(5, fill_method=None)
    returns_20d = prices.pct_change(20, fill_method=None)
    price_max_1y = prices.rolling(252).max()
    price_min_1y = prices.rolling(252).min()
    price_percentile_1y = (prices - price_min_1y) / (price_max_1y - price_min_1y)

    # Broadcast VIX
    vix_feature = pd.DataFrame(
        np.tile(vix.values.reshape(-1, 1), (1, prices.shape[1])),
        index=prices.index,
        columns=prices.columns,
    )

    # Return autocorrelation
    serial_corr_5d = returns.rolling(5).apply(lambda x: x.autocorr(lag=1), raw=False)

    # Illiquidity
    amihud_illiquidity = returns.abs() / volumes
    illiquidity_zscore = (
        amihud_illiquidity - amihud_illiquidity.mean()
    ) / amihud_illiquidity.std()

    # Beta (60-day)
    rolling_beta = pd.DataFrame(index=prices.index, columns=prices.columns)
    for ticker in tqdm(prices.columns, desc="Computing Beta"):
        r = returns[ticker]
        cov = r.rolling(60).cov(spy_returns)
        var = spy_returns.rolling(60).var()
        rolling_beta[ticker] = cov / var

    # RSI
    rsi = prices.apply(lambda x: RSIIndicator(close=x, window=14).rsi())

    # Bollinger Z-score
    zscore = (prices - prices.rolling(20).mean()) / prices.rolling(20).std()

    # Calendar signals
    day_of_week_sin = pd.DataFrame(
        np.sin(2 * np.pi * prices.index.dayofweek / 7), index=prices.index
    )
    day_of_week_sin = pd.concat([day_of_week_sin] * len(prices.columns), axis=1)
    day_of_week_sin.columns = prices.columns

    month_of_year_sin = pd.DataFrame(
        np.sin(2 * np.pi * prices.index.month / 12), index=prices.index
    )
    month_of_year_sin = pd.concat([month_of_year_sin] * len(prices.columns), axis=1)
    month_of_year_sin.columns = prices.columns

    # ADX
    adx = pd.DataFrame(index=prices.index, columns=prices.columns)
    for col in prices.columns:
        adx[col] = ADXIndicator(
            high=high[col], low=low[col], close=prices[col], window=14
        ).adx()

    # Aggregate features
    features = {
        # Price Level
        "log_prices": log_prices,
        "returns_5d": returns_5d,
        "returns_20d": returns_20d,
        # Momentum
        "price_percentile_1y": price_percentile_1y,
        "momentum_12m_1m": momentum_12m_1m,
        # Volatility
        "volatility_20d": volatility_20d,
        "vix": vix_feature,
        # Correlation
        "serial_corr_5d": serial_corr_5d,
        "beta_60d": rolling_beta,
        # Liquidity
        "volume": volumes,
        "amihud_illiquidity": amihud_illiquidity.rolling(5).mean(),
        # Trend Strength and Structure
        "rsi_14d": rsi,
        "adx_14d": adx,
        "momentum_change": momentum_change,
        # Time-Based Signals
        "day_of_week_sin": day_of_week_sin,
        "month_of_year_sin": month_of_year_sin,
        # Reversal/Mean-Reversion Signals
        "bollinger_zscore": zscore,
        "returns_1d": returns_1d,
        "volatility_zscore": volatility_zscore,
        # Event driven
        # days_to_next_earnings
        # days_since_last_earnings
        # prev_earnings_surprise
        # Fundamental
        # market_cap
        # sector_dummy
        # Macro
        # spy_rolling_corr
        # SKEW index, MOVE index
        # US 10y treasury yield
        # Engineered
        "vol_adj_momentum": vol_adj_momentum,
        "illiquidity_zscore": illiquidity_zscore,
        # beta_momentum
        # momentum_persistence
    }

    # Build feature matrix X
    X_rows = []
    for date, ticker in tqdm(signal_dates, desc="Building X"):
        row = {"date": date, "ticker": ticker}
        for fname, fmat in features.items():
            value = (
                fmat.at[date, ticker]
                if (
                    date in fmat.index
                    and ticker in fmat.columns
                    and not pd.isna(fmat.at[date, ticker])
                )
                else np.nan
            )
            row[fname] = value
        X_rows.append(row)

    X = pd.DataFrame(X_rows).set_index(["date", "ticker"]).dropna()
    Y = Y.stack().dropna().astype(int)
    X = X.reindex(Y.index).dropna()
    Y = Y.loc[X.index]

    return X, Y


def main():
    """Build and save features and labels to disk."""
    X, Y = build_features()
    X.to_parquet(config.X)
    Y.to_frame().to_parquet(config.Y)
    logger.info(f"Saved features to {config.X}")
    logger.info(f"Saved labels to {config.Y}")


if __name__ == "__main__":
    main()
