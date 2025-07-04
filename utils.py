import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional, Tuple
from sklearn.base import ClassifierMixin
import config




def generate_momentum_signals(
    momentum_df: pd.DataFrame, top_quantile: float = config.TOP_QUANTILE
) -> pd.DataFrame:
    """
    Generate momentum signals monthly based on top quantile threshold.

    Parameters:
        momentum_df (pd.DataFrame): DataFrame of momentum scores.
        top_quantile (float): Quantile to define top-performing assets.

    Returns:
        pd.DataFrame: Binary signals with 1 for top decile performers.
    """
    signals = pd.DataFrame(index=momentum_df.index, columns=momentum_df.columns, data=0)
    for date in momentum_df.index:
        momentums = momentum_df.loc[date]
        threshold = momentums.quantile(top_quantile)
        selected = momentums[momentums >= threshold].index
        signals.loc[date, selected] = 1
    return signals


def summarize_performance(returns: pd.Series) -> pd.Series:
    """
    Summarize key performance metrics for a return series.

    Parameters:
        returns (pd.Series): Daily returns.

    Returns:
        pd.Series: Dictionary-style performance summary.
    """
    returns = returns.dropna()
    cumulative = (1 + returns).prod() - 1
    annualized_return = (1 + cumulative) ** (252 / len(returns)) - 1
    annualized_vol = returns.std() * np.sqrt(252)
    sharpe = annualized_return / annualized_vol
    cum_returns = (1 + returns).cumprod()
    max_drawdown = (cum_returns / cum_returns.cummax()).min() - 1
    var = returns.quantile(0.05)
    cvar = returns[returns <= var].mean()

    return pd.Series({
        "Cumulative Return": f"{cumulative:.2%}",
        "Annualized Return": f"{annualized_return:.2%}",
        "Annualized Vol": f"{annualized_vol:.2%}",
        "Sharpe Ratio": round(sharpe, 2),
        "Minimum Return": f"{returns.min():.2%}",
        "Maximum Return": f"{returns.max():.2%}",
        "Max Drawdown": f"{max_drawdown:.2%}",
        "Value at Risk (VaR)": f"{var:.2%}",
        "Conditional VaR (CVaR)": f"{cvar:.2%}",
    })


def get_trade_outcomes(
    prices: pd.DataFrame,
    daily_signals: pd.DataFrame,
    tp: float = config.TARGET_TP_THRESHOLD,
    sl: float = config.TARGET_SL_THRESHOLD
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


def filter_signals_with_meta_model(
    daily_signals: pd.DataFrame,
    clf: ClassifierMixin,
    X_test: pd.DataFrame,
    threshold: float = config.META_PROBA_THRESHOLD
) -> pd.DataFrame:
    """
    Filters signals using meta-model probability predictions.

    Parameters:
        daily_signals (pd.DataFrame): Initial binary signals.
        clf (ClassifierMixin): Trained classification model.
        X_test (pd.DataFrame): Meta features for predictions.
        threshold (float): Probability threshold for inclusion.

    Returns:
        pd.DataFrame: Filtered signals.
    """
    filtered_signals = pd.DataFrame(0, index=daily_signals.index, columns=daily_signals.columns)
    valid_idx = X_test.index.intersection(
        daily_signals.stack()[daily_signals.stack() == 1].index
    )
    X_valid = X_test.loc[valid_idx]
    probs = clf.predict_proba(X_valid)[:, 1]
    selected_idx = valid_idx[probs >= threshold]
    for date, ticker in selected_idx:
        filtered_signals.at[date, ticker] = 1
    return filtered_signals



def backtest_strategy(
    strategy_returns: pd.Series,
    bench_spy: pd.Series,
    bench_mom: pd.Series,
    name: str = "Strategy",
    trade_count: Optional[int] = None,
    win_rate: Optional[float] = None,
    start: str = config.BACKTEST_START_DATE,
    plot: bool = True,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Backtests a trading strategy against SPY and a standard momentum benchmark.

    Parameters
    ----------
    strategy_returns : pd.Series
        Daily returns of the strategy.
    bench_spy : pd.Series
        Daily returns of SPY for comparison.
    bench_mom : pd.Series
        Daily returns of a standard momentum strategy for comparison.
    name : str, optional
        Strategy name to be used in plot title and file name.
    trade_count : Optional[int], optional
        Total number of trades (for reporting), by default None.
    win_rate : Optional[float], optional
        Win rate of trades (for reporting), by default None.
    start : str, optional
        Start date for comparing SPY and momentum benchmarks, by default "2021-01-01".
    plot : bool, optional
        Whether to generate and save a cumulative return plot, by default True.

    Returns
    -------
    Tuple[pd.Series, pd.Series, pd.Series]
        Summary statistics for: (strategy, SPY, standard momentum)
    """

    # Cumulative returns
    cumulative = (1 + strategy_returns.fillna(0)).cumprod()
    spy_cumulative = (1 + bench_spy.fillna(0)).cumprod()
    mom_cumulative = (1 + bench_mom.fillna(0)).cumprod()

    # Performance summaries
    summary = summarize_performance(strategy_returns)
    summary_spy = summarize_performance(bench_spy.loc[start:])
    summary_mom = summarize_performance(bench_mom.loc[start:])

    # Optional additional info
    if trade_count is not None:
        summary["Trade Count"] = trade_count
    if win_rate is not None:
        summary["Win Rate"] = f"{win_rate:.2%}"

    # Plotting
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative, label=name)
        plt.plot(
            spy_cumulative.loc[start:] / spy_cumulative.loc[start:].iloc[0],
            label="SPY", color="black"
        )
        plt.plot(
            mom_cumulative.loc[start:] / mom_cumulative.loc[start:].iloc[0],
            label="Standard Momentum", color="red", linestyle="--"
        )
        plt.title(f"Cumulative Returns: {name} vs Benchmarks")
        plt.legend()
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            f"{config.FIGURES_DIR / name.lower().replace(' ', '_')}_vs_spy.png"
        )
        plt.close()

    return summary, summary_spy, summary_mom
