from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


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

    return pd.Series(
        {
            "Cumulative Return": f"{cumulative:.2%}",
            "Annualized Return": f"{annualized_return:.2%}",
            "Annualized Vol": f"{annualized_vol:.2%}",
            "Sharpe Ratio": round(sharpe, 2),
            "Minimum Return": f"{returns.min():.2%}",
            "Maximum Return": f"{returns.max():.2%}",
            "Max Drawdown": f"{max_drawdown:.2%}",
            "Value at Risk (VaR)": f"{var:.2%}",
            "Conditional VaR (CVaR)": f"{cvar:.2%}",
        }
    )


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

    # Plotting
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative.loc[start:], label=name)
        plt.plot(
            spy_cumulative.loc[start:] / spy_cumulative.loc[start:].iloc[0],
            label="SPY",
            color="black",
        )
        plt.plot(
            mom_cumulative.loc[start:] / mom_cumulative.loc[start:].iloc[0],
            label="Momentum",
            color="red",
        )
        plt.title(f"Cumulative Returns: {name} vs Benchmarks")
        plt.legend()
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.grid(True)
        plt.savefig(f"{config.FIGURES_DIR / name.lower().replace(' ', '_')}_vs_spy.png")
        plt.close()

    return summary, summary_spy, summary_mom
