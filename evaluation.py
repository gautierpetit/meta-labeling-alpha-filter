import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


def summarize_performance(
    returns: pd.Series,
    filtered_signals: Optional[pd.DataFrame] = None,
    Y: Optional[pd.Series] = None,
    turnover: Optional[pd.Series] = None,
    weights_df: Optional[pd.DataFrame] = None,
    strategy: bool = True,
) -> pd.Series:
    """
    Summarize key performance metrics for a return series.

    Parameters:
        returns (pd.Series): Daily returns of the strategy.
        filtered_signals (Optional[pd.DataFrame]): Binary signal matrix (1s where trades occur).
        Y (Optional[pd.Series]): Meta-labels (+1, 0, -1 or binary).
        turnover (Optional[pd.Series]): Daily turnover values.
        weights_df (Optional[pd.DataFrame]): Weights matrix used for execution. Trades with zero weight are excluded.
        strategy (bool): Whether to compute strategy-specific metrics (e.g., win rate, trade count).

    Returns:
        pd.Series: Performance summary.
    """
    returns = returns.dropna()
    cumulative = (1 + returns).prod() - 1
    annualized_return = (1 + cumulative) ** (252 / len(returns)) - 1
    annualized_vol = returns.std() * np.sqrt(252)
    sharpe = annualized_return / annualized_vol if annualized_vol > 0 else np.nan
    cum_returns = (1 + returns).cumprod()
    max_drawdown = (cum_returns / cum_returns.cummax()).min() - 1
    var = returns.quantile(0.05)
    cvar = returns[returns <= var].mean()

    summary = pd.Series(
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

    if strategy and filtered_signals is not None and Y is not None:
        # Use weights > 0 as mask to identify *actually traded* signals
        if weights_df is not None:
            traded_mask = weights_df > 0
            traded_signals = filtered_signals[traded_mask]
        else:
            traded_signals = filtered_signals.copy()

        traded_idx = traded_signals.stack()[traded_signals.stack() == 1].index
        traded_outcomes = Y.loc[Y.index.isin(traded_idx)]

        resolved_trades = traded_outcomes[traded_outcomes != 0]
        wins = (resolved_trades == 1).sum()
        total = len(resolved_trades)

        summary["Trade Count"] = total
        summary["Win Rate"] = f"{(wins / total):.2%}" if total > 0 else "N/A"
        summary["Turnover"] = turnover.sum() if turnover is not None else "N/A"
        summary["TP Trades"] = (traded_outcomes == 1).sum()
        summary["SL Trades"] = (traded_outcomes == -1).sum()
        summary["Time Barrier Trades"] = (traded_outcomes == 0).sum()

        # --- New Metrics ---
        if total > 0:
            # Avg Holding Period (in days)
            avg_holding_period = traded_signals.sum().mean()
            summary["Avg Holding Period (days)"] = round(avg_holding_period, 2)

            # Avg PnL per Trade
            avg_pnl = cumulative / total
            summary["Avg PnL per Trade"] = f"{avg_pnl:.2%}"

            # Exposure-Weighted Win Rate
            if weights_df is not None:
                win_mask_df = (
                    resolved_trades.eq(1).unstack().reindex_like(weights_df).fillna(0)
                )
                weighted_win_rate = (
                    win_mask_df * weights_df
                ).sum().sum() / weights_df.sum().sum()

                summary["Exposure-Weighted Win Rate"] = f"{weighted_win_rate:.2%}"

    return summary


def backtest_strategy(
    strategy_returns: pd.Series,
    strategy_returns_w_costs: pd.Series,
    turnover: pd.Series,
    bench_spy: pd.Series,
    bench_mom: pd.Series,
    filtered_signals: pd.DataFrame,
    Y: pd.Series,
    weights_df: pd.DataFrame,
    name: str = "Strategy",
    start: str = config.BACKTEST_START_DATE,
    plot: bool = True,
    save: bool = True,
) -> pd.DataFrame:
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
    start : str, optional
        Start date for comparing SPY and momentum benchmarks, by default "2021-01-01".
    plot : bool, optional
        Whether to generate and save a cumulative return plot, by default True.
    filtered_signals : pd.DataFrame
        Binary signal matrix (1s where trades occur).
    Y : pd.Series
        Meta-labels (+1, 0, -1 or binary).
    weights_df: (Optional[pd.DataFrame]): Weights matrix used for execution. Trades with zero weight are excluded.

    Returns
    -------
    Tuple[pd.Series, pd.Series, pd.Series]
        Summary statistics for: (strategy, SPY, standard momentum)
    """

    # Cumulative returns
    cumulative = (1 + strategy_returns.fillna(0)).cumprod()
    cumulative_costs = (1 + strategy_returns_w_costs.fillna(0)).cumprod()
    spy_cumulative = (1 + bench_spy.fillna(0)).cumprod()
    mom_cumulative = (1 + bench_mom.fillna(0)).cumprod()

    # Performance summaries
    summary = summarize_performance(
        strategy_returns, filtered_signals, Y, turnover, weights_df
    )
    summary_costs = summarize_performance(
        strategy_returns_w_costs, filtered_signals, Y, turnover, weights_df
    )
    summary_spy = summarize_performance(bench_spy.loc[start:], strategy=False)
    summary_mom = summarize_performance(bench_mom.loc[start:], strategy=False)

    # Plotting
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative.loc[start:], label=name)
        plt.plot(cumulative_costs.loc[start:], label=f"{name} with costs")
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
        plt.suptitle(f"Cumulative Returns: {name} vs Benchmarks")
        plt.legend()
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.grid(True)
        plt.savefig(f"{config.FIGURES_DIR / name.lower().replace(' ', '_')}_vs_spy.png")
        plt.close()

        plt.figure(figsize=(12, 6))
        plt.plot(turnover.rolling(20).mean(), label="20-day avg turnover")
        plt.suptitle(f"Turnover Trend: {name}")
        plt.xlabel("Date")
        plt.ylabel("Turnover")
        plt.grid(True)
        plt.savefig(
            f"{config.FIGURES_DIR / name.lower().replace(' ', '_')}_turnover_avg"
        )
        plt.close()

    summary_df = pd.concat([summary, summary_costs, summary_spy, summary_mom], axis=1)
    summary_df.columns = [
        f"{name} (No Costs)",
        f"{name} (With Costs)",
        "SPY",
        "Standard Momentum",
    ]
    if save:
        summary_df.to_excel(config.PERFORMANCE_SUMMARY_XLSX)
        logger.info(f"Backtest summary saved to: {config.PERFORMANCE_SUMMARY_XLSX}")

    return summary_df
