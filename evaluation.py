import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_loader import load_returns
import config
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)




def compute_drawdown(returns: pd.Series) -> tuple[pd.Series, int]:
    """
    Compute the drawdown and maximum drawdown duration for a series of returns.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy.

    Returns
    -------
    tuple[pd.Series, int]
        - Drawdown series as a percentage.
        - Maximum drawdown duration in days.
    """
    if not isinstance(returns, pd.Series):
        raise TypeError("Input 'returns' must be a pandas Series.")

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative / running_max) - 1.0

    underwater = cumulative < running_max
    durations = underwater.groupby((underwater != underwater.shift()).cumsum()).cumsum()
    max_duration = int(durations.max()) if not durations.empty else 0

    return drawdown, max_duration


def plot_drawdown_underwater(
    returns: pd.Series,
    bench_returns: Optional[pd.Series] = None,
    label: tuple = ("Strategy", "Benchmark"),
    figsize: tuple = (12, 4),
    fixed_scale: bool = False,
    save: bool = True,
    file: str = "drawdown_underwater.png"
) -> None:
    """
    Plot drawdown under water graph for strategy and optional benchmark.

    Parameters
    ----------
    returns : pd.Series
        Strategy periodic returns (indexed by datetime).
    bench_returns : Optional[pd.Series], optional
        Benchmark returns, by default None.
    label : tuple, optional
        Labels for the legend (strategy, benchmark), by default ("Strategy", "Benchmark").
    figsize : tuple, optional
        Matplotlib figure size, by default (12, 4).
    fixed_scale : bool, optional
        Whether to fix y-axis to [-100, 0], by default False.
    save : bool, optional
        Whether to save the plot, by default True.
    file : str, optional
        Filename for saving the plot (requires config.RESULTS_DIR), by default "drawdown_underwater.png".

    Returns
    -------
    None
    """
    if not isinstance(returns, pd.Series):
        raise ValueError("Input 'returns' must be a pandas Series with datetime index.")

    strategy_dd, strategy_dd_duration = compute_drawdown(returns)

    plt.figure(figsize=figsize)
    plt.plot(strategy_dd * 100, label=f"{label[0]} Drawdown", color='steelblue')
    plt.fill_between(strategy_dd.index, strategy_dd * 100, 0, color='steelblue', alpha=0.3)

    if bench_returns is not None:
        benchmark_dd, benchmark_dd_duration = compute_drawdown(bench_returns.loc[returns.index[0]:])
        plt.plot(benchmark_dd * 100, label=f"{label[1]} Drawdown", color='darkorange', linestyle='--')

    plt.title("Drawdown Under Water")
    plt.ylabel("Drawdown (%)")
    plt.xlabel("Date")
    if fixed_scale:
        plt.ylim([-100, 0])
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.gca().text(
        0.01, 0.35,
        f"Max DD: {strategy_dd.min():.2%}\n"
        f"Avg DD: {strategy_dd.mean():.2%}\n"
        f"Max Duration: {strategy_dd_duration} days",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.7)
    )

    plt.tight_layout()

    if save:
        filename = config.RESULTS_DIR / file
        plt.savefig(filename)
        logger.info(f"Drawdown plot saved to: {filename}")

    plt.close()

def plot_cumulative_returns(
    strategy: pd.Series,
    strategy_costs: pd.Series,
    spy: pd.Series,
    mom: pd.Series,
    name: str = "Strategy",
    start: str = config.BACKTEST_START_DATE,
    save: bool = True,
    file: str = "cumulative_returns.png"
) -> None:
    """
    Plots cumulative returns of the strategy vs SPY and a standard momentum benchmark.

    Parameters:
    - strategy: Raw strategy returns (no costs).
    - strategy_costs: Strategy returns with transaction costs.
    - spy: SPY benchmark returns.
    - mom: Momentum benchmark returns.
    - name: Label for the strategy.
    - start: Start date for plotting.
    - save: If True, saves the plot to disk.
    """
    # Compute cumulative returns
    cumulative = (1 + strategy.fillna(0)).cumprod()
    cumulative_costs = (1 + strategy_costs.fillna(0)).cumprod()
    spy_cumulative = (1 + spy.fillna(0)).cumprod()
    mom_cumulative = (1 + mom.fillna(0)).cumprod()

    plt.figure(figsize=(12, 6))
    plt.plot(cumulative.loc[start:], label=f"{name}, Cumulative: {cumulative.loc[start:].iloc[-1]:.2f}", color='steelblue')
    plt.plot(cumulative_costs.loc[start:], label=f"{name} net, Cumulative: {cumulative_costs.loc[start:].iloc[-1]:.2f}", color='grey')
    plt.plot(
        spy_cumulative.loc[start:] / spy_cumulative.loc[start:].iloc[0],
        label=f"SPY, Cumulative: {spy_cumulative.loc[start:].iloc[-1]:.2f}",
        color="black"
    )
    plt.plot(
        mom_cumulative.loc[start:] / mom_cumulative.loc[start:].iloc[0],
        label=f"Momentum, Cumulative: {mom_cumulative.loc[start:].iloc[-1]:.2f}",
        color="darkorange"
    )
    plt.suptitle(f"Cumulative Returns: {name} vs Benchmarks")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)

    if save:
        filename = config.RESULTS_DIR / file
        plt.savefig(filename)
        logger.info(f"Cumulative returns plot saved to: {filename}")
    plt.close()

def plot_turnover(
    turnover: pd.Series,
    name: str = "Strategy",
    window: int = 5,
    save: bool = True,
    file: str = "avg_turnover.png"
) -> None:
    """
    Plots rolling average turnover.

    Parameters:
    - turnover: Daily turnover values.
    - name: Strategy name for title/filename.
    - window: Rolling window (default 20).
    - save: If True, saves the plot to disk.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(turnover.rolling(window).mean(), label=f"{window}-day avg turnover, Max: {turnover.max():.2f}")
    plt.axhline(turnover.mean(), color="steelblue", linestyle="--", alpha=0.7, label=f"Average Turnover: {turnover.mean():.2f}")
    plt.suptitle(f"Turnover Trend: {name}")
    plt.title(f"Total turnover: {turnover.sum():.2f}")
    plt.xlabel("Date")
    plt.ylabel("Turnover")
    plt.grid(True)
    plt.legend()

    if save:
        filename = config.RESULTS_DIR / file
        plt.savefig(filename)
        logger.info(f"Turnover plot saved to: {filename}")
    plt.close()


def plot_rolling_correlation(strategy_returns: pd.Series,
                              spy_returns: pd.Series,
                              mom_returns: pd.Series,
                              window: int = 20,
                              name: str = "Strategy",
                              figsize: tuple = (12, 4),
                              save: bool = True,
                              file: str = "rolling_corr.png") -> None:
    """
    Plot rolling correlation between strategy returns and benchmark returns.

    Parameters
    ----------
    strategy_returns : pd.Series
        Daily returns of the strategy.
    benchmark_returns : pd.Series
        Daily returns of the benchmark (e.g., SPY).
    window : int
        Rolling window size in days (default is 63 ~ 3 months).
    name : str
        Name of the strategy (for plot label).
    bench_name : str
        Name of the benchmark (for plot label).
    figsize : tuple
        Figure size.
    save : bool
        Whether to save the plot to disk.
    filename : str
        Filename for saving.
    """


    strategy_corr_rolling = strategy_returns.rolling(window).corr(
        spy_returns
    )
    mom_corr_rolling = mom_returns.rolling(window).corr(
        spy_returns
    )

    strategy_corr = strategy_returns.corr(spy_returns)
    mom_corr = mom_returns.corr(spy_returns)

    plt.figure(figsize=figsize)
    plt.plot(strategy_corr_rolling, label=f"{name}, Std Dev: {strategy_corr_rolling.std():.2f}", color='steelblue')
    plt.plot(mom_corr_rolling, label=f"Momentum, Std Dev: {mom_corr_rolling.std():.2f}", color='darkorange')
    plt.axhline(strategy_corr, color='steelblue', linestyle='--', label=f"Overall Correlation: {strategy_corr:.2f}", alpha=0.6)
    plt.axhline(mom_corr, color='darkorange', linestyle='--', label=f"Overall Correlation: {mom_corr:.2f}", alpha=0.6)
    plt.title(f"{window}-Day Rolling Correlation to SPY")
    plt.ylabel("Correlation")
    plt.xlabel("Date")
    plt.ylim([-1,1])
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    if save:
        filename = config.RESULTS_DIR / file
        plt.savefig(filename)
        logger.info(f"Correlation plot saved to: {filename}")
    plt.close()

def plot_leverage(weights_df: pd.DataFrame,
                name: str = "Strategy",
                figsize: tuple = (12, 4),
                save: bool = True,
                file: str = "daily_leverage.png") -> None:
    """
    Plot daily leverage (sum of absolute weights).
    """
    daily_leverage = weights_df.abs().sum(axis=1)
    net_exposure = weights_df.sum(axis=1)
    plt.figure(figsize=figsize)
    plt.plot(daily_leverage, label=f"Daily Leverage (Gross): Max {daily_leverage.max():.2f}",color='steelblue')
    plt.plot(net_exposure, label=f"Daily Net Exposure: Max {net_exposure.max():.2f}",color='grey')
    plt.axhline(daily_leverage.mean(), color="steelblue", linestyle="--", alpha=0.7, label=f"Average Leverage: {daily_leverage.mean():.2f}")
    plt.axhline(net_exposure.mean(), color="grey", linestyle="--", alpha=0.7, label=f"Average Net Exposure: {net_exposure.mean():.2f}")
    plt.title("Daily Leverage: " + name)
    plt.ylabel("Leverage")
    plt.xlabel("Date")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    if save:
        filename = config.RESULTS_DIR / file
        plt.savefig(filename)
        logger.info(f"Leverage plot saved to: {filename}")
    plt.close()
    

def plot_rolling_sharpe(strategy_returns: pd.Series,
                        mom_returns: pd.Series,
                        window: int = 60,
                        name: str = "Strategy",
                        figsize: tuple = (12, 4),
                        save: bool = True,
                        method: str = "compound",
                        file: str = "rolling_sharpe.png") -> None:
    """
    Plot rolling Sharpe ratio of the strategy and a benchmark.

    Parameters
    ----------
    strategy_returns : pd.Series
        Daily returns of the strategy.
    mom_returns : pd.Series
        Daily returns of the momentum benchmark.
    window : int
        Rolling window size in days.
    name : str
        Name of the strategy.
    figsize : tuple
        Size of the plot.
    save : bool
        If True, saves the plot.
    file : str
        File name to save.
    """

    r_sharpe = rolling_sharpe(strategy_returns, window, method=method)
    r_sharpe_mom = rolling_sharpe(mom_returns, window, method=method)

    plt.figure(figsize=figsize)
    plt.plot(r_sharpe, label=f"{name}", color='steelblue')
    plt.plot(r_sharpe_mom, label="Momentum", color='darkorange')

    
    plt.axhline(r_sharpe.mean(), color='steelblue', linestyle='--', alpha=0.7, label=f"Avg Sharpe {name}: {round(r_sharpe.mean(),2)}")
    plt.axhline(r_sharpe_mom.mean(), color='darkorange', linestyle='--', alpha=0.7, label=f"Avg Sharpe Momentum: {round(r_sharpe_mom.mean(),2)}")

    plt.title(f"{window}-Day Rolling Sharpe")
    plt.ylabel("Sharpe Ratio")
    plt.xlabel("Date")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    if save:
        filename = config.RESULTS_DIR / file
        plt.savefig(filename)
        logger.info(f"Rolling Sharpe plot saved to: {filename}")

    plt.close()



def rolling_sharpe(returns: pd.Series, window: int, method: str = "compound") -> pd.Series:
    """
    Calculate rolling Sharpe ratio.

    Parameters:
    - returns: pd.Series of daily returns
    - window: rolling window size (in days)
    - method: 
        - "compound": uses compound annualized return over the window
        - "simple": uses mean return over window, annualized

    Returns:
    - pd.Series of rolling Sharpe ratios
    """
    if method not in ["compound", "simple"]:
        raise ValueError("method must be 'compound' or 'simple'")

    if method == "compound":
        def compound_annual(x):
            if len(x) == 0:
                return np.nan
            cumulative = (1 + x).prod() - 1
            return (1 + cumulative) ** (252 / len(x)) - 1

        rolling_annual_return = returns.rolling(window).apply(compound_annual, raw=False)
    else:  # simple
        daily_mean = returns.rolling(window).mean()
        rolling_annual_return = daily_mean * 252

    rolling_annual_vol = returns.rolling(window).std() * np.sqrt(252)
    sharpe = rolling_annual_return / rolling_annual_vol
    return sharpe


def compute_pnl_per_trade(weights_df: pd.DataFrame,
                          filtered_signals: pd.DataFrame,
                          returns: pd.DataFrame) -> pd.Series:
    """
    Efficiently computes realized PnL per trade.

    Parameters
    ----------
    weights_df : pd.DataFrame
        Portfolio weights (including 0 during no position).
    filtered_signals : pd.DataFrame
        Entry signals where 1 indicates trade entry.
    returns : pd.DataFrame
        Daily returns per asset.

    Returns
    -------
    pd.Series
        Realized PnL per trade, indexed by (date, asset).
    """
    if not all(isinstance(df, pd.DataFrame) for df in [weights_df, filtered_signals, returns]):
        raise TypeError("All inputs must be pandas DataFrames.")

    pnl_list = []
    trade_keys = []  # Store (date, asset)

    for asset in filtered_signals.columns:
        entries = filtered_signals.index[filtered_signals[asset] == 1]
        for entry_date in entries:
            if entry_date not in weights_df.index:
                continue

            weight = weights_df.at[entry_date, asset]
            if weight == 0:
                continue

            future_weights = weights_df.loc[entry_date:, asset]
            zero_mask = (future_weights == 0)
            exit_date = zero_mask.idxmax() if zero_mask.any() else future_weights.index[-1]

            trade_weights = weights_df.loc[entry_date:exit_date, asset]
            trade_returns = returns.loc[entry_date:exit_date, asset]

            trade_pnl = (trade_weights * trade_returns).sum()
            pnl_list.append(trade_pnl)
            trade_keys.append((entry_date, asset))

    return pd.Series(pnl_list, index=pd.MultiIndex.from_tuples(trade_keys, names=["date", "asset"]), name="Trade PnL")


def plot_alpha_beta(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    name: str = "Strategy",
    bench_name: str = "SPY",
    figsize: tuple = (6, 6),
    save: bool = True,
    file: str = "alpha_beta_regression.png",
    plot: bool = False
) -> tuple[float, float]:
    """
    Plot strategy returns vs. benchmark with CAPM-style regression line.

    Parameters
    ----------
    strategy_returns : pd.Series
        Daily returns of the strategy.
    benchmark_returns : pd.Series
        Daily returns of the benchmark (e.g., SPY).
    name : str, optional
        Label for the strategy, by default "Strategy".
    bench_name : str, optional
        Label for the benchmark, by default "SPY".
    figsize : tuple, optional
        Size of the plot, by default (6, 6).
    save : bool, optional
        Whether to save the figure, by default True.
    file : str, optional
        Output filename (inside config.RESULTS_DIR), by default "alpha_beta_regression.png".
    plot : bool, optional
        Whether to display the plot, by default False.

    Returns
    -------
    tuple[float, float]
        Annualized alpha and beta of the strategy relative to the benchmark.
    """
    if not isinstance(strategy_returns, pd.Series) or not isinstance(benchmark_returns, pd.Series):
        raise TypeError("Both 'strategy_returns' and 'benchmark_returns' must be pandas Series.")

    df = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    x = df.iloc[:, 1].values.reshape(-1, 1)  # Benchmark
    y = df.iloc[:, 0].values  # Strategy

    reg = LinearRegression().fit(x, y)
    alpha = reg.intercept_ * 252  # Annualized
    beta = reg.coef_[0]

    if plot:
        x_pred = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
        y_pred = reg.predict(x_pred)

        plt.figure(figsize=figsize)
        plt.scatter(x, y, alpha=0.3, s=10, label="Daily Returns")
        plt.plot(x_pred, y_pred, color='red', label=f"Fit: y = {beta:.2f}x + {alpha:.2f}")
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)
        plt.axvline(0, color='gray', linestyle='--', linewidth=1)

        plt.xlabel(f"{bench_name} Daily Returns")
        plt.ylabel(f"{name} Daily Returns")
        plt.title(f"{name} vs. {bench_name} – Alpha/Beta Regression")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        if save:
            filename = config.RESULTS_DIR / file
            plt.savefig(filename)
            logger.info(f"Alpha-Beta plot saved to: {filename}")

        plt.close()

    return alpha, beta




def summarize_performance(
    returns: pd.Series,
    bench_spy: pd.Series,
    filtered_signals: Optional[pd.DataFrame] = None,
    Y: Optional[pd.Series] = None,
    turnover: Optional[pd.Series] = None,
    weights_df: Optional[pd.DataFrame] = None,
    strategy: bool = True,
) -> pd.Series:
    returns = returns.dropna()
    cumulative = (1 + returns).prod() - 1
    annualized_return = (1 + cumulative) ** (252 / len(returns)) - 1
    annualized_vol = returns.std() * np.sqrt(252)
    semi_vol = returns[returns < 0].std() * np.sqrt(252)
    sharpe = annualized_return / annualized_vol if annualized_vol > 0 else np.nan
    sortino = annualized_return / semi_vol if semi_vol > 0 else np.nan
    drawdown, drawdown_duration = compute_drawdown(returns)
    max_drawdown = drawdown.min()
    avg_drawdown = drawdown.mean()
    var = returns.quantile(0.05)
    cvar = returns[returns <= var].mean()
    skew = returns.skew()
    kurt = returns.kurt()
    corr = returns.corr(bench_spy)
    #corr_std = returns.rolling(20).corr(bench_spy).std()
    alpha, beta = plot_alpha_beta(returns, bench_spy)
    monthly_returns = returns.resample("M").sum()



    summary = pd.Series(
        {
            #"Cumulative Return": f"{cumulative:.2%}",
            "Annualized Return": f"{annualized_return:.2%}",
            "Annualized Vol": f"{annualized_vol:.2%}",
            "Semi-volatility": f"{semi_vol:.2%}",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Sortino Ratio": f"{sortino:.2f}",
            "Conditional VaR (CVaR)": f"{cvar:.2%}",

            "Max Drawdown": f"{max_drawdown:.2%}",
            "Avg Drawdown": f"{avg_drawdown:.2%}",
            "Max Drawdown Duration (days)": drawdown_duration,
            "Skew": f"{skew:.3f}",
            "Kurtosis": f"{kurt:.3f}",

            "Correlation to SPY": f"{corr:.3f}",
            # "Std Dev Correlation": f"{corr_std:.3f}",
            "Alpha": f"{alpha:.3f}",
            "Beta": f"{beta:.3f}",
            "Positive Months": f"{(monthly_returns > 0).sum()}",
            "Negative Months": f"{(monthly_returns < 0).sum()}",

        }
    )

    if strategy and filtered_signals is not None and Y is not None:
        # Apply trading mask if weights_df is available
        traded_signals = filtered_signals.copy()
        if weights_df is not None:
            traded_signals[weights_df == 0] = 0

        stacked_trades = traded_signals.stack()
        traded_idx = stacked_trades[stacked_trades != 0].index
        traded_outcomes = Y.loc[Y.index.isin(traded_idx)]

        total = len(traded_outcomes)

        aligned_weights = (
            weights_df.stack().loc[traded_outcomes.index]
            if weights_df is not None
            else pd.Series(np.nan, index=traded_outcomes.index)
        )
        assert aligned_weights.index.equals(traded_outcomes.index)

        long_mask = aligned_weights > 0
        short_mask = aligned_weights < 0

        long_trades = traded_outcomes[long_mask]
        short_trades = traded_outcomes[short_mask]

        long_hit_rate = (long_trades == 1).sum() / len(long_trades) if len(long_trades) > 0 else np.nan
        short_hit_rate = (short_trades == -1).sum() / len(short_trades) if len(short_trades) > 0 else np.nan


        # Leverage and Exposure
        """if weights_df is not None:
            leverage = weights_df.abs().sum(axis=1)
            net_exposure = weights_df.sum(axis=1)
        else:
            leverage = pd.Series(np.nan, index=returns.index)
            net_exposure = pd.Series(np.nan, index=returns.index)"""

        # Holding period: average streak of nonzero entries in traded_signals
        holding_periods = []
        for _, ticker_signals in traded_signals.items():
            active = (ticker_signals != 0).astype(int)
            streak = 0
            for val in active:
                if val == 1:
                    streak += 1
                else:
                    if streak > 0:
                        holding_periods.append(streak)
                    streak = 0
            if streak > 0:
                holding_periods.append(streak)
        avg_holding_period = np.mean(holding_periods) if holding_periods else 0.0

        # Trade PnL stats
        asset_returns = load_returns()
        trade_pnls = compute_pnl_per_trade(weights_df, filtered_signals, asset_returns)

        total_wins = (trade_pnls > 0).sum()
        #total_losses = (trade_pnls < 0).sum()
        losses_sum = -trade_pnls[trade_pnls < 0].sum()
        profit_factor = (
            trade_pnls[trade_pnls > 0].sum() / losses_sum if losses_sum > 0 else np.nan
        )

        # Weighted win rate
        trade_weights = weights_df.stack().loc[trade_pnls.index].abs()
        win_indicators = (trade_pnls > 0).astype(int)
        weighted_win_rate = (
            (win_indicators * trade_weights).sum() / trade_weights.sum()
            if trade_weights.sum() > 0
            else np.nan
        )

        #summary["Avg Leverage (Gross)"] = round(leverage.mean(), 2)
        #summary["Max Leverage (Gross)"] = round(leverage.max(), 2)
        #summary["Avg Net Exposure"] = round(net_exposure.mean(), 2)
        summary["Trade Count"] = total
        summary["Win Rate"] = f"{(total_wins / len(trade_pnls)):.2%}" if len(trade_pnls) > 0 else "N/A"
        #summary["Avg Daily Turnover"] = round(turnover.mean(), 2) if turnover is not None else np.nan
        #summary["Total Daily Turnover"] = round(turnover.sum(), 2) if turnover is not None else np.nan
        summary["TP Trades"] = (traded_outcomes == 1).sum()
        summary["SL Trades"] = (traded_outcomes == -1).sum()
        summary["Timeout / No Signal"] = (traded_outcomes == 0).sum()
        summary["Avg Holding Period (days)"] = round(avg_holding_period, 2)
        summary["Notional-Weighted Win Rate"] = f"{weighted_win_rate:.2%}" if not np.isnan(weighted_win_rate) else "N/A"
        summary["Avg PnL per Trade"] = f"{trade_pnls.mean():.2%}"
        
        summary["Median PnL per Trade"] = f"{trade_pnls.median():.2%}"
        summary["Profit Factor"] = f"{profit_factor:.2f}" if not np.isnan(profit_factor) else "N/A"
        summary["Long Hit Rate"] = f"{long_hit_rate:.2%}" if not np.isnan(long_hit_rate) else "N/A"
        summary["Short Hit Rate"] = f"{short_hit_rate:.2%}" if not np.isnan(short_hit_rate) else "N/A"

    return summary.astype(str)




def backtest_strategy(
    strategy_returns: pd.Series,
    strategy_returns_w_costs: pd.Series,
    turnover: pd.Series,
    bench_spy: pd.Series,
    bench_mom: pd.Series,
    bench_mom_ls: pd.Series,
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

    # Performance summaries
    summary = summarize_performance(
        strategy_returns,bench_spy, filtered_signals, Y, turnover, weights_df
    )
    summary_costs = summarize_performance(
        strategy_returns_w_costs,bench_spy, filtered_signals, Y, turnover, weights_df
    )
    summary_spy = summarize_performance(bench_spy.loc[start:],bench_spy, strategy=False)
    summary_mom = summarize_performance(bench_mom.loc[start:],bench_spy, strategy=False)
    summary_mom_ls = summarize_performance(bench_mom_ls.loc[start:],bench_spy, strategy=False)

    # Plotting
    if plot:
        plot_cumulative_returns(
            strategy_returns,
            strategy_returns_w_costs,
            bench_spy,
            bench_mom,
            name=name,
            start=start,
            save=save
        )
        plot_turnover(turnover, name=name, save=save)
        plot_drawdown_underwater(strategy_returns,bench_mom, (name,"Standard Momentum"),save=save,file="drawdown_vs_mom.png")
        plot_drawdown_underwater(strategy_returns,save=save)
        plot_rolling_correlation(strategy_returns,bench_spy,bench_mom, save=save)
        plot_leverage(weights_df,name)
        plot_rolling_sharpe(strategy_returns,bench_mom,method="compound",save=save)
        plot_alpha_beta(strategy_returns, bench_spy,plot=True, name=name, save=save)


    summary_df = pd.concat([summary, summary_costs, summary_spy, summary_mom, summary_mom_ls], axis=1)
    summary_df.columns = [
        f"{name} (Gross)",
        f"{name} (Net)",
        "SPY",
        "Standard Momentum",
        "Standard Momentum (Long Short)"
    ]
    if save:
        summary_df.to_excel(config.PERFORMANCE_SUMMARY_XLSX)
        logger.info(f"Backtest summary saved to: {config.PERFORMANCE_SUMMARY_XLSX}")

    return summary_df

