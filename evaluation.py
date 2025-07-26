import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_loader import load_returns
import config
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)




def compute_drawdown(returns: pd.Series) -> pd.Series:
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative / running_max) - 1.0
    return drawdown

def plot_drawdown_underwater(
    returns: pd.Series,
    bench_returns: Optional[pd.Series] = None,
    label: tuple = ("Strategy", "Benchmark"),
    figsize: tuple = (12, 4),
    fixed_scale: bool = False,
    save: bool = True,
    file: str = "drawdown_underwater.png"
):
    """
    Plot drawdown under water graph for strategy and optional benchmark.
    
    Parameters:
    - returns (pd.Series): Strategy periodic returns (indexed by datetime).
    - bench_returns (pd.Series, optional): Benchmark returns.
    - label (tuple): Labels for the legend (strategy, benchmark).
    - figsize (tuple): Matplotlib figure size.
    - fixed_scale (bool): Whether to fix y-axis to [-100, 0].
    - save (bool): Whether to save the plot.
    - filename (str): Filename for saving the plot (requires config.RESULTS_DIR).
    """
    if not isinstance(returns, pd.Series):
        raise ValueError("Input 'returns' must be a pandas Series with datetime index.")

    strategy_dd = compute_drawdown(returns)

    plt.figure(figsize=figsize)
    plt.plot(strategy_dd * 100, label=f"{label[0]} Drawdown", color='steelblue')
    plt.fill_between(strategy_dd.index, strategy_dd * 100, 0, color='steelblue', alpha=0.3)

    if bench_returns is not None:
        benchmark_dd = compute_drawdown(bench_returns[returns.index[0]:])
        plt.plot(benchmark_dd * 100, label=f"{label[1]} Drawdown", color='darkorange', linestyle='--')

    plt.title("Drawdown Under Water")
    plt.ylabel("Drawdown (%)")
    plt.xlabel("Date")
    if fixed_scale:
        plt.ylim([-100, 0])
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
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
):
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
    plt.plot(cumulative.loc[start:], label=name)
    plt.plot(cumulative_costs.loc[start:], label=f"{name} with costs")
    plt.plot(
        spy_cumulative.loc[start:] / spy_cumulative.loc[start:].iloc[0],
        label="SPY",
        color="black"
    )
    plt.plot(
        mom_cumulative.loc[start:] / mom_cumulative.loc[start:].iloc[0],
        label="Momentum",
        color="red"
    )
    plt.suptitle(f"Cumulative Returns: {name} vs Benchmarks")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)

    if save:
        filename = config.RESULTS_DIR / f"{name.lower().replace(' ', '_')}_vs_spy.png"
        plt.savefig(filename)
        logger.info(f"Cumulative returns plot saved to: {filename}")
    plt.close()

def plot_turnover(
    turnover: pd.Series,
    name: str = "Strategy",
    window: int = 20,
    save: bool = True
):
    """
    Plots rolling average turnover.

    Parameters:
    - turnover: Daily turnover values.
    - name: Strategy name for title/filename.
    - window: Rolling window (default 20).
    - save: If True, saves the plot to disk.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(turnover.rolling(window).mean(), label=f"{window}-day avg turnover")
    plt.suptitle(f"Turnover Trend: {name}")
    plt.xlabel("Date")
    plt.ylabel("Turnover")
    plt.grid(True)
    plt.legend()

    if save:
        filename = config.RESULTS_DIR / f"{name.lower().replace(' ', '_')}_turnover_avg.png"
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


    startegy_corr = strategy_returns.rolling(window).corr(
        spy_returns
    )
    mom_corr = mom_returns.rolling(window).corr(
        spy_returns
    )

    plt.figure(figsize=figsize)
    plt.plot(startegy_corr, label=f"{name} vs SPY", color='steelblue')
    plt.plot(mom_corr, label="Momentum vs SPY", color='darkorange')
    plt.axhline(0, color='gray', linestyle='--', alpha=0.6)
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
    plt.plot(daily_leverage, label="Daily Leverage (Gross)",color='steelblue')
    plt.plot(net_exposure, label="Daily Net Exposure",color='grey')
    plt.axhline(daily_leverage.mean(), color="steelblue", linestyle="--", alpha=0.7, label=f"Average Leverage: {daily_leverage.mean():.2f}")
    plt.axhline(net_exposure.mean(), color="grey", linestyle="--", alpha=0.7, label=f"Average Net Exposure: {net_exposure.mean():.2f}")
    plt.title("Daily Leverage Over Time")
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

    # Optional: horizontal average lines
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
    pd.Series of PnLs per trade.
    """
    pnl_list = []
    trade_keys = []  # store (date, asset)

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
            trade_keys.append((entry_date, asset))  # <- save entry point

    return pd.Series(pnl_list, index=pd.MultiIndex.from_tuples(trade_keys, names=["date", "asset"]), name="Trade PnL")





def compute_alpha_beta(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> tuple[float, float]:
    """
    Computes CAPM-style alpha and beta from linear regression.

    Parameters
    ----------
    strategy_returns : pd.Series
        Daily returns of the strategy.
    benchmark_returns : pd.Series
        Daily returns of the benchmark (e.g., SPY).

    Returns
    -------
    alpha : float
        Annualized alpha (intercept × 252).
    beta : float
        Regression beta coefficient (sensitivity to benchmark).
    """
    df = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    X = df.iloc[:, 1].values.reshape(-1, 1)  # Benchmark
    y = df.iloc[:, 0].values  # Strategy

    reg = LinearRegression().fit(X, y)
    alpha = reg.intercept_ * 252
    beta = reg.coef_[0]

    return alpha, beta




def summarize_performance(
    returns: pd.Series,
    bench_spy:pd.Series,
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
    semi_vol = returns[returns<0].std() * np.sqrt(252)
    sharpe = annualized_return / annualized_vol if annualized_vol > 0 else np.nan
    sortino = annualized_return / semi_vol if semi_vol > 0 else np.nan
    max_drawdown = compute_drawdown(returns).min()
    avg_drawdown = compute_drawdown(returns).mean()
    var = returns.quantile(0.05)
    cvar = returns[returns <= var].mean()
    skew = returns.skew()
    kurt = returns.kurt()
    corr = returns.corr(bench_spy)
    corr_std = returns.rolling(20).corr(bench_spy).std()
    alpha, beta = compute_alpha_beta(returns, bench_spy)
    

    summary = pd.Series(
        {
            "Cumulative Return": f"{cumulative:.2%}",
            "Annualized Return": f"{annualized_return:.2%}",
            "Annualized Vol": f"{annualized_vol:.2%}",
            "Semi-volatility":f"{semi_vol:.2%}",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Sortino Ratio": f"{sortino:.2f}",
            #"Minimum Return": f"{returns.min():.2%}",
            #"Maximum Return": f"{returns.max():.2%}",
            "Max Drawdown": f"{max_drawdown:.2%}",
            "Avg Drawdown": f"{avg_drawdown:.2%}",
            #"Value at Risk (VaR)": f"{var:.2%}",
            "Conditional VaR (CVaR)": f"{cvar:.2%}",
            "Skew": f"{skew:.3f}",
            "Kurtosis": f"{kurt:.3f}",
            "Correlation to SPY": f"{corr:.3f}",
            "Std Dev Correlation": f"{corr_std:.3f}",
            "Alpha": f"{alpha:.3f}",
            "Beta": f"{beta:.3f}"
        }
    )

    if strategy:
        # Use weights > 0 as mask to identify *actually traded* signals
        if weights_df is not None:
            traded_mask = weights_df > 0
            traded_signals = filtered_signals[traded_mask]
        else:
            traded_signals = filtered_signals.copy()

        traded_idx = traded_signals.stack()[traded_signals.stack() == 1].index
        traded_outcomes = Y.loc[Y.index.isin(traded_idx)]
        
        total = len(traded_outcomes)
        
        aligned_weights = weights_df.stack().loc[traded_outcomes.index]

        long_mask = weights_df.stack() > 0
        short_mask = aligned_weights < 0
        long_wins = (traded_outcomes == 1) & long_mask
        short_wins = (traded_outcomes == -1) & short_mask
        long_total = long_mask.sum()
        short_total = short_mask.sum()
        # Win rate per side
        long_hit_rate = long_wins.sum() / long_total if long_total > 0 else np.nan
        short_hit_rate = short_wins.sum() / short_total if short_total > 0 else np.nan

        
        leverage = weights_df.abs().sum(axis=1) if weights_df is not None else pd.Series(np.nan, index=returns.index)

        net_exposure = weights_df.sum(axis=1)
        
        avg_holding_period = traded_signals.sum().mean()
        #avg_pnl = cumulative / total if total > 0 else "N/A"

        

        asset_returns = load_returns()
        trade_pnls = compute_pnl_per_trade(weights_df, filtered_signals, asset_returns)

        profit_factor = trade_pnls[trade_pnls > 0].sum() / -trade_pnls[trade_pnls < 0].sum()

        wins = (trade_pnls > 0).sum()

        trade_weights = weights_df.stack().loc[trade_pnls.index].abs()
        win_indicators = (trade_pnls > 0).astype(int)
        weighted_win_rate = (win_indicators * trade_weights).sum() / trade_weights.sum()


        summary["Avg Leverage (Gross)"] = round(leverage.mean(),2)
        summary["Max Leverage (Gross)"] = round(leverage.max(),2)
        summary["Avg Net Exposure"] = round(net_exposure.mean(),2)
        summary["Trade Count"] = total
        summary["Win Rate"] = f"{(wins / len(trade_pnls)):.2%}" if len(trade_pnls) > 0 else "N/A"
        summary["Avg Daily Turnover"] = round(turnover.mean(),2)
        summary["TP Trades"] = (traded_outcomes == 1).sum()
        summary["SL Trades"] = (traded_outcomes == -1).sum()
        summary["Timeout / No Signal"] = (traded_outcomes == 0).sum()
        summary["Avg Holding Period (days)"] = round(avg_holding_period, 2)
        summary["Notional-Weighted Win Rate"] = f"{weighted_win_rate:.2%}"
        summary["Avg PnL per Trade"] = f"{trade_pnls.mean():.2%}"
        summary["Median PnL per Trade"] = f"{trade_pnls.median():.2%}"
        summary["Profit Factor"] = f"{profit_factor:.2f}"
        summary["Long Hit Rate"] = f"{long_hit_rate:.2%}" if not np.isnan(long_hit_rate) else "N/A"
        summary["Short Hit Rate"] = f"{short_hit_rate:.2%}" if not np.isnan(short_hit_rate) else "N/A"

    return summary.astype(str)



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

    # Performance summaries
    summary = summarize_performance(
        strategy_returns,bench_spy, filtered_signals, Y, turnover, weights_df
    )
    summary_costs = summarize_performance(
        strategy_returns_w_costs,bench_spy, filtered_signals, Y, turnover, weights_df
    )
    summary_spy = summarize_performance(bench_spy.loc[start:],bench_spy, strategy=False)
    summary_mom = summarize_performance(bench_mom.loc[start:],bench_spy, strategy=False)

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

