"""Evaluation and plotting helpers for backtests and diagnostics.

This module contains utilities to compute drawdowns, rolling Sharpe,
PnL per trade and to render commonly used diagnostic plots. Functions
use explicit type hints and ensure saved outputs are written to
`config.RESULTS_DIR` with parent directories created automatically.
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import src.config as config
from src.data_loader import load_returns

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
    Tuple[pd.Series, int]
        - Drawdown series as a percentage.
        - Maximum drawdown duration in days.
    """
    if not isinstance(returns, pd.Series):
        raise TypeError("Input 'returns' must be a pandas Series.")

    cumulative: pd.Series = (1 + returns).cumprod()
    running_max: pd.Series = cumulative.cummax()
    drawdown: pd.Series = (cumulative / running_max) - 1.0

    underwater = cumulative < running_max
    durations = underwater.groupby((underwater != underwater.shift()).cumsum()).cumsum()
    max_duration = int(durations.max()) if not durations.empty else 0

    return drawdown, max_duration


def plot_drawdown_underwater(
    returns: pd.Series,
    bench_returns: pd.Series | None = None,
    label: tuple[str, str] = ("Strategy", "Benchmark"),
    figsize: tuple[int, int] = (12, 4),
    fixed_scale: bool = False,
    save: bool = True,
    file: str = "drawdown_underwater.png",
) -> None:
    """
    Plot drawdown under water graph for strategy and optional benchmark.

    Parameters
    ----------
    returns : pd.Series
        Strategy periodic returns (indexed by datetime).
    bench_returns : Optional[pd.Series], optional
        Benchmark returns, by default None.
    label : Tuple[str, str], optional
        Labels for the legend (strategy, benchmark), by default ("Strategy", "Benchmark").
    figsize : Tuple[int, int], optional
        Matplotlib figure size, by default (12, 4).
    fixed_scale : bool, optional
        Whether to fix y-axis to [-100, 0], by default False.
    save : bool, optional
        Whether to save the plot, by default True.
    file : str, optional
        Filename for saving the plot, by default "drawdown_underwater.png".

    Returns
    -------
    None
    """
    if not isinstance(returns, pd.Series):
        raise ValueError("Input 'returns' must be a pandas Series with datetime index.")

    strategy_dd, strategy_dd_duration = compute_drawdown(returns)

    plt.figure(figsize=figsize)
    plt.plot(strategy_dd * 100, label=f"{label[0]} Drawdown", color="steelblue")
    plt.fill_between(strategy_dd.index, strategy_dd * 100, 0, color="steelblue", alpha=0.3)

    if bench_returns is not None:
        benchmark_dd, benchmark_dd_duration = compute_drawdown(
            bench_returns.loc[returns.index[0] :]
        )
        plt.plot(
            benchmark_dd * 100,
            label=f"{label[1]} Drawdown",
            color="darkorange",
            linestyle="--",
        )

    plt.title("Drawdown Under Water")
    plt.ylabel("Drawdown (%)")
    plt.xlabel("Date")
    if fixed_scale:
        plt.ylim([-100, 0])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.gca().text(
        0.01,
        0.35,
        f"Max DD: {strategy_dd.min():.2%}\n"
        f"Avg DD: {strategy_dd.mean():.2%}\n"
        f"Max Duration: {strategy_dd_duration} days",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.7),
    )

    plt.tight_layout()

    if save:
        filename = Path(config.RESULTS_DIR) / file
        filename.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filename)
        logger.info("Drawdown plot saved to: %s", filename)

    plt.close()


def plot_cumulative_returns(
    strategy: pd.Series,
    strategy_costs: pd.Series,
    spy: pd.Series,
    mom: pd.Series,
    start: str,
    name: str = "Strategy",
    save: bool = True,
    file: str = "cumulative_returns.png",
) -> None:
    """
    Plots cumulative returns of the strategy vs SPY and a standard momentum benchmark.

    Parameters
    ----------
    strategy : pd.Series
        Raw strategy returns (no costs).
    strategy_costs : pd.Series
        Strategy returns with transaction costs.
    spy : pd.Series
        SPY benchmark returns.
    mom : pd.Series
        Momentum benchmark returns.
    start : str, optional
        Start date for plotting.
    name : str, optional
        Label for the strategy, by default "Strategy".

    save : bool, optional
        Whether to save the plot, by default True.
    file : str, optional
        Filename for saving the plot, by default "cumulative_returns.png".

    Returns
    -------
    None
    """
    # Compute cumulative returns
    cumulative: pd.Series = (1 + strategy.fillna(0)).cumprod()
    cumulative_costs: pd.Series = (1 + strategy_costs.fillna(0)).cumprod()
    spy_cumulative: pd.Series = (1 + spy.fillna(0)).cumprod()
    mom_cumulative: pd.Series = (1 + mom.fillna(0)).cumprod()

    plt.figure(figsize=(12, 6))
    plt.plot(
        cumulative.loc[start:] / cumulative.loc[start:].iloc[0],
        label=f"{name}, Cumulative: {cumulative.loc[start:].iloc[-1]:.2f}",
        color="steelblue",
    )
    plt.plot(
        cumulative_costs.loc[start:] / cumulative_costs.loc[start:].iloc[0],
        label=f"{name} net, Cumulative: {cumulative_costs.loc[start:].iloc[-1]:.2f}",
        color="grey",
    )
    plt.plot(
        spy_cumulative.loc[start:] / spy_cumulative.loc[start:].iloc[0],
        label=f"SPY, Cumulative: {spy_cumulative.loc[start:].iloc[-1]:.2f}",
        color="black",
    )
    plt.plot(
        mom_cumulative.loc[start:] / mom_cumulative.loc[start:].iloc[0],
        label=f"Momentum, Cumulative: {mom_cumulative.loc[start:].iloc[-1]:.2f}",
        color="darkorange",
    )
    plt.suptitle(f"Cumulative Returns: {name} vs Benchmarks")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)

    if save:
        filename = Path(config.RESULTS_DIR) / file
        filename.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filename)
        logger.info("Cumulative returns plot saved to: %s", filename)
    plt.close()


def plot_turnover(
    turnover: pd.Series,
    name: str = "Strategy",
    window: int = 5,
    save: bool = True,
    file: str = "avg_turnover.png",
) -> None:
    """
    Plots rolling average turnover.

    Parameters
    ----------
    turnover : pd.Series
        Daily turnover values.
    name : str, optional
        Strategy name for title/filename, by default "Strategy".
    window : int, optional
        Rolling window size, by default 5.
    save : bool, optional
        Whether to save the plot, by default True.
    file : str, optional
        Filename for saving the plot, by default "avg_turnover.png".

    Returns
    -------
    None
    """
    plt.figure(figsize=(12, 6))
    plt.plot(
        turnover.rolling(window).mean(),
        label=f"{window}-day avg turnover, Max: {turnover.max():.2f}",
    )
    plt.axhline(
        turnover.mean(),
        color="steelblue",
        linestyle="--",
        alpha=0.7,
        label=f"Average Turnover: {turnover.mean():.2f}",
    )
    plt.suptitle(f"Turnover Trend: {name}")
    plt.title(f"Total turnover: {turnover.sum():.2f}")
    plt.xlabel("Date")
    plt.ylabel("Turnover")
    plt.grid(True)
    plt.legend()

    if save:
        filename = Path(config.RESULTS_DIR) / file
        filename.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filename)
        logger.info("Turnover plot saved to: %s", filename)
    plt.close()


def plot_rolling_correlation(
    strategy_returns: pd.Series,
    spy_returns: pd.Series,
    mom_returns: pd.Series,
    window: int = 20,
    name: str = "Strategy",
    figsize: tuple[int, int] = (12, 4),
    save: bool = True,
    file: str = "rolling_corr.png",
) -> None:
    """
    Plot rolling correlation between strategy returns and benchmark returns.

    Parameters
    ----------
    strategy_returns : pd.Series
        Daily returns of the strategy.
    spy_returns : pd.Series
        Daily returns of the benchmark SPY.
    mom_returns : pd.Series
        Daily returns of the momentum benchmark.
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

    strategy_corr_rolling = strategy_returns.rolling(window).corr(spy_returns)
    mom_corr_rolling = mom_returns.rolling(window).corr(spy_returns)

    strategy_corr = strategy_returns.corr(spy_returns)
    mom_corr = mom_returns.corr(spy_returns)

    plt.figure(figsize=figsize)
    plt.plot(
        strategy_corr_rolling,
        label=f"{name}, Std Dev: {strategy_corr_rolling.std():.2f}",
        color="steelblue",
    )
    plt.plot(
        mom_corr_rolling,
        label=f"Momentum, Std Dev: {mom_corr_rolling.std():.2f}",
        color="darkorange",
    )
    plt.axhline(
        strategy_corr,
        color="steelblue",
        linestyle="--",
        label=f"Overall Correlation: {strategy_corr:.2f}",
        alpha=0.6,
    )
    plt.axhline(
        mom_corr,
        color="darkorange",
        linestyle="--",
        label=f"Overall Correlation: {mom_corr:.2f}",
        alpha=0.6,
    )
    plt.title(f"{window}-Day Rolling Correlation to SPY")
    plt.ylabel("Correlation")
    plt.xlabel("Date")
    plt.ylim([-1, 1])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    if save:
        filename = Path(config.RESULTS_DIR) / file
        filename.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filename)
        logger.info("Correlation plot saved to: %s", filename)
    plt.close()


def plot_leverage(
    weights_df: pd.DataFrame,
    name: str = "Strategy",
    figsize: tuple[int, int] = (12, 4),
    save: bool = True,
    file: str = "daily_leverage.png",
) -> None:
    """
    Plot daily leverage (sum of absolute weights).

    Parameters
    ----------
    weights_df : pd.DataFrame
        DataFrame with asset weights over time.
    name : str, optional
        Strategy name for title, by default "Strategy".
    figsize : Tuple[int, int], optional
        Figure size, by default (12, 4).
    save : bool, optional
        Whether to save the plot, by default True.
    file : str, optional
        Filename for saving the plot, by default "daily_leverage.png".

    Returns
    -------
    None
    """
    daily_leverage = weights_df.abs().sum(axis=1)
    net_exposure = weights_df.sum(axis=1)
    plt.figure(figsize=figsize)
    plt.plot(
        daily_leverage,
        label=f"Daily Leverage (Gross): Max {daily_leverage.max():.2f}",
        color="steelblue",
    )
    plt.plot(
        net_exposure,
        label=f"Daily Net Exposure: Max {net_exposure.max():.2f}",
        color="grey",
    )
    plt.axhline(
        daily_leverage.mean(),
        color="steelblue",
        linestyle="--",
        alpha=0.7,
        label=f"Average Leverage: {daily_leverage.mean():.2f}",
    )
    plt.axhline(
        net_exposure.mean(),
        color="grey",
        linestyle="--",
        alpha=0.7,
        label=f"Average Net Exposure: {net_exposure.mean():.2f}",
    )
    plt.title("Daily Leverage: " + name)
    plt.ylabel("Leverage")
    plt.xlabel("Date")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    if save:
        filename = Path(config.RESULTS_DIR) / file
        filename.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filename)
        logger.info("Leverage plot saved to: %s", filename)
    plt.close()


def plot_rolling_sharpe(
    strategy_returns: pd.Series,
    mom_returns: pd.Series,
    window: int = 60,
    name: str = "Strategy",
    figsize: tuple[int, int] = (12, 4),
    save: bool = True,
    method: str = "compound",
    file: str = "rolling_sharpe.png",
) -> None:
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

    Returns
    -------
    None
    """

    r_sharpe = rolling_sharpe(strategy_returns, window, method=method)
    r_sharpe_mom = rolling_sharpe(mom_returns, window, method=method)

    plt.figure(figsize=figsize)
    plt.plot(r_sharpe, label=f"{name}", color="steelblue")
    plt.plot(r_sharpe_mom, label="Momentum", color="darkorange")

    plt.axhline(
        r_sharpe.mean(),
        color="steelblue",
        linestyle="--",
        alpha=0.7,
        label=f"Avg Sharpe {name}: {round(r_sharpe.mean(), 2)}",
    )
    plt.axhline(
        r_sharpe_mom.mean(),
        color="darkorange",
        linestyle="--",
        alpha=0.7,
        label=f"Avg Sharpe Momentum: {round(r_sharpe_mom.mean(), 2)}",
    )

    plt.title(f"{window}-Day Rolling Sharpe")
    plt.ylabel("Sharpe Ratio")
    plt.xlabel("Date")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    if save:
        filename = Path(config.RESULTS_DIR) / file
        filename.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filename)
        logger.info("Rolling Sharpe plot saved to: %s", filename)

    plt.close()


def rolling_sharpe(returns: pd.Series, window: int, method: str = "compound") -> pd.Series:
    """
    Calculate rolling Sharpe ratio.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy.
    window : int
        Rolling window size in days.
    method : str
        - "compound": uses compound annualized return over the window
        - "simple": uses mean return over window, annualized

    Returns
    -------
    pd.Series
        Rolling Sharpe ratios.
    """
    if method not in ["compound", "simple"]:
        raise ValueError("method must be 'compound' or 'simple'")

    if method == "compound":

        def compound_annual(x: pd.Series) -> float:
            if len(x) == 0:
                return float("nan")
            cumulative = float((1 + x).prod() - 1)
            return float((1 + cumulative) ** (252 / len(x)) - 1)

        rolling_annual_return = returns.rolling(window).apply(compound_annual, raw=False)
    else:  # simple
        daily_mean = returns.rolling(window).mean()
        rolling_annual_return = daily_mean * 252

    rolling_annual_vol = returns.rolling(window).std() * np.sqrt(252)
    sharpe = rolling_annual_return / rolling_annual_vol
    return sharpe


def compute_pnl_per_trade(
    weights_df: pd.DataFrame, filtered_signals: pd.DataFrame, returns: pd.DataFrame
) -> pd.Series:
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

    stacked_signals: pd.Series = filtered_signals.stack()
    stacked_weights: pd.Series = weights_df.stack()
    stacked_returns: pd.Series = returns.stack()

    entry_signals: pd.Series = stacked_signals[stacked_signals != 0]

    pnl_list: list[float] = []
    trade_keys: list[tuple[pd.Timestamp, Any]] = []

    # be explicit for mypy
    dates_unique: pd.Index = stacked_returns.index.get_level_values(0).unique().sort_values()

    for key, _ in entry_signals.items():
        date = key[0]  # pd.Timestamp
        asset = key[1]

        if key not in stacked_weights.index:
            continue

        weight = float(stacked_weights.loc[key])
        if weight == 0.0:
            continue

        # weights for the same asset, from entry onward
        future_weights = stacked_weights.loc[(slice(date, None), asset)]
        zero_mask: pd.Series = future_weights == 0
        exit_date: pd.Timestamp
        if bool(zero_mask.any()):
            # idxmax() returns a composite index; pick the date component
            exit_date = zero_mask.idxmax()[0]  # type: ignore[index]
        else:
            exit_date = future_weights.index[-1][0]  # type: ignore[index]

        # first trading day strictly AFTER the entry date
        i = int(dates_unique.searchsorted(date, side="right"))
        if i < len(dates_unique):
            start = dates_unique[i]
            trade_returns: pd.Series = stacked_returns.loc[(slice(start, exit_date), asset)]
        else:
            trade_returns = pd.Series(dtype=float)

        pnl = float((trade_returns * weight).sum())

        pnl_list.append(pnl)
        trade_keys.append((date, asset))

    return pd.Series(
        pnl_list,
        index=pd.MultiIndex.from_tuples(trade_keys, names=["date", "asset"]),
        name="Trade PnL",
    )


def plot_alpha_beta(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    name: str = "Strategy",
    bench_name: str = "SPY",
    figsize: tuple[int, int] = (6, 6),
    save: bool = True,
    file: str = "alpha_beta_regression.png",
    plot: bool = False,
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
    Tuple[float, float]
        Annualized alpha and beta of the strategy relative to the benchmark.
    """
    if not isinstance(strategy_returns, pd.Series) or not isinstance(benchmark_returns, pd.Series):
        raise TypeError("Both 'strategy_returns' and 'benchmark_returns' must be pandas Series.")

    df = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    # use to_numpy() to avoid ExtensionArray issues
    x = df.iloc[:, 1].to_numpy().reshape(-1, 1)  # Benchmark
    y = df.iloc[:, 0].to_numpy()  # Strategy

    reg = LinearRegression().fit(x, y)
    alpha = reg.intercept_ * 252  # Annualized
    beta = reg.coef_[0]

    if plot:
        x_pred = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
        y_pred = reg.predict(x_pred)

        plt.figure(figsize=figsize)
        plt.scatter(x, y, alpha=0.3, s=10, label="Daily Returns")
        plt.plot(x_pred, y_pred, color="red", label=f"Fit: y = {beta:.2f}x + {alpha:.2f}")
        plt.axhline(0, color="gray", linestyle="--", linewidth=1)
        plt.axvline(0, color="gray", linestyle="--", linewidth=1)

        plt.xlabel(f"{bench_name} Daily Returns")
        plt.ylabel(f"{name} Daily Returns")
        plt.title(f"{name} vs. {bench_name} – Alpha/Beta Regression")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        if save:
            filename = Path(config.RESULTS_DIR) / file
            filename.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(filename)
            logger.info("Alpha-Beta plot saved to: %s", filename)

        plt.close()

    return alpha, beta


def summarize_performance(
    returns: pd.Series,
    bench_spy: pd.Series,
    filtered_signals: pd.DataFrame | None = None,
    Y: pd.Series | None = None,
    turnover: pd.Series | None = None,
    weights_df: pd.DataFrame | None = None,
    strategy: bool = True,
) -> pd.Series:
    """
    Summarizes the performance of a trading strategy.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy.
    bench_spy : pd.Series
        Daily returns of the SPY benchmark.
    filtered_signals : Optional[pd.DataFrame], optional
        Binary signal matrix (1s where trades occur), by default None.
    Y : Optional[pd.Series], optional
        Meta-labels (+1, 0, -1 or binary), by default None.
    turnover : Optional[pd.Series], optional
        Daily turnover values, by default None.
    weights_df : Optional[pd.DataFrame], optional
        Weights matrix used for execution, by default None.
    strategy : bool, optional
        Indicates if the summary is for a strategy (True) or a benchmark (False), by default True.

    Returns
    -------
    pd.Series
        A series containing various performance metrics.
    """
    returns = returns.dropna()
    if len(returns) == 0:
        logger.warning("Empty returns passed to summarize_performance; returning empty Series")
        return pd.Series(dtype=str)
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
    alpha, beta = plot_alpha_beta(returns, bench_spy, plot=False)
    monthly_returns = returns.resample("M").sum()

    summary = pd.Series(
        {
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
            "Alpha": f"{alpha:.3f}",
            "Beta": f"{beta:.3f}",
            "Positive Months": f"{(monthly_returns > 0).sum()}",
            "Negative Months": f"{(monthly_returns < 0).sum()}",
        }
    )

    if strategy and filtered_signals is not None and Y is not None:
        # Apply trading mask if weights_df is available
        traded_signals: pd.DataFrame = filtered_signals
        if weights_df is not None:
            traded_signals = traded_signals.where(weights_df != 0, other=0)

        stacked_trades: pd.Series = traded_signals.stack()
        traded_idx: pd.MultiIndex = stacked_trades[stacked_trades != 0].index  # (date, asset)

        # mypy-friendly boolean mask for the Series index
        mask: np.ndarray = np.in1d(Y.index, traded_idx)
        traded_outcomes: pd.Series = Y.loc[mask]

        total = int(traded_outcomes.shape[0])

        # Align entry side to the same multiindex as traded outcomes
        entry_side: pd.Series = stacked_trades.reindex(traded_outcomes.index).astype(int)

        long_mask: pd.Series = entry_side > 0
        short_mask: pd.Series = entry_side < 0

        success: pd.Series = traded_outcomes == 1
        long_hit_rate = float(success[long_mask].mean()) if bool(long_mask.any()) else np.nan
        short_hit_rate = float(success[short_mask].mean()) if bool(short_mask.any()) else np.nan

        def _avg_streak_from_weights(w: pd.DataFrame) -> float:
            if w is None:
                return 0.0
            active = (w.abs() > 1e-8).astype(int)
            streaks: list[int] = []
            # iterate columns explicitly (mypy)
            for col in active.columns:
                s = active[col].to_numpy()
                if s.sum() == 0:
                    continue
                dif = np.diff(np.r_[0, s, 0])
                starts = np.where(dif == 1)[0]
                ends = np.where(dif == -1)[0]
                streaks.extend((ends - starts).tolist())
            return float(np.mean(streaks)) if streaks else 0.0

        avg_holding_days: float = (
            _avg_streak_from_weights(weights_df) if weights_df is not None else 0.0
        )

        # Trade PnL stats (guard weights_df may be None)
        asset_returns = load_returns()
        if weights_df is not None:
            trade_pnls: pd.Series = compute_pnl_per_trade(
                weights_df, filtered_signals, asset_returns
            )
        else:
            trade_pnls = pd.Series(dtype=float)

        total_wins = int((trade_pnls > 0).sum()) if not trade_pnls.empty else 0
        losses_sum = float(-trade_pnls[trade_pnls < 0].sum()) if not trade_pnls.empty else 0.0
        profit_factor = (
            float(trade_pnls[trade_pnls > 0].sum()) / losses_sum if losses_sum > 0 else np.nan
        )

        # Weighted win rate
        if weights_df is not None and not trade_pnls.empty:
            trade_weights = weights_df.stack().loc[trade_pnls.index].abs()
            win_indicators = (trade_pnls > 0).astype(int)
            tw_sum = float(trade_weights.to_numpy().sum())
            weighted_win_rate = (
                float((win_indicators * trade_weights).to_numpy().sum()) / tw_sum
                if tw_sum > 0
                else np.nan
            )
        else:
            weighted_win_rate = np.nan

        summary["Trade Count"] = total
        summary["Win Rate"] = (
            f"{(total_wins / len(trade_pnls)):.2%}" if len(trade_pnls) > 0 else "N/A"
        )
        summary["Successful (+1) Trades"] = int((traded_outcomes == 1).sum())
        summary["Bad (-1) Trades"] = int((traded_outcomes == -1).sum())
        summary["Timeout (0) Trades"] = int((traded_outcomes == 0).sum())
        summary["Avg Holding Period (days)"] = round(avg_holding_days, 2)
        summary["Notional-Weighted Win Rate"] = (
            f"{weighted_win_rate:.2%}" if not np.isnan(weighted_win_rate) else "N/A"
        )
        summary["Avg PnL per Trade"] = f"{trade_pnls.mean():.2%}" if not trade_pnls.empty else "N/A"
        summary["Median PnL per Trade"] = (
            f"{trade_pnls.median():.2%}" if not trade_pnls.empty else "N/A"
        )
        summary["Profit Factor"] = f"{profit_factor:.2f}" if not np.isnan(profit_factor) else "N/A"
        summary["Long Hit Rate"] = f"{long_hit_rate:.2%}" if not np.isnan(long_hit_rate) else "N/A"
        summary["Short Hit Rate"] = (
            f"{short_hit_rate:.2%}" if not np.isnan(short_hit_rate) else "N/A"
        )

    else:
        rows = [
            "Trade Count",
            "Win Rate",
            "Successful (+1) Trades",
            "Bad (-1) Trades",
            "Timeout (0) Trades",
            "Avg Holding Period (days)",
            "Notional-Weighted Win Rate",
            "Avg PnL per Trade",
            "Median PnL per Trade",
            "Profit Factor",
            "Long Hit Rate",
            "Short Hit Rate",
        ]
        for i in rows:
            summary[i] = "N/A"

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
    start: str = config.FOLD3_START,
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
        Strategy name to be used in plot title and file name, by default "Strategy".
    start : str, optional
        Start date for comparing SPY and momentum benchmarks, by default config.FOLD3_START.
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
        strategy_returns,
        bench_spy,
        filtered_signals,
        Y,
        turnover,
        weights_df,
    )
    summary_costs = summarize_performance(
        strategy_returns_w_costs,
        bench_spy,
        filtered_signals,
        Y,
        turnover,
        weights_df,
    )

    _, beta_net = plot_alpha_beta(strategy_returns_w_costs, bench_spy, plot=False)
    df_hedge = pd.concat([strategy_returns_w_costs, bench_spy], axis=1).dropna()
    beta_neutral = df_hedge.iloc[:, 0] - beta_net * df_hedge.iloc[:, 1]
    summary_beta_neutral = summarize_performance(beta_neutral, bench_spy, strategy=False)

    blend_50_50 = equal_weight_blend(strategy_returns_w_costs, bench_spy, w=0.5)
    summary_blend = summarize_performance(blend_50_50, bench_spy, strategy=False)

    summary_spy = summarize_performance(bench_spy, bench_spy, strategy=False)
    summary_mom = summarize_performance(bench_mom, bench_spy, strategy=False)
    summary_mom_ls = summarize_performance(bench_mom_ls, bench_spy, strategy=False)

    if plot:
        render_bundle(
            gross=strategy_returns,
            net=strategy_returns_w_costs,
            spy=bench_spy,
            mom=bench_mom,
            name=name,
            folder="strategy",
            start=start,
            weights=weights_df,
            turnover=turnover,
            save=save,
        )

        render_bundle(
            gross=blend_50_50,
            net=blend_50_50,
            spy=bench_spy,
            mom=bench_mom,
            name=f"{name} + SPY (50/50)",
            folder="blend",
            start=start,
            save=save,
        )

        render_bundle(
            gross=beta_neutral,
            net=beta_neutral,
            spy=bench_spy,
            mom=bench_mom,
            name=f"{name} (Beta Neutral)",
            folder="beta_neutral",
            start=start,
            save=save,
        )

    summary_df = pd.concat(
        [
            summary,
            summary_costs,
            summary_blend,
            summary_beta_neutral,
            summary_spy,
            summary_mom,
            summary_mom_ls,
        ],
        axis=1,
    )
    summary_df.columns = [
        f"{name} (Gross)",
        f"{name} (Net)",
        "50% Blend (Net)",
        "Beta Neutral (Net)",
        "SPY",
        "Standard Momentum",
        "Standard Momentum (Long Short)",
    ]
    if save:
        out_dir = Path(config.RESULTS_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_df.to_excel(out_dir / "performance_summary.xlsx")
        logger.info("Backtest summary saved to: %s", out_dir / "performance_summary.xlsx")
        strategy_returns_w_costs.to_csv(out_dir / "strategy_net.csv")
        logger.info("Strategy net returns saved to: %s", out_dir / "strategy_net.csv")
        blend_50_50.to_csv(out_dir / "blend_50_50.csv")
        logger.info("Blend 50/50 returns saved to: %s", out_dir / "blend_50_50.csv")
        beta_neutral.to_csv(out_dir / "beta_neutral.csv")
        logger.info("Beta neutral returns saved to: %s", out_dir / "beta_neutral.csv")

    return summary_df


def equal_weight_blend(a: pd.Series, b: pd.Series, w: float = 0.5) -> pd.Series:
    """
    Daily-rebalanced equal-weight blend of two return series.
    Returns index-aligned (drops dates with NaNs in either leg).
    """
    df = pd.concat([a, b], axis=1).dropna()
    return w * df.iloc[:, 0] + (1 - w) * df.iloc[:, 1]


def render_bundle(
    gross: pd.Series,
    net: pd.Series,
    spy: pd.Series,
    mom: pd.Series,
    *,
    name: str,
    folder: str,
    start: str,
    weights: pd.DataFrame | None = None,
    turnover: pd.Series | None = None,
    save: bool = True,
) -> None:
    """Render the standard plot set for a return series."""
    # ensure subfolder exists
    out = Path(config.RESULTS_DIR) / folder
    out.mkdir(parents=True, exist_ok=True)

    plot_cumulative_returns(
        gross,
        net,
        spy,
        mom,
        name=name,
        start=start,
        save=save,
        file=f"{folder}/cumulative_returns.png",
    )
    plot_drawdown_underwater(
        net,
        mom,
        (name, "Standard Momentum"),
        save=save,
        file=f"{folder}/drawdown_underwater.png",
    )
    plot_alpha_beta(
        net,
        spy,
        name=name,
        plot=True,
        save=save,
        file=f"{folder}/alpha_beta_regression.png",
    )
    plot_rolling_correlation(
        net,
        spy,
        mom,
        name=name,
        save=save,
        file=f"{folder}/rolling_corr.png",
    )
    plot_rolling_sharpe(
        net,
        mom,
        name=name,
        method="compound",
        save=save,
        file=f"{folder}/rolling_sharpe.png",
    )
    if turnover is not None:
        plot_turnover(turnover, name=name, save=save, file=f"{folder}/turnover.png")
    if weights is not None:
        plot_leverage(weights, name=name, save=save, file=f"{folder}/daily_leverage.png")
