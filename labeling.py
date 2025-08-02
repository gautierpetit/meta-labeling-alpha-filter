import itertools
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import config


def apply_triple_barrier(
    prices: pd.DataFrame,
    daily_signals: pd.DataFrame,
    volatility: pd.DataFrame,
    pt_sl_factor: Tuple[float, float] = config.PT_SL_FACTOR,
    max_holding_period: int = config.MAX_HOLDING_PERIOD,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies the triple barrier method for long/short trades.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily price data.
    daily_signals : pd.DataFrame
        Signal matrix with values in {-1, 0, 1}.
    volatility : pd.DataFrame
        Rolling volatility estimates.
    pt_sl_factor : Tuple[float, float], optional
        Tuple of (pt_multiplier, sl_multiplier), by default config.PT_SL_FACTOR.
    max_holding_period : int, optional
        Maximum holding period in days, by default config.MAX_HOLDING_PERIOD.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - labels: DataFrame with {1=TP, -1=SL, 0=timeout}.
        - label_times: DataFrame with (t0, t1) for trade resolution.
    """
    labels = pd.DataFrame(index=daily_signals.index, columns=daily_signals.columns)
    t1_matrix = pd.DataFrame(index=daily_signals.index, columns=daily_signals.columns)

    for date in tqdm(daily_signals.index, desc="Applying Triple Barrier"):
        tickers = daily_signals.columns[daily_signals.loc[date] != 0]
        for ticker in tickers:
            side = daily_signals.at[date, ticker]  # 1 for long, -1 for short
            entry_price = prices.at[date, ticker]
            vol = volatility.at[date, ticker]

            if pd.isna(entry_price) or pd.isna(vol):
                continue

            tp_level, sl_level = calculate_barrier_levels(
                entry_price, vol, side, pt_sl_factor
            )

            future_dates = prices.index[prices.index > date][:max_holding_period]
            future_prices = prices.loc[future_dates, ticker]

            label, exit_date = determine_label_and_exit_date(
                future_prices, tp_level, sl_level, side
            )

            labels.at[date, ticker] = label
            if exit_date:
                t1_matrix.at[date, ticker] = exit_date

    label_times = (
        t1_matrix.stack()
        .rename("t1")
        .to_frame()
        .assign(t0=lambda df: df.index.get_level_values(0))
    )

    return labels, label_times


def calculate_barrier_levels(
    entry_price: float, vol: float, side: int, pt_sl_factor: Tuple[float, float]
) -> Tuple[float, float]:
    """
    Calculate take-profit and stop-loss levels based on entry price, volatility, and trade side.

    Parameters
    ----------
    entry_price : float
        Entry price of the trade.
    vol : float
        Volatility estimate.
    side : int
        Trade side (1 for long, -1 for short).
    pt_sl_factor : Tuple[float, float]
        Tuple of (pt_multiplier, sl_multiplier).

    Returns
    -------
    Tuple[float, float]
        Take-profit and stop-loss levels.
    """
    if side == 1:  # Long trade
        tp_level = entry_price * (1 + pt_sl_factor[0] * vol)
        sl_level = entry_price * (1 - pt_sl_factor[1] * vol)
    else:  # Short trade
        tp_level = entry_price * (1 - pt_sl_factor[0] * vol)
        sl_level = entry_price * (1 + pt_sl_factor[1] * vol)

    return tp_level, sl_level


def determine_label_and_exit_date(
    future_prices: pd.Series, tp_level: float, sl_level: float, side: int
) -> Tuple[int, pd.Timestamp]:
    """
    Determine the label and exit date for a trade based on future prices.

    Parameters
    ----------
    future_prices : pd.Series
        Future prices of the asset.
    tp_level : float
        Take-profit level.
    sl_level : float
        Stop-loss level.
    side : int
        Trade side (1 for long, -1 for short).

    Returns
    -------
    Tuple[int, pd.Timestamp]
        Label (1=TP, -1=SL, 0=timeout) and exit date.
    """
    for future_date, price in future_prices.items():
        if (side == 1 and price >= tp_level) or (side == -1 and price <= tp_level):
            return 1, future_date  # Take-profit hit
        elif (side == 1 and price <= sl_level) or (side == -1 and price >= sl_level):
            return -1, future_date  # Stop-loss hit

    return 0, future_prices.index[-1] if not future_prices.empty else None


def scan_pt_sl_grid(
    prices: pd.DataFrame,
    daily_signals: pd.DataFrame,
    volatility: pd.DataFrame,
    pt_range: Tuple[int, int] = (1, 4),
    sl_range: Tuple[int, int] = (1, 4),
    max_holding_period: int = 20,
) -> pd.DataFrame:
    """
    Grid-search pt/sl factor combinations and evaluate label distributions.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily price data.
    daily_signals : pd.DataFrame
        Signal matrix with values in {-1, 0, 1}.
    volatility : pd.DataFrame
        Rolling volatility estimates.
    pt_range : Tuple[int, int], optional
        Range of pt multipliers to test, by default (1, 4).
    sl_range : Tuple[int, int], optional
        Range of sl multipliers to test, by default (1, 4).
    max_holding_period : int, optional
        Maximum holding period in days, by default 20.

    Returns
    -------
    pd.DataFrame
        DataFrame of (pt, sl) vs label proportions.
    """
    results: List[Dict] = []

    for pt, sl in itertools.product(
        range(pt_range[0], pt_range[1] + 1), range(sl_range[0], sl_range[1] + 1)
    ):
        labels, _ = apply_triple_barrier(
            prices=prices,
            daily_signals=daily_signals,
            volatility=volatility,
            pt_sl_factor=(pt, sl),
            max_holding_period=max_holding_period,
        )

        label_counts = labels.stack().value_counts(normalize=True).to_dict()

        results.append(
            {
                "pt": pt,
                "sl": sl,
                "label_1": label_counts.get(1, 0),
                "label_0": label_counts.get(0, 0),
                "label_-1": label_counts.get(-1, 0),
                "coverage": labels.stack().notna().mean(),
            }
        )

    results_df = pd.DataFrame(results)

    for label in ["label_1", "label_0", "label_-1"]:
        pivot = results_df.pivot(index="pt", columns="sl", values=label)
        plt.figure(figsize=(6, 4))
        sns.heatmap(pivot, annot=True, cmap="coolwarm")
        plt.title(f"Proportion of {label}")
        plt.savefig(config.FIGURES_DIR / f"heatmap_TP_SL_{label}.png")
        plt.close()

    results_df.to_excel(config.MODELS_DIR / "pt_sl_grid.xlsx")

    return results_df
