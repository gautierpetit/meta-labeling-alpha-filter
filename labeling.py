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
    tp_sl_factor: Tuple[float, float] = config.PT_SL_FACTOR,
    max_holding_period: int = config.MAX_HOLDING_PERIOD,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies the triple barrier method for labeling trades.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily price data.
    daily_signals : pd.DataFrame
        Signal matrix with values in {-1, 0, 1}.
    volatility : pd.DataFrame
        Rolling volatility estimates.
    tp_sl_factor : Tuple[float, float], optional
        Tuple of (take-profit multiplier, stop-loss multiplier), by default config.PT_SL_FACTOR.
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
            side = daily_signals.at[date, ticker]
            entry_price = prices.at[date, ticker]
            vol = volatility.at[date, ticker]

            if pd.isna(entry_price) or pd.isna(vol) or vol == 0:
                continue

            tp_level, sl_level = calculate_barrier_levels(
                entry_price, vol, side, tp_sl_factor
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
    entry_price: float, vol: float, side: int, tp_sl_factor: Tuple[float, float]
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
    tp_sl_factor : Tuple[float, float]
        Tuple of (take-profit multiplier, stop-loss multiplier).

    Returns
    -------
    Tuple[float, float]
        Take-profit and stop-loss levels.
    """
    if side == 1:  # Long trade
        tp_level = entry_price * (1 + tp_sl_factor[0] * vol)
        sl_level = entry_price * (1 - tp_sl_factor[1] * vol)
    else:  # Short trade
        tp_level = entry_price * (1 - tp_sl_factor[0] * vol)
        sl_level = entry_price * (1 + tp_sl_factor[1] * vol)

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


def scan_tp_sl_grid(
    prices: pd.DataFrame,
    daily_signals: pd.DataFrame,
    volatility: pd.DataFrame,
    tp_range: Tuple[int, int] = (1, 4),
    sl_range: Tuple[int, int] = (1, 4),
    max_holding_period: int = 20,
) -> pd.DataFrame:
    """
    Grid-search tp/sl factor combinations and evaluate label distributions.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily price data.
    daily_signals : pd.DataFrame
        Signal matrix with values in {-1, 0, 1}.
    volatility : pd.DataFrame
        Rolling volatility estimates.
    tp_range : Tuple[int, int], optional
        Range of take-profit multipliers to test, by default (1, 4).
    sl_range : Tuple[int, int], optional
        Range of stop-loss multipliers to test, by default (1, 4).
    max_holding_period : int, optional
        Maximum holding period in days, by default 20.

    Returns
    -------
    pd.DataFrame
        DataFrame of (tp, sl) vs label proportions.
    """
    results: List[Dict] = []

    for tp, sl in itertools.product(
        range(tp_range[0], tp_range[1] + 1), range(sl_range[0], sl_range[1] + 1)
    ):
        labels, _ = apply_triple_barrier(
            prices=prices,
            daily_signals=daily_signals,
            volatility=volatility,
            tp_sl_factor=(tp, sl),
            max_holding_period=max_holding_period,
        )

        label_counts = labels.stack().value_counts(normalize=True).to_dict()

        results.append(
            {
                "tp": tp,
                "sl": sl,
                "label_1": label_counts.get(1, 0),
                "label_0": label_counts.get(0, 0),
                "label_-1": label_counts.get(-1, 0),
                "coverage": labels.stack().notna().mean(),
            }
        )

    results_df = pd.DataFrame(results)

    results_df["label_balance"] = abs(results_df["label_1"] - results_df["label_-1"])
    results_df.sort_values(by=["label_0", "label_balance"], inplace=True)
    results_df.to_excel(config.FIGURES_DIR / "tp_sl_grid.xlsx")

    return results_df


def plot_tp_sl_distribution():
    results_df = pd.read_excel(config.FIGURES_DIR / "tp_sl_grid.xlsx")
    pivot_label1 = results_df.pivot(index="tp", columns="sl", values="label_1")
    pivot_label0 = results_df.pivot(index="tp", columns="sl", values="label_0")
    pivot_label_1 = results_df.pivot(index="tp", columns="sl", values="label_-1")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Use label_0 as the background color (timeouts)
    sns.heatmap(pivot_label0, annot=False, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax)

    # Overlay the full label distribution in each cell
    for i in range(pivot_label0.shape[0]):
        for j in range(pivot_label0.shape[1]):
            pt = pivot_label0.index[i]
            sl = pivot_label0.columns[j]
            v1 = pivot_label1.loc[pt, sl]
            v0 = pivot_label0.loc[pt, sl]
            v_1 = pivot_label_1.loc[pt, sl]
            text = f"{v1:.0%}\n{v0:.0%}\n{v_1:.0%}"
            ax.text(j + 0.5, i + 0.5, text, ha="center", va="center", color="black")

    ax.set_title("Label Distribution (PT vs SL)\n[label_1 / label_0 / label_-1]")
    plt.xlabel("Stop-Loss (SL × σ)")
    plt.ylabel("Take-Profit (PT × σ)")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "heatmap_TP_SL_combined.png")
    plt.close()


def plot_before_after_label_distribution(
    tp_sl_old,
    tp_sl_new,
    prices: pd.DataFrame,
    daily_signals: pd.DataFrame,
    volatility: pd.DataFrame,
):
    """
    Compare label distributions before and after applying a new tp/sl factor.

    Parameters
    ----------
    tp_sl_old : Tuple[float, float]
        Old take-profit and stop-loss factors.
    tp_sl_new : Tuple[float, float]
        New take-profit and stop-loss factors.
    prices : pd.DataFrame
        Daily price data.
    daily_signals : pd.DataFrame
        Signal matrix with values in {-1, 0, 1}.
    volatility : pd.DataFrame
        Rolling volatility estimates.

    Returns
    -------
    None
    """
    labels_old, _ = apply_triple_barrier(
        prices, daily_signals, volatility, tp_sl_factor=tp_sl_old
    )
    labels_new, _ = apply_triple_barrier(
        prices, daily_signals, volatility, tp_sl_factor=tp_sl_new
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, labels, title in zip(
        axes,
        [labels_old, labels_new],
        [f"Old Labels (PT/SL={tp_sl_old})", f"New Labels (PT/SL={tp_sl_new})"],
    ):
        vc = labels.stack().value_counts(normalize=True).sort_index()
        vc.plot(kind="bar", ax=ax, color="skyblue", title=title)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Proportion")

    plt.tight_layout()
    plt.savefig("figures/label_distribution_before_after.png")
    plt.close()
