import itertools
from typing import Tuple

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
    Applies the triple barrier labeling method and outputs binary labels + label_times for purging.

    Returns:
        - labels_binary: DataFrame of binary labels (1 = TP, 0 = SL), excludes undecided (time limit)
        - label_times: Multi-index DataFrame with t0 (entry time) and t1 (label resolution time)
    """
    labels = pd.DataFrame(index=daily_signals.index, columns=daily_signals.columns)
    t1_matrix = pd.DataFrame(index=daily_signals.index, columns=daily_signals.columns)

    for date in tqdm(daily_signals.index):
        tickers = daily_signals.columns[daily_signals.loc[date] == 1]
        for ticker in tickers:
            entry_price = prices.at[date, ticker]
            vol = volatility.at[date, ticker]

            if pd.isna(entry_price) or pd.isna(vol):
                continue

            tp_level = entry_price * (1 + pt_sl_factor[0] * vol)
            sl_level = entry_price * (1 - pt_sl_factor[1] * vol)

            future_dates = prices.index[prices.index > date][:max_holding_period]
            future_prices = prices.loc[future_dates, ticker]

            label = 0
            for future_date, price in future_prices.items():
                if price >= tp_level:
                    label = 1
                    t1_matrix.at[date, ticker] = future_date
                    break
                elif price <= sl_level:
                    label = -1
                    t1_matrix.at[date, ticker] = future_date
                    break
            else:
                # No barrier hit: use last date of window as time barrier
                if not future_dates.empty:
                    t1_matrix.at[date, ticker] = future_dates[-1]

            labels.at[date, ticker] = label

    # Only keep clear TP/SL, drop time-barrier trades
    labels_binary = labels[labels != 0].replace({-1: 0, 1: 1}).dropna(how="all")

    # Construct label_times: DataFrame with MultiIndex (t0, asset) -> t1
    label_times = (
        t1_matrix.stack()
        .rename("t1")
        .to_frame()
        .assign(t0=lambda df: df.index.get_level_values(0))
        .loc[labels_binary.stack().index]
    )

    return labels, label_times


def scan_pt_sl_grid(
    prices,
    daily_signals,
    volatility,
    pt_range=(1, 4),
    sl_range=(1, 4),
    max_holding_period=20,
):
    """
    Grid-search pt/sl factor combinations and evaluate label distributions.

    Returns:
        DataFrame of (pt, sl) vs label proportions.
    """
    results = []

    for pt, sl in itertools.product(
        range(pt_range[0], pt_range[1] + 1), range(sl_range[0], sl_range[1] + 1)
    ):
        labels, label_times = apply_triple_barrier(
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

    results = pd.DataFrame(results)

    for label in ["label_1", "label_0", "label_-1"]:
        pivot = results.pivot(index="pt", columns="sl", values=label)
        plt.figure(figsize=(6, 4))
        sns.heatmap(pivot, annot=True, cmap="coolwarm")
        plt.title(f"Proportion of {label}")
        plt.savefig(config.FIGURES_DIR / "heatmanp_TP_SL.png")
        plt.close()

    results.to_excel(config.MODELS_DIR / "pt_sl_grid.xlsx")

    return results
