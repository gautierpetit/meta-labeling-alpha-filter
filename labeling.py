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
    Applies the triple barrier method for long/short trades.

    Parameters:
        prices: Daily price data.
        daily_signals: Signal matrix with values in {-1, 0, 1}.
        volatility: Rolling volatility estimates.
        pt_sl_factor: Tuple of (pt_multiplier, sl_multiplier).
        max_holding_period: Maximum holding period in days.

    Returns:
        labels: DataFrame with {1=TP, -1=SL, 0=timeout}.
        label_times: DataFrame with (t0, t1) for trade resolution.
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

            if side == 1:  # Long trade
                tp_level = entry_price * (1 + pt_sl_factor[0] * vol)
                sl_level = entry_price * (1 - pt_sl_factor[1] * vol)
            else:  # Short trade
                tp_level = entry_price * (1 - pt_sl_factor[0] * vol)
                sl_level = entry_price * (1 + pt_sl_factor[1] * vol)

            future_dates = prices.index[prices.index > date][:max_holding_period]
            future_prices = prices.loc[future_dates, ticker]

            label = 0
            for future_date, price in future_prices.items():
                if (side == 1 and price >= tp_level) or (side == -1 and price <= tp_level): # Profitable trade
                    label = 1
                    t1_matrix.at[date, ticker] = future_date
                    break
                elif (side == 1 and price <= sl_level) or (side == -1 and price >= sl_level): # Unprofitable trade
                    label = -1
                    t1_matrix.at[date, ticker] = future_date
                    break
            else:
                if not future_dates.empty:
                    t1_matrix.at[date, ticker] = future_dates[-1]

            labels.at[date, ticker] = label

    label_times = (
        t1_matrix.stack()
        .rename("t1")
        .to_frame()
        .assign(t0=lambda df: df.index.get_level_values(0))
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
        plt.savefig(config.FIGURES_DIR / f"heatmap_TP_SL_{label}.png")
        plt.close()

    results.to_excel(config.MODELS_DIR / "pt_sl_grid.xlsx")

    return results
