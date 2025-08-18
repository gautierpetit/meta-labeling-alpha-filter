"""Labeling utilities implementing the triple-barrier method and helper scans.

This module contains two implementations of the triple-barrier labeling
algorithm (a straightforward looped version and a vectorized implementation),
plus helpers to scan parameter grids (TP/SL multipliers and holding periods).
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import src.config as config

logger = logging.getLogger(__name__)


def apply_triple_barrier_ref(
    prices: pd.DataFrame,
    daily_signals: pd.DataFrame,
    volatility: pd.DataFrame,
    tp_sl_factor: tuple[float, float] = config.PT_SL_FACTOR,
    max_holding_period: int = config.MAX_HOLDING_PERIOD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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

            # Calculate take-profit and stop-loss levels
            if side == 1:  # Long trade
                tp_level = entry_price * (1 + tp_sl_factor[0] * vol)
                sl_level = entry_price * (1 - tp_sl_factor[1] * vol)
            else:  # Short trade
                tp_level = entry_price * (1 - tp_sl_factor[0] * vol)
                sl_level = entry_price * (1 + tp_sl_factor[1] * vol)

            # Determine the label and exit date
            future_dates = prices.index[prices.index > date][:max_holding_period]
            future_prices = prices.loc[future_dates, ticker]

            label, exit_date = 0, None
            for future_date, price in future_prices.items():
                if (side == 1 and price >= tp_level) or (side == -1 and price <= tp_level):
                    label, exit_date = 1, future_date  # Take-profit hit
                    break
                elif (side == 1 and price <= sl_level) or (side == -1 and price >= sl_level):
                    label, exit_date = -1, future_date  # Stop-loss hit
                    break

            if label == 0 and not future_prices.empty:
                exit_date = future_prices.index[-1]  # Timeout

            labels.at[date, ticker] = label
            if exit_date:
                t1_matrix.at[date, ticker] = exit_date

    # make types explicit for static checkers: stack() returns a Series
    # Ensure t1_series is explicitly a Series
    stacked: pd.Series = t1_matrix.stack()
    t1_series: pd.Series = stacked.rename("t1")
    label_times = t1_series.to_frame().assign(t0=lambda df: df.index.get_level_values(0))

    return labels, label_times


def apply_triple_barrier(
    prices: pd.DataFrame,
    daily_signals: pd.DataFrame,
    volatility: pd.DataFrame,
    tp_sl_factor: tuple[float, float] = config.PT_SL_FACTOR,
    max_holding_period: int = config.MAX_HOLDING_PERIOD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Vectorized triple-barrier labeling with identical outputs to the loop version.

    Rules preserved:
      - First hit wins; TP checked before SL on the same day.
      - Timeout at last day in the window if neither barrier hit.
      - Skip entries with NaN price/vol or vol==0 (leave NaN label/t1).
    """
    idx = prices.index
    cols = prices.columns

    P = prices.to_numpy(copy=False)  # (T, N)
    V = volatility.reindex(idx).to_numpy(copy=False)
    S = daily_signals.reindex(idx).to_numpy(copy=False).astype(np.int8)

    T, N = P.shape
    tpf, slf = float(tp_sl_factor[0]), float(tp_sl_factor[1])

    # Outputs (float for labels to allow NaN; exit positions as int with -1=unset)
    labels_arr = np.full((T, N), np.nan, dtype=float)
    exit_pos = np.full((T, N), -1, dtype=np.int32)

    for i in range(T):
        sides = S[i, :]  # -1, 0, +1
        active_mask = sides != 0
        if not np.any(active_mask):
            continue

        # Filter invalid entries (NaN price/vol or vol==0)
        entry = P[i, active_mask]
        vol_i = V[i, active_mask]
        valid = (~np.isnan(entry)) & (~np.isnan(vol_i)) & (vol_i != 0)
        if not np.any(valid):
            continue

        col_idx = np.where(active_mask)[0][valid]
        s = sides[active_mask][valid]  # (-1,+1)
        e = entry[valid]
        v = vol_i[valid]

        # Compute barrier levels
        tp = np.where(s == 1, e * (1.0 + tpf * v), e * (1.0 - tpf * v))
        sl = np.where(s == 1, e * (1.0 - slf * v), e * (1.0 + slf * v))

        # Future window [i+1 .. end] inclusive
        end = min(i + max_holding_period, T - 1)
        if end <= i:
            continue
        W = end - i  # number of rows in the window
        Wslice = P[i + 1 : end + 1, :][:, col_idx]  # (W, K)

        # Vectorized hits with side-specific inequalities
        K = len(col_idx)
        hits_tp = np.zeros((W, K), dtype=bool)
        hits_sl = np.zeros((W, K), dtype=bool)

        is_long = s == 1
        if np.any(is_long):
            j = np.where(is_long)[0]
            hits_tp[:, j] = Wslice[:, j] >= tp[j]
            hits_sl[:, j] = Wslice[:, j] <= sl[j]
        if np.any(~is_long):
            j = np.where(~is_long)[0]
            hits_tp[:, j] = Wslice[:, j] <= tp[j]
            hits_sl[:, j] = Wslice[:, j] >= sl[j]

        # First passage indices (0..W-1); W denotes "no hit"
        any_tp = hits_tp.any(axis=0)
        any_sl = hits_sl.any(axis=0)
        idx_tp = np.where(any_tp, hits_tp.argmax(axis=0), W)
        idx_sl = np.where(any_sl, hits_sl.argmax(axis=0), W)

        # Who hits first? (TP gets priority on exact same day, matching your loop order)
        first_idx = np.minimum(idx_tp, idx_sl)
        has_event = first_idx < W

        tp_first = (idx_tp < idx_sl) & has_event
        sl_first = (idx_sl < idx_tp) & has_event
        tie_tp = (idx_tp == idx_sl) & (idx_tp < W)  # TP before SL on tie

        # Assign labels at entry row i, selected columns
        if np.any(tp_first | tie_tp):
            labels_arr[i, col_idx[tp_first | tie_tp]] = 1.0
        if np.any(sl_first):
            labels_arr[i, col_idx[sl_first]] = -1.0
        # Timeouts: label 0 at entry time
        timeout = ~has_event
        if np.any(timeout):
            labels_arr[i, col_idx[timeout]] = 0.0

        # Exit positions (absolute row indices)
        # Events: i+1+first_idx; Timeouts: end
        abs_exit = np.where(has_event, i + 1 + first_idx, end)
        exit_pos[i, col_idx] = abs_exit.astype(np.int32)

    # → DataFrame outputs identical to the original interface
    labels = pd.DataFrame(labels_arr, index=idx, columns=cols)

    valid = exit_pos >= 0
    r, c = np.where(valid)

    mi = pd.MultiIndex.from_arrays([idx[r], cols[c]], names=["Date", "Ticker"])
    t1 = pd.Series(idx[exit_pos[r, c]], index=mi, name="t1")

    label_times = t1.to_frame().assign(t0=lambda df: df.index.get_level_values(0))

    return labels, label_times


def scan_tp_sl_grid(
    prices: pd.DataFrame,
    daily_signals: pd.DataFrame,
    volatility: pd.DataFrame,
    tp_range: tuple[int, int] = (1, 4),
    sl_range: tuple[int, int] = (1, 4),
    max_holding_period: int = config.MAX_HOLDING_PERIOD,
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
        Maximum holding period in days, by default 63.

    Returns
    -------
    pd.DataFrame
        DataFrame of (tp, sl) vs label proportions.
    """
    results: list[dict] = []

    for tp, sl in tqdm(
        itertools.product(range(tp_range[0], tp_range[1] + 1), range(sl_range[0], sl_range[1] + 1))
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
    results_df["entropy"] = -(
        results_df["label_1"] * np.log(results_df["label_1"] + 1e-9)
        + results_df["label_0"] * np.log(results_df["label_0"] + 1e-9)
        + results_df["label_-1"] * np.log(results_df["label_-1"] + 1e-9)
    )
    results_df.sort_values(by=["label_0", "label_balance"], inplace=True)
    Path(config.FIGURES_DIR).mkdir(parents=True, exist_ok=True)
    results_df.to_excel(Path(config.FIGURES_DIR) / f"tp_sl_grid_{max_holding_period}.xlsx")

    # Pivot data for plotting
    pivot_entropy = results_df.pivot(index="tp", columns="sl", values="entropy")
    pivot_label1 = results_df.pivot(index="tp", columns="sl", values="label_1")
    pivot_label0 = results_df.pivot(index="tp", columns="sl", values="label_0")
    pivot_label_minus1 = results_df.pivot(index="tp", columns="sl", values="label_-1")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Use entropy as the background color
    sns.heatmap(
        pivot_entropy,
        annot=False,
        fmt=".2f",
        cmap="coolwarm",
        cbar=True,
        cbar_kws={"label": "Entropy"},
        ax=ax,
    )

    # Overlay the full label distribution in each cell
    for i in range(pivot_entropy.shape[0]):
        for j in range(pivot_entropy.shape[1]):
            pt = pivot_entropy.index[i]
            sl = pivot_entropy.columns[j]
            v1 = pivot_label1.loc[pt, sl]
            v0 = pivot_label0.loc[pt, sl]
            v_1 = pivot_label_minus1.loc[pt, sl]
            text = f"{v1:.0%}\n{v0:.0%}\n{v_1:.0%}"
            ax.text(j + 0.5, i + 0.5, text, ha="center", va="center", color="black")

    ax.set_title("Label Distribution (PT vs SL) with Entropy\n[label_1 / label_0 / label_-1]")
    plt.xlabel("Stop-Loss (SL × σ)")
    plt.ylabel("Take-Profit (PT × σ)")
    plt.tight_layout()
    out_path = Path(config.FIGURES_DIR) / f"heatmap_TP_SL_{max_holding_period}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()

    return results_df


def scan_holding_period_range(
    prices: pd.DataFrame,
    daily_signals: pd.DataFrame,
    volatility: pd.DataFrame,
    tp_sl_factor: tuple[float, float],
    holding_period_range: tuple[int, int],
) -> pd.DataFrame:
    """
    Scan a range of max_holding_period values for a fixed tp/sl factor and evaluate label distributions.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily price data.
    daily_signals : pd.DataFrame
        Signal matrix with values in {-1, 0, 1}.
    volatility : pd.DataFrame
        Rolling volatility estimates.
    tp_sl_factor : Tuple[float, float]
        Fixed take-profit and stop-loss multipliers.
    holding_period_range : Tuple[int, int], optional
        Range of max_holding_period values to test, by default (1, 63).

    Returns
    -------
    pd.DataFrame
        DataFrame of max_holding_period vs label proportions.
    """
    results: list[dict] = []

    # include upper bound
    for max_holding_period in tqdm(range(holding_period_range[0], holding_period_range[1] + 1)):
        labels, _ = apply_triple_barrier(
            prices=prices,
            daily_signals=daily_signals,
            volatility=volatility,
            tp_sl_factor=tp_sl_factor,
            max_holding_period=max_holding_period,
        )

        label_counts = labels.stack().value_counts(normalize=True).to_dict()

        results.append(
            {
                "max_holding_period": max_holding_period,
                "label_1": label_counts.get(1, 0),
                "label_0": label_counts.get(0, 0),
                "label_-1": label_counts.get(-1, 0),
                "coverage": labels.stack().notna().mean(),
            }
        )

    results_df = pd.DataFrame(results)

    results_df["label_balance"] = abs(results_df["label_1"] - results_df["label_-1"])
    results_df["entropy"] = -(
        results_df["label_1"] * np.log(results_df["label_1"] + 1e-9)
        + results_df["label_0"] * np.log(results_df["label_0"] + 1e-9)
        + results_df["label_-1"] * np.log(results_df["label_-1"] + 1e-9)
    )
    results_df.sort_values(by=["label_0", "label_balance"], inplace=True)
    Path(config.FIGURES_DIR).mkdir(parents=True, exist_ok=True)
    results_df.to_excel(
        Path(config.FIGURES_DIR)
        / f"holding_period_scan_tp{tp_sl_factor[0]}_sl{tp_sl_factor[1]}.xlsx"
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot label distributions
    ax.plot(
        results_df["max_holding_period"],
        results_df["label_1"],
        label="Label 1 (Good trade)",
    )
    ax.plot(
        results_df["max_holding_period"],
        results_df["label_0"],
        label="Label 0 (Timeout)",
    )
    ax.plot(
        results_df["max_holding_period"],
        results_df["label_-1"],
        label="Label -1 (Bad trade)",
    )

    # Add a secondary y-axis for entropy
    ax2 = ax.twinx()
    ax2.plot(
        results_df["max_holding_period"],
        results_df["entropy"],
        label="Entropy",
        color="red",
        linestyle="--",
    )
    ax2.set_ylabel("Entropy", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    # Add labels and title
    ax.set_title(
        f"Label Distribution and Entropy vs Max Holding Period\n(TP={tp_sl_factor[0]}, SL={tp_sl_factor[1]})"
    )
    ax.set_xlabel("Max Holding Period (days)")
    ax.set_ylabel("Proportion")
    ax.legend(loc="lower left")
    ax.grid(True, linestyle="-")

    # Save the plot
    plt.tight_layout()
    out_path = (
        Path(config.FIGURES_DIR)
        / f"holding_period_distribution_entropy_tp{tp_sl_factor[0]}_sl{tp_sl_factor[1]}.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()

    return results_df
