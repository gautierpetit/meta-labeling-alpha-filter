"""
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

Position sizing helpers.

This module contains the position-sizing logic used by the backtests and
live strategies. The main function `compute_probability_weighted_returns`
turns filtered candidate signals and predicted probabilities into an
executed book, P&L series and cost estimates.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin

import src.config as config
from src.utils import get_class_to_index

logger = logging.getLogger(__name__)


def compute_probability_weighted_returns(
    clf: ClassifierMixin,
    filtered_signals: pd.DataFrame,
    X_test: pd.DataFrame,
    returns: pd.DataFrame,
    prob_weighting: bool = config.PROB_WEIGHTING,
    target_vol: float = config.TARGET_VOL,
    leverage_cap: float = config.LEVERAGE_CAP,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.Series, pd.Series]:
    """Compute executed weights, P&L and cost estimates from candidate signals.

    Parameters
    ----------
    clf
        Trained classifier exposing `predict_proba` and compatible with
        `src.utils.get_class_to_index`.
    filtered_signals
        DataFrame same shape as `returns` with values in {-1,0,1} for
        candidate signals.
    X_test
        Feature matrix aligned to the same index as `filtered_signals.stack()`
        (typically a MultiIndex of (date, ticker)).
    returns
        DataFrame of per-ticker returns (aligned to `filtered_signals`).
    prob_weighting
        If True, weight candidates by predicted probability according to
        `WEIGHT_MODE` in config.
    target_vol
        Annualized target volatility; use -1 to disable vol targeting.
    leverage_cap
        Maximum allowed leverage; use -1 to disable cap.

    Returns
    -------
    port_returns, net_returns, weights_exec, turnover, costs
        Tuple containing portfolio returns (Series), net returns after
        costs (Series), executed weights (DataFrame), daily turnover
        (Series) and daily costs (Series).
    """

    # 1) Compute initial weights
    weights = filtered_signals.astype(float).copy()
    row_sums = weights.abs().sum(axis=1).replace(0, np.nan)
    weights = weights.div(row_sums, axis=0).fillna(0.0)
    # read weight mode early to avoid NameError in logs later
    mode = getattr(config, "WEIGHT_MODE", "prob")

    if prob_weighting:
        proba_array = np.asarray(clf.predict_proba(X_test))  # (n, 3)
        idx = get_class_to_index(clf)
        p1 = proba_array[:, idx[1]]
        p0 = proba_array[:, idx[0]]
        pm1 = proba_array[:, idx[-1]]

        signal_values = filtered_signals.stack().reindex(X_test.index).fillna(0.0).values

        if mode == "prob":
            raw_w = signal_values * p1
        elif mode == "margin":
            margin = np.maximum(0.0, p1 - np.maximum(p0, pm1))
            raw_w = signal_values * margin
        elif mode == "odds":
            eps = 1e-6
            p = np.clip(p1, eps, 1 - eps)
            logit = np.log(p) - np.log(1 - p)
            raw_w = signal_values * np.tanh(0.5 * logit)
        else:
            raw_w = signal_values * p1

        weights = (
            pd.Series(raw_w, index=X_test.index)
            .unstack()
            .reindex_like(filtered_signals)
            .fillna(0.0)
        )
        row_sums = weights.abs().sum(axis=1).replace(0, np.nan)
        weights = weights.div(row_sums, axis=0).fillna(0.0)

    assert (
        weights.shape == returns.shape
        and weights.index.equals(returns.index)
        and weights.columns.equals(returns.columns)
    )

    # 2) Vol targeting (based on forward PnL of target book)
    raw_returns = (returns.shift(-1) * weights).sum(axis=1)

    realized_vol = raw_returns.ewm(span=config.VOL_SPAN, adjust=False).std() * np.sqrt(252)
    if target_vol != -1:
        rv = realized_vol.shift(1).replace(0, np.nan)
        scaling = (target_vol / rv).clip(0.0, 3.0).fillna(0.0)
        # add inertia
        scaling = 0.9 * scaling.shift(1).fillna(1.0) + 0.1 * scaling
        weights = weights.mul(scaling, axis=0).fillna(0.0)

    # 3) Leverage cap
    leverage = weights.abs().sum(axis=1).replace(0, np.nan)
    if leverage_cap != -1:
        cap_ratio = (leverage_cap / leverage).clip(upper=1).fillna(1)
        weights = weights.mul(cap_ratio, axis=0)

    # 4) apply the λ-blend to the *final* target weights (reduces churn)
    lam = getattr(config, "LAMBDA_BLEND", 0.5)
    Wt = weights.to_numpy()
    out = np.zeros_like(Wt)
    prev = np.zeros(Wt.shape[1])
    for t in range(Wt.shape[0]):
        out[t] = prev + lam * (Wt[t] - prev)
        prev = out[t]
    weights = pd.DataFrame(out, index=weights.index, columns=weights.columns)

    # 5) Pathwise micro-trade filter to build executed weights
    def apply_min_trade_pathwise(W: pd.DataFrame, eps: float | None = None) -> pd.DataFrame:
        if eps is None:
            eps = getattr(config, "MIN_TRADE_EPS", 0.001)
        A = W.to_numpy(copy=True)
        T, N = A.shape
        prev = np.zeros(N, dtype=A.dtype)
        for t in range(T):
            d = A[t] - prev
            small = np.abs(d) < eps
            if small.any():
                A[t, small] = prev[small]  # keep previous executed weights
            prev = A[t]
        return pd.DataFrame(A, index=W.index, columns=W.columns)

    weights_exec = apply_min_trade_pathwise(weights)
    # zero out tiny executed weights
    prune_eps = getattr(config, "MIN_TRADE_EPS", 0.001)
    weights_exec = weights_exec.mask(weights_exec.abs() < prune_eps, 0.0)

    # 6) Costs: per-side turnover on executed book (no double count on flips)
    w_prev = weights_exec.shift().fillna(0.0)

    w_long = weights_exec.clip(lower=0.0)
    w_prev_long = w_prev.clip(lower=0.0)

    w_short_pos = (-weights_exec).clip(lower=0.0)
    w_prev_short_pos = (-w_prev).clip(lower=0.0)

    long_turn = (w_long - w_prev_long).abs().sum(axis=1)
    short_turn = (w_short_pos - w_prev_short_pos).abs().sum(axis=1)
    turnover = long_turn + short_turn

    long_tc = config.LONG_SIDE_TC  # e.g., 0.001
    short_tc = config.SHORT_SIDE_TC  # e.g., 0.002
    costs = long_turn * long_tc + short_turn * short_tc

    # 7) P&L on executed book (use forward returns)
    port_returns = (returns.shift(-1) * weights_exec).sum(axis=1)
    net_returns = port_returns - costs

    # 8) Log / return executed book
    strategy_type = "Long-Only" if config.LONG_ONLY else "Long/Short"
    logger.info("(%s) Strategy", strategy_type)
    logger.info("Probability weighting: %s, mode: %s", str(prob_weighting), str(mode))
    logger.info(
        "Vol target: %.2f%%, Realized vol (avg): %.2f%%",
        target_vol * 100.0,
        float(realized_vol.mean() * 100.0),
    )
    logger.info("Leverage cap: %.2f", float(leverage_cap))
    logger.info("Lambda blend: %.2f, micro-trade filter: %.2f", float(lam), float(prune_eps))
    avg_w = weights_exec[weights_exec != 0].mean().mean()
    logger.info("Average weight (non-zero): %.4f", float(avg_w if not np.isnan(avg_w) else 0.0))
    logger.info("Trades/day (avg): %.2f", float(weights_exec.astype(bool).sum(axis=1).mean()))
    logger.info("Total turnover: %.4f", float(turnover.sum()))
    logger.info(
        "Avg daily cost: %.5f, Annualized: %.2f%%",
        float(costs.mean()),
        (1 + float(costs.mean())) ** 252 - 1.0,
    )

    exec_lev = weights_exec.abs().sum(axis=1)
    logger.info(
        "Average leverage (exec): %.2f, Max leverage: %.2f",
        float(exec_lev.mean()),
        float(exec_lev.max()),
    )

    return port_returns, net_returns, weights_exec.fillna(0.0), turnover, costs
