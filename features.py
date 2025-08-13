import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import ta
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view

import config
from data_loader import (
    load_high_prices,
    load_low_prices,
    load_monthly_prices,
    load_prices,
    load_rates,
    load_returns,
    load_spy_prices,
    load_spy_returns,
    load_vix,
    load_volumes,
)
from labeling import apply_triple_barrier
from strategy import get_daily_signals

logger = logging.getLogger(__name__)


def build_features() -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Generate feature matrix X and binary outcome labels Y for meta-modeling.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix indexed by (date, ticker).
    Y : pd.Series
        Multiclass labels indicating trade success (1 = Successful trades, 0 = Timeout, -1 = Bad trades).
    label_times : pd.Series
        Time boundaries for labels, for model evaluation.
    """
    logger.info("Starting feature generation.")

    # Load data
    prices = load_prices()
    monthly_prices = load_monthly_prices()
    returns = load_returns()
    volumes = load_volumes()
    low = load_low_prices()
    high = load_high_prices()
    spy_prices = load_spy_prices()
    spy_returns = load_spy_returns()
    vix = load_vix()
    rates = load_rates()
    volatility = returns.rolling(63).std()

    # Labeling
    logger.info("Generating labels using triple barrier method.")
    daily_signals = get_daily_signals(
        prices, monthly_prices, long_only=config.LONG_ONLY
    )
    Y, label_times = apply_triple_barrier(prices, daily_signals, volatility)

    # Core price & return features
    logger.info("Computing core price and return features.")
    price_max_63d = prices.rolling(63).max()
    price_min_63d = prices.rolling(63).min()
    price_percentile = (prices - price_min_63d) / (price_max_63d - price_min_63d)

    volatility_63d = prices.pct_change(fill_method=None).rolling(63).std()
    volatility_21d = prices.pct_change(fill_method=None).rolling(21).std()
    volatility_zscore = (
        volatility_63d - volatility_63d.rolling(252).mean().shift(1)
    ) / (volatility_63d.rolling(252).std().shift(1))

    vov_63 = volatility_21d.rolling(63).std()

    price_max_252 = prices.rolling(252).max()
    price_min_252 = prices.rolling(252).min()
    dist_52w = (prices - price_min_252) / (price_max_252 - price_min_252)

    # Momentum
    logger.info("Computing momentum features.")
    momentum_63d = prices.pct_change(63, fill_method=None)
    momentum_21d = prices.pct_change(21, fill_method=None)
    momentum_63_21d = momentum_63d - momentum_21d
    vol_adj_momentum = momentum_63d / volatility_63d
    mom_persistence = (returns > 0).rolling(63).mean()

    mom_12m = prices.pct_change(252, fill_method=None)
    mom_1m = prices.pct_change(21, fill_method=None)
    mom_12m_1m = mom_12m - mom_1m

    # 63d max drawdown
    roll = prices.rolling(63)
    mdd_63 = prices / roll.max() - 1.0

    # Downside specific risk
    ret = prices.pct_change()
    dsv_63 = (ret.clip(upper=0) ** 2).rolling(63).mean()

    def ES95_optimized(x):
        q = np.percentile(x, 5)
        return -np.mean(x[x <= q])

    es95_63 = ret.rolling(63).apply(ES95_optimized, raw=True)

    dd = prices / prices.rolling(63).max() - 1.0
    dd_speed_21 = dd.diff().rolling(21).min()

    # Regime conditioned trend quality:

    def rolling_slope_r2(df: pd.DataFrame, window: int = 63):
        """
        Rolling slope & R² for each column using simple column-by-column apply.
        Works with DataFrame (dates x tickers).
        """
        if isinstance(df, pd.Series):
            df = df.to_frame()

        idx = np.arange(window, dtype=np.float64)
        xc = idx - idx.mean()
        Sxx = np.sum(xc**2)

        slopes = {}
        r2s = {}

        for col in df.columns:
            y = df[col].to_numpy(dtype=np.float64)
            slope_col = np.full_like(y, np.nan)
            r2_col = np.full_like(y, np.nan)

            for i in range(window - 1, len(y)):
                y_window = y[i - window + 1 : i + 1]
                if np.isnan(y_window).any():
                    continue

                yc = y_window - y_window.mean()
                beta = np.dot(xc, yc) / Sxx
                yhat = beta * xc
                sst = np.sum(yc**2)
                r2_val = 1 - np.sum((yc - yhat) ** 2) / sst if sst > 0 else 0

                slope_col[i] = beta
                r2_col[i] = r2_val

            slopes[col] = slope_col
            r2s[col] = r2_col

        return pd.DataFrame(slopes, index=df.index), pd.DataFrame(r2s, index=df.index)

    lp = np.log(prices)
    slope_63, r2_63 = rolling_slope_r2(lp, window=63)
    trend_strength_63 = slope_63 * r2_63

    vol_trend_63, _ = rolling_slope_r2(volatility_21d, window=63)

    time_from_high_63 = prices.rolling(63).apply(
        lambda x: len(x) - 1 - np.argmax(x), raw=True
    )
    time_from_low_63 = prices.rolling(63).apply(
        lambda x: len(x) - 1 - np.argmin(x), raw=True
    )
    time_from_high_252 = prices.rolling(252).apply(
        lambda x: len(x) - 1 - np.argmax(x), raw=True
    )
    time_from_low_252 = prices.rolling(252).apply(
        lambda x: len(x) - 1 - np.argmin(x), raw=True
    )

    # Macro
    logger.info("Computing macroeconomic features.")
    vix_feature = pd.DataFrame(
        np.tile(vix.values.reshape(-1, 1), (1, prices.shape[1])),
        index=prices.index,
        columns=prices.columns,
    )
    vix_high = pd.DataFrame(
        np.tile((vix > 25).astype(int).values.reshape(-1, 1), (1, prices.shape[1])),
        index=prices.index,
        columns=prices.columns,
    )
    ten_year = pd.DataFrame(
        np.tile(rates["DGS10"].values.reshape(-1, 1), (1, prices.shape[1])),
        index=prices.index,
        columns=prices.columns,
    )
    ten_year_minus = pd.DataFrame(
        np.tile(rates["T10Y3M"].values.reshape(-1, 1), (1, prices.shape[1])),
        index=prices.index,
        columns=prices.columns,
    )
    inversion = pd.DataFrame(
        np.tile(
            (rates["T10Y3M"] < 0).astype(int).values.reshape(-1, 1),
            (1, prices.shape[1]),
        ),
        index=prices.index,
        columns=prices.columns,
    )

    spy_returns_63d = spy_prices.pct_change(63, fill_method=None).to_frame()
    market_returns_zscore = (
        spy_returns_63d - spy_returns_63d.rolling(252).mean().shift(1)
    ) / (spy_returns_63d.rolling(252).std().shift(1))
    market_returns_zscore_feature = pd.DataFrame(
        np.tile(market_returns_zscore.values.reshape(-1, 1), (1, prices.shape[1])),
        index=prices.index,
        columns=prices.columns,
    )
    # market volatility z score
    spy_vol = spy_prices.pct_change(fill_method=None).rolling(63).std()
    market_vol_zscore = (spy_vol - spy_vol.rolling(252).mean().shift(1)) / (
        spy_vol.rolling(252).std().shift(1)
    )
    market_vol_zscore_feature = pd.DataFrame(
        np.tile(market_vol_zscore.values.reshape(-1, 1), (1, prices.shape[1])),
        index=prices.index,
        columns=prices.columns,
    )

    # Correlation features
    logger.info("Computing correlation features.")
    beta_63d = pd.DataFrame(index=prices.index, columns=prices.columns)
    corr_spy_63d = pd.DataFrame(index=prices.index, columns=prices.columns)
    for ticker in tqdm(prices.columns, desc="Computing Beta & Corr SPY"):
        r = returns[ticker]
        beta_63d[ticker] = (
            r.rolling(63).cov(spy_returns) / spy_returns.rolling(63).var()
        )
        corr_spy_63d[ticker] = r.rolling(63).corr(spy_returns)

    beta_21 = ret.apply(
        lambda col: col.rolling(21).cov(spy_returns) / spy_returns.rolling(21).var(),
        axis=0,
    )
    beta_instability_63 = beta_21.rolling(63).std()

    # Calendar features
    logger.info("Computing calendar features.")
    month_of_year_sin = (
        pd.DataFrame(np.sin(2 * np.pi * prices.index.month / 12), index=prices.index)
        .reindex(prices.index)
        .ffill()
    )
    month_of_year_sin = pd.concat([month_of_year_sin] * len(prices.columns), axis=1)
    month_of_year_sin.columns = prices.columns

    is_month_end = pd.DataFrame(0, index=prices.index, columns=prices.columns)
    is_month_end.loc[
        prices.index.to_series().groupby(prices.index.to_period("M")).last().values
    ] = 1

    is_month_start = pd.DataFrame(0, index=prices.index, columns=prices.columns)
    is_month_start.loc[
        prices.index.to_series().groupby(prices.index.to_period("M")).first().values
    ] = 1

    # Volume-based features
    logger.info("Computing volume-based features.")
    volume_surge = volumes / volumes.rolling(63).mean()

    # Skew-based features
    logger.info("Computing skew-based features.")
    return_skew_21d = returns.rolling(21).skew()
    return_skew_63d = returns.rolling(63).skew()
    volume_skew_21d = volumes.rolling(21).skew()

    amihud_illiquidity = np.log(1e-10 + (returns.abs() / volumes).rolling(63).mean())

    logger.info("Computing cross-sectional features.")

    def xsrank(df: pd.DataFrame) -> pd.DataFrame:
        return df.rank(pct=True, axis=1, method="average")

    def xszscore(df):
        m = df.mean(axis=1).values[:, None]
        s = df.std(axis=1).values[:, None]
        return (df - m) / (s + 1e-9)

    def compute_ta_indicators(prices, high, low, volumes) -> Dict[str, pd.DataFrame]:
        """
        Computes various TA indicators from the `ta` package in a flexible way.

        Returns
        -------
        indicators : dict[str, pd.DataFrame]
            Dictionary of indicator name -> DataFrame with shape (dates x tickers).
        """

        indicator_specs = [
            # Momentum
            {
                "name": "rsi",
                "class": ta.momentum.RSIIndicator,
                "method": "rsi",
                "inputs": ["close", "window"],
            },
            {
                "name": "roc",
                "class": ta.momentum.ROCIndicator,
                "method": "roc",
                "inputs": ["close", "window"],
            },
            {
                "name": "trix",
                "class": ta.trend.TRIXIndicator,
                "method": "trix",
                "inputs": ["close", "window"],
            },
            {
                "name": "macd",
                "class": ta.trend.MACD,
                "method": "macd",
                "inputs": ["close", "window_slow", "window_fast", "window_sign"],
            },
            # Trend
            {
                "name": "adx",
                "class": ta.trend.ADXIndicator,
                "method": "adx",
                "inputs": ["high", "low", "close", "window"],
            },
            {
                "name": "cci",
                "class": ta.trend.CCIIndicator,
                "method": "cci",
                "inputs": ["high", "low", "close", "window"],
            },
            {
                "name": "sma",
                "class": ta.trend.SMAIndicator,
                "method": "sma_indicator",
                "inputs": ["close", "window"],
            },
            {
                "name": "ema",
                "class": ta.trend.EMAIndicator,
                "method": "ema_indicator",
                "inputs": ["close", "window"],
            },
            {
                "name": "kama",
                "class": ta.momentum.KAMAIndicator,
                "method": "kama",
                "inputs": ["close", "window"],
            },
            # Oscillators
            {
                "name": "stoch",
                "class": ta.momentum.StochasticOscillator,
                "method": "stoch",
                "inputs": ["high", "low", "close", "window"],
            },
            {
                "name": "williams_r",
                "class": ta.momentum.WilliamsRIndicator,
                "method": "williams_r",
                "inputs": ["high", "low", "close", "lbp"],
            },
            # Volume
            {
                "name": "obv",
                "class": ta.volume.OnBalanceVolumeIndicator,
                "method": "on_balance_volume",
                "inputs": ["close", "volume"],
            },
            {
                "name": "vwap",
                "class": ta.volume.VolumeWeightedAveragePrice,
                "method": "volume_weighted_average_price",
                "inputs": ["high", "low", "close", "volume", "window"],
            },
            {
                "name": "cmf",
                "class": ta.volume.ChaikinMoneyFlowIndicator,
                "method": "chaikin_money_flow",
                "inputs": ["high", "low", "close", "volume", "window"],
            },
            {
                "name": "adi",
                "class": ta.volume.AccDistIndexIndicator,
                "method": "acc_dist_index",
                "inputs": ["high", "low", "close", "volume"],
            },
            # Volatility
            {
                "name": "atr",
                "class": ta.volatility.AverageTrueRange,
                "method": "average_true_range",
                "inputs": ["high", "low", "close", "window"],
            },
        ]

        indicators = {}
        for spec in tqdm(indicator_specs, "Computing TA indicators"):
            df = pd.DataFrame(index=prices.index, columns=prices.columns)
            for ticker in prices.columns:
                # Gather input series
                input_series = {
                    "close": prices[ticker],
                    "high": high.get(ticker),
                    "low": low.get(ticker),
                    "volume": volumes.get(ticker),
                    "window": 63,
                    "window_slow": 126,
                    "window_fast": 63,
                    "window_sign": 32,
                    "lbp": 63,
                }
                # Filter only the required inputs
                inputs = {k: v for k, v in input_series.items() if k in spec["inputs"]}
                try:
                    indicator = spec["class"](**inputs)
                    df[ticker] = getattr(indicator, spec["method"])()
                except Exception as e:
                    logger.warning(
                        f"Failed to compute {spec['name']} for {ticker}: {e}"
                    )
                    df[ticker] = np.nan
            indicators[spec["name"]] = df

        return indicators

    ta_indicators = compute_ta_indicators(prices, high, low, volumes)

    # Combine all features
    logger.info("Combining all features.")
    features = {
        "price_percentile": price_percentile,
        "market_returns_zscore": market_returns_zscore_feature,
        "market_vol_zscore": market_vol_zscore_feature,
        "volatility_63d": volatility_63d,
        "volatility_21d": volatility_21d,
        "vov_63": vov_63,
        "vol_trend_63": vol_trend_63,
        "price_max_63d": price_max_63d,
        "price_min_63d": price_min_63d,
        "dist_52w": dist_52w,
        "mom_12m": mom_12m,
        "mom_1m": mom_1m,
        "mom_12m_1m": mom_12m_1m,
        "mdd_63": mdd_63,
        "dsv_63": dsv_63,
        "es95_63": es95_63,
        "dd_speed_21": dd_speed_21,
        "trend_strength_63": trend_strength_63,
        "time_from_high_63": time_from_high_63,
        "time_from_low_63": time_from_low_63,
        "time_from_high_252": time_from_high_252,
        "time_from_low_252": time_from_low_252,
        "vix": vix_feature,
        "vix_high": vix_high,
        "beta_63d": beta_63d,
        "beta_instability_63": beta_instability_63,
        "corr_spy_63d": corr_spy_63d,
        "volume": volumes,
        "amihud_illiquidity": amihud_illiquidity,
        "month_of_year_sin": month_of_year_sin,
        "is_month_end": is_month_end,
        "is_month_start": is_month_start,
        "bollinger_zscore": (prices - prices.rolling(63).mean())
        / prices.rolling(63).std(),
        "volatility_zscore": volatility_zscore,
        "10yTbill": ten_year,
        "yield_curve_slope": ten_year_minus,
        "yc_inversion": inversion,
        "vol_adj_momentum": vol_adj_momentum,
        "momentum_21d": momentum_21d,
        "momentum_63d": momentum_63d,
        "momentum_63_21d": momentum_63_21d,
        "mom_persistence_63d": mom_persistence,
        "volume_surge": volume_surge,
        "return_skew_21d": return_skew_21d,
        "return_skew_63d": return_skew_63d,
        "volume_skew_21d": volume_skew_21d,
        "cs_rank_mom21d": xsrank(momentum_21d),
        "cs_rank_mom63d": xsrank(momentum_63d),
        "cs_rank_price_percentile": xsrank(price_percentile),
        "cs_rank_volume_surge": xsrank(volume_surge),
        "cs_rank_vol_adj_momentum": xsrank(vol_adj_momentum),
        "cs_z_volatility": xszscore(volatility_63d),
        "cs_z_beta": xszscore(beta_63d),
        "cs_z_corr_spy": xszscore(corr_spy_63d),
        "cs_z_rsi": xszscore(ta_indicators["rsi"]),
        "cs_z_obv": xszscore(ta_indicators["obv"]),
    }

    # Add TA indicators to features
    features.update(ta_indicators)

    # Build final feature matrix
    logger.info("Building final feature matrix.")
    X_rows = []
    for date, ticker in tqdm(Y.stack().index, desc="Building X"):
        row = {"date": date, "ticker": ticker}
        for fname, fmat in features.items():
            value = (
                fmat.at[date, ticker]
                if (
                    date in fmat.index
                    and ticker in fmat.columns
                    and not pd.isna(fmat.at[date, ticker])
                )
                else np.nan
            )
            row[fname] = value
        X_rows.append(row)

    X = pd.DataFrame(X_rows).set_index(["date", "ticker"])
    X = X.replace([np.inf, -np.inf], np.nan)
    # Drop worst rows (≥10% NaNs)
    X = X.dropna(thresh=int(0.9 * X.shape[1]))
    # Forward + Backward fill
    X = (
        X.groupby(level=1)
        .apply(lambda g: g.ffill(limit=10))
        .reset_index(level=0, drop=True)
    )

    # Safe fallback for sparse leftovers
    X = X.fillna(X.median())
    # Sort for consistency with Y
    X = X.sort_index(level=[0, 1])

    # Correlation pruning
    logger.info("Pruning highly correlated features.")
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]

    # Dropping bad features
    to_drop.append("is_month_start")
    to_drop.append("is_month_end")
    to_drop.append("yc_inversion")
    X = X.drop(columns=to_drop)
    logger.info(f"Features dropped due to high correlation: {to_drop}")

    Y = Y.stack().dropna().astype(int)
    X = X.reindex(Y.index).dropna()
    Y = Y.loc[X.index]

    logger.info("Feature generation completed successfully.")
    return X, Y, label_times


def build_meta_features(
    X_base: pd.DataFrame,
    base_model_lgbm,  # expects .predict_proba(), .predict(), and either .classes_ or .class_labels_
    base_model_mlp,  # your (Rolling)VectorScaledSoftmax wrapper with .predict_proba/.predict and .class_labels_
    class_order=(-1, 0, 1),
    regime_cols=(
        "vix",
        "volatility_zscore",
        "volatility_63d",
        "vix_high",
        "10yTbill",
        "yield_curve_slope",
        "yc_inversion",
        "month_of_year_sin",
    ),
) -> pd.DataFrame:
    """
    Build stacking meta-features from two base models (LGBM, MLP).
    Leakage-free if the supplied models are OOS for X_base.
    """

    # --- helper to align proba columns to a fixed class order ---
    def _proba_df(model, X, prefix, class_order):
        # figure out label ordering exposed by the model
        labels = getattr(model, "class_labels_", getattr(model, "classes_", None))
        if labels is None:
            raise ValueError(f"{prefix}: model must expose classes_ or class_labels_.")
        idx_map = {int(c): i for i, c in enumerate(labels)}
        P = model.predict_proba(X)
        # reorder columns to (class_order)
        cols = [f"{prefix}_{c}" for c in class_order]
        mat = np.column_stack([P[:, idx_map[c]] for c in class_order])
        return pd.DataFrame(mat, index=X.index, columns=cols)

    # --- base predictions (calibrated or raw depending on the wrapper passed) ---
    proba_clf = _proba_df(base_model_lgbm, X_base, "proba_clf", class_order)
    proba_mlp = _proba_df(base_model_mlp, X_base, "proba_mlp", class_order)

    clf_pred = pd.Series(
        base_model_lgbm.predict(X_base), index=X_base.index, name="clf_pred"
    )
    mlp_pred = pd.Series(
        base_model_mlp.predict(X_base), index=X_base.index, name="mlp_pred"
    )

    # --- utilities (vectorized) ---
    eps = 1e-12

    def entropy(df):  # per-row entropy
        A = df.values
        return pd.Series(-(A * np.log(A + eps)).sum(1), index=df.index)

    def top2_gap(df):  # max - 2nd max
        A = df.values
        part = np.partition(A, -2, axis=1)
        return pd.Series(part[:, -1] - part[:, -2], index=df.index)

    # columns aligned to (-1, 0, 1)
    c_m1, c_0, c_p1 = (
        f"proba_clf_{class_order[0]}",
        f"proba_clf_{class_order[1]}",
        f"proba_clf_{class_order[2]}",
    )
    m_m1, m_0, m_p1 = (
        f"proba_mlp_{class_order[0]}",
        f"proba_mlp_{class_order[1]}",
        f"proba_mlp_{class_order[2]}",
    )

    # --- core meta features ---
    proba_gap_clf = (proba_clf[c_p1] - proba_clf[[c_0, c_m1]].max(1)).rename(
        "proba_gap_clf"
    )
    proba_gap_mlp = (proba_mlp[m_p1] - proba_mlp[[m_0, m_m1]].max(1)).rename(
        "proba_gap_mlp"
    )
    model_agreement = (clf_pred.values == mlp_pred.values).astype(int)
    model_agreement = pd.Series(
        model_agreement, index=X_base.index, name="model_agreement"
    )
    confidence_agreement = (
        (proba_clf[c_p1] - proba_mlp[m_p1]).abs().rename("confidence_agreement")
    )
    confidence_mean = ((proba_clf[c_p1] + proba_mlp[m_p1]) * 0.5).rename(
        "confidence_mean"
    )
    entropy_clf = entropy(proba_clf).rename("entropy_clf")
    entropy_mlp = entropy(proba_mlp).rename("entropy_mlp")

    # Directional scores (useful for trade side decisions)
    dir_clf = (proba_clf[c_p1] - proba_clf[c_m1]).rename("dir_clf")  # P(+1) - P(-1)
    dir_mlp = (proba_mlp[m_p1] - proba_mlp[m_m1]).rename("dir_mlp")

    # Non-timeout confidence
    nt_clf = (1.0 - proba_clf[c_0]).rename("non_timeout_clf")
    nt_mlp = (1.0 - proba_mlp[m_0]).rename("non_timeout_mlp")

    # Log-odds ratios (more linear for meta learners)
    logodds_clf = (
        np.log(proba_clf[c_p1] + eps) - np.log(proba_clf[c_m1] + eps)
    ).rename("logodds_clf")
    logodds_mlp = (
        np.log(proba_mlp[m_p1] + eps) - np.log(proba_mlp[m_m1] + eps)
    ).rename("logodds_mlp")

    # Margin (max minus second max) per model
    margin_clf = top2_gap(proba_clf).rename("margin_clf")
    margin_mlp = top2_gap(proba_mlp).rename("margin_mlp")

    # Disagreement intensity (JS divergence between the two distributions)
    P = proba_clf.values
    Q = proba_mlp.values
    M = 0.5 * (P + Q)
    js_div = 0.5 * (
        (P * np.log((P + eps) / (M + eps))).sum(1)
        + (Q * np.log((Q + eps) / (M + eps))).sum(1)
    )
    js_div = pd.Series(js_div, index=X_base.index, name="js_div")

    # Simple blend (acts like an ensemble baseline and a meta feature)
    blend_p1 = (proba_clf[c_p1] + proba_mlp[m_p1]) * 0.5
    blend_p1 = blend_p1.rename("blend_p1")

    # --- regime features (safe selection and fill) ---
    regime = []
    for col in regime_cols:
        if col in X_base.columns:
            regime.append(X_base[col].astype(float).rename(col))
    if regime:
        regime = (
            pd.concat(regime, axis=1)
            .reindex(X_base.index)
            .fillna(method="ffill")
            .fillna(0.0)
        )
    else:
        regime = pd.DataFrame(index=X_base.index)

    # --- assemble ---
    frames = [
        proba_clf,
        proba_mlp,
        clf_pred.rename("clf_pred"),
        mlp_pred.rename("mlp_pred"),
        proba_gap_clf,
        proba_gap_mlp,
        entropy_clf,
        entropy_mlp,
        model_agreement,
        confidence_agreement,
        confidence_mean,
        dir_clf,
        dir_mlp,
        nt_clf,
        nt_mlp,
        logodds_clf,
        logodds_mlp,
        margin_clf,
        margin_mlp,
        js_div,
        blend_p1,
        regime,
    ]
    X_meta = pd.concat(frames, axis=1)

    # basic hygiene
    X_meta.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_meta.fillna(0.0, inplace=True)

    logger.info(f"X_meta shape: {X_meta.shape}")
    return X_meta


def main():
    logger.info("Starting main feature generation pipeline.")
    X, Y, label_times = build_features()
    X.to_parquet(config.X)
    Y.to_frame().to_parquet(config.Y)
    logger.info(f"Saved features to {config.X}")
    logger.info(f"Saved labels to {config.Y}")
    logger.info("Feature generation pipeline completed.")


if __name__ == "__main__":
    main()
