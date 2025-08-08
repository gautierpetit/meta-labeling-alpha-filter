import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
import ta
from tqdm import tqdm

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
from modeling import scale_features
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
        Binary labels indicating trade success (1 = TP hit, 0 = SL hit).
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
    volatility = returns.rolling(20).std()

    # Labeling
    logger.info("Generating labels using triple barrier method.")
    daily_signals = get_daily_signals(
        prices, monthly_prices, long_only=config.LONG_ONLY
    )
    Y, label_times = apply_triple_barrier(prices, daily_signals, volatility)

    # Core price & return features
    logger.info("Computing core price and return features.")
    log_prices = np.log(prices)
    returns_20d = prices.pct_change(20, fill_method=None)
    price_max_20d = prices.rolling(20).max()
    price_min_20d = prices.rolling(20).min()
    price_percentile = (prices - price_min_20d) / (price_max_20d - price_min_20d)
    volatility_20d = prices.pct_change(fill_method=None).rolling(20).std()
    volatility_60d = prices.pct_change(fill_method=None).rolling(60).std()
    volatility_zscore = (volatility_20d - volatility_20d.mean()) / volatility_20d.std()

    # Momentum
    logger.info("Computing momentum features.")
    momentum_10d = prices.pct_change(10, fill_method=None)
    momentum_20d = prices.pct_change(20, fill_method=None)
    momentum_20_5d = prices.pct_change(20, fill_method=None) - prices.pct_change(
        5, fill_method=None
    )
    vol_adj_momentum = momentum_20d / volatility_20d
    mom_persistence = (returns_20d > 0).rolling(20).mean()

    # Macro
    logger.info("Computing macroeconomic features.")
    vix_feature = pd.DataFrame(
        np.tile(vix.values.reshape(-1, 1), (1, prices.shape[1])),
        index=prices.index,
        columns=prices.columns,
    )
    vix_high = pd.DataFrame(
        np.tile((vix>25).astype(int).values.reshape(-1, 1), (1, prices.shape[1])),
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

    spy_returns_5d = spy_prices.pct_change(5, fill_method=None).to_frame()
    spy_returns_5d_feature = pd.DataFrame(
        np.tile(spy_returns_5d.values.reshape(-1, 1), (1, prices.shape[1])),
        index=prices.index,
        columns=prices.columns,
    )

    spy_returns_10d = spy_prices.pct_change(10, fill_method=None)
    market_returns_zscore = (spy_returns_10d - spy_returns_10d.mean()) / spy_returns_10d.std()
    market_returns_zscore_feature = pd.DataFrame(
        np.tile(market_returns_zscore.values.reshape(-1, 1), (1, prices.shape[1])),
        index=prices.index,
        columns=prices.columns,
    )
    # market volatility z score
    spy_vol = spy_prices.pct_change(fill_method=None).rolling(30).std()
    market_vol_zscore = (spy_vol - spy_vol.mean()) / spy_vol.std()
    market_vol_zscore_feature = pd.DataFrame(
        np.tile(market_vol_zscore.values.reshape(-1, 1), (1, prices.shape[1])),
        index=prices.index,
        columns=prices.columns,
    )

    # Correlation features
    logger.info("Computing correlation features.")
    beta_20d = pd.DataFrame(index=prices.index, columns=prices.columns)
    corr_spy_20d = pd.DataFrame(index=prices.index, columns=prices.columns)
    for ticker in tqdm(prices.columns, desc="Computing Beta & Corr SPY"):
        r = returns[ticker]
        beta_20d[ticker] = (
            r.rolling(20).cov(spy_returns) / spy_returns.rolling(20).var()
        )
        corr_spy_20d[ticker] = r.rolling(20).corr(spy_returns)

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
    volume_surge = volumes / volumes.rolling(20).mean()

    # Skew-based features
    logger.info("Computing skew-based features.")
    return_skew_20d = returns.rolling(20).skew()
    return_skew_60d = returns.rolling(60).skew()
    volume_skew_20d = volumes.rolling(20).skew()

    logger.info("Computing cross-sectional features.")

    def xsrank(df: pd.DataFrame) -> pd.DataFrame:
        return df.rank(pct=True, axis=1, method="average")

    def xszscore(df: pd.DataFrame) -> pd.DataFrame:
        return (df - df.mean(axis=1).values[:, None]) / df.std(axis=1).values[:, None]

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
                "inputs": ["close"],
            },
            {
                "name": "roc",
                "class": ta.momentum.ROCIndicator,
                "method": "roc",
                "inputs": ["close"],
            },
            {
                "name": "trix",
                "class": ta.trend.TRIXIndicator,
                "method": "trix",
                "inputs": ["close"],
            },
            {
                "name": "macd",
                "class": ta.trend.MACD,
                "method": "macd",
                "inputs": ["close"],
            },
            # Trend
            {
                "name": "adx",
                "class": ta.trend.ADXIndicator,
                "method": "adx",
                "inputs": ["high", "low", "close"],
            },
            {
                "name": "cci",
                "class": ta.trend.CCIIndicator,
                "method": "cci",
                "inputs": ["high", "low", "close"],
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
                "inputs": ["close"],
            },
            {
                "name": "kama",
                "class": ta.momentum.KAMAIndicator,
                "method": "kama",
                "inputs": ["close"],
            },
            # Oscillators
            {
                "name": "stoch",
                "class": ta.momentum.StochasticOscillator,
                "method": "stoch",
                "inputs": ["high", "low", "close"],
            },
            {
                "name": "williams_r",
                "class": ta.momentum.WilliamsRIndicator,
                "method": "williams_r",
                "inputs": ["high", "low", "close"],
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
                "inputs": ["high", "low", "close", "volume"],
            },
            {
                "name": "cmf",
                "class": ta.volume.ChaikinMoneyFlowIndicator,
                "method": "chaikin_money_flow",
                "inputs": ["high", "low", "close", "volume"],
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
                "inputs": ["high", "low", "close"],
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
                    "window": 14,
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
        "log_prices": log_prices,
        "returns_20d": returns_20d,
        "price_percentile": price_percentile,
        "spy_returns_5d": spy_returns_5d_feature,
        "market_returns_zscore": market_returns_zscore_feature,
        "market_vol_zscore": market_vol_zscore_feature,
        "volatility_20d": volatility_20d,
        "volatility_60d": volatility_60d,
        "vix": vix_feature,
        "vix_high": vix_high,
        "beta_20d": beta_20d,
        "corr_spy_20d": corr_spy_20d,
        "volume": volumes,
        "amihud_illiquidity": (returns.abs() / volumes).rolling(5).mean(),
        "month_of_year_sin": month_of_year_sin,
        "is_month_end": is_month_end,
        "is_month_start": is_month_start,
        "bollinger_zscore": (prices - prices.rolling(20).mean())
        / prices.rolling(20).std(),
        "volatility_zscore": volatility_zscore,
        "10yTbill": ten_year,
        "yield_curve_slope": ten_year_minus,
        "yc_inversion": inversion,
        "vol_adj_momentum": vol_adj_momentum,
        "momentum_10d": momentum_10d,
        "momentum_20d": momentum_20d,
        "momentum_20_5d": momentum_20_5d,
        "mom_persistence_20d": mom_persistence,
        "volume_surge": volume_surge,
        "return_skew_20d": return_skew_20d,
        "return_skew_60d": return_skew_60d,
        "volume_skew_20d": volume_skew_20d,
        "cs_rank_mom10d": xsrank(momentum_10d),
        "cs_rank_mom20d": xsrank(momentum_20d),
        "cs_rank_price_percentile": xsrank(price_percentile),
        "cs_rank_volume_surge": xsrank(volume_surge),
        "cs_rank_vol_adj_momentum": xsrank(vol_adj_momentum),
        "cs_z_volatility": xszscore(volatility_20d),
        "cs_z_beta": xszscore(beta_20d),
        "cs_z_corr_spy": xszscore(corr_spy_20d),
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
    X = (
        X.groupby(level=1)
        .apply(lambda g: g.bfill(limit=5))
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
    X = X.drop(columns=to_drop)
    logger.info(f"Features dropped due to high correlation: {to_drop}")

    Y = Y.stack().dropna().astype(int)
    X = X.reindex(Y.index).dropna()
    Y = Y.loc[X.index]

    logger.info("Feature generation completed successfully.")
    return X, Y, label_times


def build_meta_features(
    X_base: pd.DataFrame,
    scaler: StandardScaler,
    daily_signals: pd.DataFrame,
    base_model_lgbm,
    base_model_mlp,
    
) -> pd.DataFrame:
    """
    Build meta features using:
    - First-stage model predictions
    - Meta-level engineered features
    - Market regime context
    - Raw signal direction
    """
    X_base_scaled = pd.DataFrame(scaler.transform(X_base), index=X_base.index, columns=X_base.columns)

    logger.info("First-stage model predictions.")
    proba_clf = pd.DataFrame(
        base_model_lgbm.predict_proba(X_base),
        index=X_base.index,
        columns=[f"proba_clf_{cls}" for cls in base_model_lgbm.classes_],
    )
    proba_mlp = pd.DataFrame(
        base_model_mlp.predict_proba(X_base_scaled),
        index=X_base.index,
        columns=[f"proba_mlp_{cls}" for cls in base_model_mlp.class_labels_],
    )
    clf_pred = pd.DataFrame(
        base_model_lgbm.predict(X_base), index=X_base.index, columns=["clf_pred"]
    )
    mlp_pred = pd.DataFrame(
        base_model_mlp.predict(X_base_scaled), index=X_base.index, columns=["mlp_pred"]
    )

    logger.info("Engineering meta features.")
    proba_gap_clf = proba_clf["proba_clf_1"] - proba_clf[
        ["proba_clf_0", "proba_clf_-1"]
    ].max(axis=1)
    proba_gap_mlp = proba_mlp["proba_mlp_1"] - proba_mlp[
        ["proba_mlp_0", "proba_mlp_-1"]
    ].max(axis=1)
    model_agreement = (
        (clf_pred["clf_pred"] == mlp_pred["mlp_pred"])
        .astype(int)
        .rename("model_agreement")
    )
    confidence_agreement = (
        (proba_clf["proba_clf_1"] - proba_mlp["proba_mlp_1"])
        .abs()
        .rename("confidence_agreement")
    )
    confidence_mean = (
        (proba_clf["proba_clf_1"] + proba_mlp["proba_mlp_1"]) / 2
    ).rename("confidence_mean")


    logger.info("Combining features.")
    feature_frames = [
        proba_clf,
        proba_mlp,
        clf_pred,
        mlp_pred,
        proba_gap_clf.rename("proba_gap_clf"),
        proba_gap_mlp.rename("proba_gap_mlp"),
        model_agreement,
        confidence_agreement,
        confidence_mean,
        X_base["vix"].loc[X_base.index],
        X_base["volatility_zscore"].loc[X_base.index],
        X_base["volatility_20d"].loc[X_base.index],
        X_base["vix_high"].loc[X_base.index],
        X_base["10yTbill"].loc[X_base.index],
        X_base["yield_curve_slope"].loc[X_base.index],
        X_base["yc_inversion"].loc[X_base.index],
        X_base["month_of_year_sin"].loc[X_base.index],
        daily_signals.stack().reindex_like(X_base).rename("raw_signal_direction"),
    ]

    X_meta = pd.concat(feature_frames, axis=1)

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
