import logging

import numpy as np
import pandas as pd
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
from strategy import get_daily_signals
from modeling import scale_features

logger = logging.getLogger(__name__)


def build_features() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Generate feature matrix X and binary outcome labels Y for meta-modeling.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix indexed by (date, ticker)
    Y : pd.Series
        Binary labels indicating trade success (1 = TP hit, 0 = SL hit)
    label_times : pd.Series
        Time boundaries for labels, for model evaluation
    """

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
    daily_signals, signal_dates = get_daily_signals(prices, monthly_prices)
    Y, label_times = apply_triple_barrier(prices, daily_signals, volatility)

    # Core price & return features
    log_prices = np.log(prices)
    returns_20d = prices.pct_change(20, fill_method=None)
    price_max_20d = prices.rolling(20).max()
    price_min_20d = prices.rolling(20).min()
    price_percentile = (prices - price_min_20d) / (price_max_20d - price_min_20d)
    volatility_20d = prices.pct_change(fill_method=None).rolling(20).std()
    volatility_60d = prices.pct_change(fill_method=None).rolling(60).std()
    volatility_zscore = (volatility_20d - volatility_20d.mean()) / volatility_20d.std()

    # Momentum
    momentum_10d = prices.pct_change(10, fill_method=None)
    momentum_20d = prices.pct_change(20, fill_method=None)
    momentum_20_5d = prices.pct_change(20, fill_method=None) - prices.pct_change(
        5, fill_method=None
    )
    vol_adj_momentum = momentum_20d / volatility_20d
    mom_persistence = (returns_20d > 0).rolling(20).mean()

    # Macro
    vix_feature = pd.DataFrame(
        np.tile(vix.values.reshape(-1, 1), (1, prices.shape[1])),
        index=prices.index,
        columns=prices.columns,
    )
    rates_feature = pd.DataFrame(
        np.tile(rates.values.reshape(-1, 1), (1, prices.shape[1])),
        index=prices.index,
        columns=prices.columns,
    )

    spy_returns_5d = spy_prices.pct_change(5, fill_method=None).to_frame()
    spy_returns_5d_feature = pd.DataFrame(
        np.tile(spy_returns_5d.values.reshape(-1, 1), (1, prices.shape[1])),
        index=prices.index,
        columns=prices.columns,
    )

    # Correlation features
    beta_20d = pd.DataFrame(index=prices.index, columns=prices.columns)
    corr_spy_20d = pd.DataFrame(index=prices.index, columns=prices.columns)
    for ticker in tqdm(prices.columns, desc="Computing Beta & Corr SPY"):
        r = returns[ticker]
        beta_20d[ticker] = (
            r.rolling(20).cov(spy_returns) / spy_returns.rolling(20).var()
        )
        corr_spy_20d[ticker] = r.rolling(20).corr(spy_returns)

    # Calendar features
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
    volume_surge = volumes / volumes.rolling(20).mean()

    def compute_ta_indicators(prices, high, low, volumes) -> dict:
        """
        Computes various TA indicators from the `ta` package in a flexible way.

        Returns
        -------
        indicators : dict[str, pd.DataFrame]
            Dictionary of indicator name -> DataFrame with shape (dates x tickers)
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
        for spec in tqdm(indicator_specs, "Computing TA indicators:"):
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
                    print(f"Failed to compute {spec['name']} for {ticker}: {e}")
                    df[ticker] = np.nan
            indicators[spec["name"]] = df

        return indicators

    # Combine all features
    features = {
        "log_prices": log_prices,
        "returns_20d": returns_20d,
        "price_percentile": price_percentile,
        "spy_returns_5d": spy_returns_5d_feature,
        "volatility_20d": volatility_20d,
        "volatility_60d": volatility_60d,
        "vix": vix_feature,
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
        "10yTbill": rates_feature,
        "vol_adj_momentum": vol_adj_momentum,
        "momentum_10d": momentum_10d,
        "momentum_20d": momentum_20d,
        "momentum_20_5d": momentum_20_5d,
        "mom_persistence_20d": mom_persistence,
        "volume_surge": volume_surge,
    }

    ta_indicators = compute_ta_indicators(prices, high, low, volumes)

    # Then add to your `features` dictionary
    features.update(ta_indicators)

    # Build final feature matrix
    X_rows = []
    for date, ticker in tqdm(signal_dates, desc="Building X"):
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

    X = pd.DataFrame(X_rows).set_index(["date", "ticker"]).dropna()

    # Correlation prunning
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
    # Dropping bad features
    to_drop.append("is_month_start")
    to_drop.append("is_month_end")
    X = X.drop(columns=to_drop)

    Y = Y.stack().dropna().astype(int)
    X = X.reindex(Y.index).dropna()
    Y = Y.loc[X.index]

    return X, Y, label_times


def main():
    X, Y, label_times = build_features()
    X.to_parquet(config.X)
    Y.to_frame().to_parquet(config.Y)
    logger.info(f"Saved features to {config.X}")
    logger.info(f"Saved labels to {config.Y}")


if __name__ == "__main__":
    main()



def build_meta_features(
        X_base: pd.DataFrame,
        base_model_lgbm,
        base_model_mlp,
        scale: bool = True,
    ) -> pd.DataFrame:
        """
        Build meta features by augmenting original features with:
        - LGBM predicted probabilities
        - MLP predicted probabilities
        - LGBM predicted class
        - MLP predicted class

        Args:
            X_base (pd.DataFrame): Input features to apply base models on.
            base_model_lgbm: Calibrated LightGBM model (must have .predict_proba and .predict).
            base_model_mlp: KerasSoftmaxWrapper for MLP model.
            scale (bool): Whether to standard scale the result.

        Returns:
            pd.DataFrame: Augmented meta feature set.
        """

        # LGBM Probabilities
        P_lgbm = pd.DataFrame(
            base_model_lgbm.predict_proba(X_base),
            index=X_base.index,
            columns=[f"proba_clf_{cls}" for cls in base_model_lgbm.classes_]
        )

        # MLP Probabilities
        P_mlp = pd.DataFrame(
            base_model_mlp.predict_proba(X_base),
            index=X_base.index,
            columns=[f"proba_mlp_{cls}" for cls in base_model_mlp.class_labels_]
        )

        # Predicted classes
        Pred_lgbm = pd.DataFrame(
            base_model_lgbm.predict(X_base),
            index=X_base.index,
            columns=["clf_pred"]
        )

        Pred_mlp = pd.DataFrame(
            base_model_mlp.predict(X_base),
            index=X_base.index,
            columns=["mlp_pred"]
        )

        # Concatenate original features and meta-features
        X_meta = pd.concat([X_base, P_lgbm, P_mlp, Pred_lgbm, Pred_mlp], axis=1)

        if scale:
            X_meta = scale_features(X_meta)

        return X_meta
