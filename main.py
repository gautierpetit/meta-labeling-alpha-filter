"""
Main script for running the meta-labeling alpha filter pipeline.

This script orchestrates the entire workflow, including data loading, feature engineering,
model training, evaluation, and backtesting.

Modules:
- Data loading
- Feature engineering
- Model training and evaluation
- Strategy backtesting

Author: Gautier Petit
Date: August 2, 2025
"""

import logging
import warnings
from time import time

import joblib
import tensorflow as tf
from keras.models import load_model

import config
from config_private import NTFY_SERVER
from data_loader import (
    load_features,
    load_labels,
    load_monthly_prices,
    load_prices,
    load_returns,
    load_spy_returns,
)
from evaluation import backtest_strategy
from features import build_meta_features
from features import main as build_and_save_features
from mlp_modeling import KerasSoftmaxWrapper, mlp_nested_cv
from modeling import (
    calibrate_model,
    split_train_test,
    train_meta_model,
)
from notifications import send_notification
from analysis import shap_explain, feature_importance, evaluate_model

from signals import filter_signals_with_meta_model
from sizing import compute_probability_weighted_returns
from strategy import compute_momentum, get_daily_signals
from labeling import scan_tp_sl_grid, plot_tp_sl_distribution


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
tf.get_logger().setLevel("ERROR")
lightgb_logger = logging.getLogger("LightGBM")
lightgb_logger.setLevel(logging.ERROR)  # Suppress LightGBM warnings


def main():
    """
    Main function to execute the meta-labeling alpha filter pipeline.
    """
    start = time()
    logging.info("=== 1. Load raw data ===")
    prices = load_prices()
    monthly_prices = load_monthly_prices()
    returns = load_returns()
    spy_returns = load_spy_returns()
    volatility = returns.rolling(20).std()

    logging.info("=== 2. Generate base strategy signals ===")
    daily_signals_lo = get_daily_signals(prices, monthly_prices, long_only=True)
    mom_returns_lo = compute_momentum(prices, daily_signals_lo, long_only=True)

    daily_signals_ls = get_daily_signals(prices, monthly_prices, long_only=False)
    mom_returns_ls = compute_momentum(prices, daily_signals_ls, long_only=False)

    logging.info("=== 3. Build and save features ===")
    scan_tp_sl_grid(
        prices, daily_signals_ls, volatility, tp_range=(2, 6), sl_range=(2, 6)
    )
    plot_tp_sl_distribution()
    build_and_save_features()

    logging.info("=== 4. Load and prepare meta-modeling data ===")
    X = load_features()
    Y = load_labels().squeeze()

    X_fold1, X_fold2, X_fold3, Y_fold1, Y_fold2, Y_fold3 = split_train_test(X, Y)

    logging.info("=== 5. Classifier Training and calibration ===")
    clf = train_meta_model(X_fold1, Y_fold1, "Bayesian")
    # clf = joblib.load(config.CLF_PATH)

    clf_cal = calibrate_model(clf, X_fold1, Y_fold1)
    # clf_cal = joblib.load(config.CLF_CAL_PATH)

    logging.info("=== 6. MLP training ===")
    (
        mlp_v1t,
        scaler,
        best_fold_hp_v1t,
        acc_mlp_v1t,
        auc_mlp_v1t,
        ll_mlp_v1t,
        hparams_v1t,
    ) = mlp_nested_cv(X_fold1, Y_fold1, "Bayesian", "mlpv1t")
    # mlp_v1t = load_model(config.MLPV1T)

    mlp_v1 = KerasSoftmaxWrapper(mlp_v1t, label_map=config.LABEL_MAP, scaler=scaler)

    joblib.dump(mlp_v1, config.MLPV1)
    # mlp_v1 = joblib.load(config.MLPV1)

    logging.info("=== 7. Generate Meta-features ===")

    daily_signals = daily_signals_lo if config.LONG_ONLY else daily_signals_ls
    X_meta_f2 = build_meta_features(X_fold2, scaler, daily_signals, clf_cal, mlp_v1)

    X_meta_f3 = build_meta_features(X_fold3, scaler, daily_signals, clf_cal, mlp_v1)

    logging.info("=== 8. Stacking Ensemble ===")

    (
        mlp_v2t,
        scaler_meta,
        best_fold_hp_v2t,
        acc_mlp_v2t,
        auc_mlp_v2t,
        ll_mlp_v2t,
        hparams_v2t,
    ) = mlp_nested_cv(X_meta_f2, Y_fold2, "Bayesian", "mlpv2t")
    # mlp_v2t = load_model(config.MLPV2T)

    mlp_v2 = KerasSoftmaxWrapper(
        mlp_v2t, label_map=config.LABEL_MAP, scaler=scaler_meta
    )

    joblib.dump(mlp_v2, config.MLPV2)
    # mlp_v2 = joblib.load(config.MLPV2)

    logging.info("=== 9. Classifier Analysis ===")
    shap_explain(model=clf, X_test=X_fold2, name="CLF")
    feature_importance(model=clf, shap_values=None, X_test=X_fold2, name="CLF")
    evaluate_model(clf_cal, X_fold2, Y_fold2, "CLF")

    logging.info("=== 10. MLP Analysis ===")
    X_fold2_scaled = scaler.transform(X_fold2)
    X_fold3_scaled = scaler.transform(X_fold3)

    shap_values_v1 = shap_explain(model=mlp_v1t, X_test=X_fold2_scaled, X_train=X_fold3_scaled, name="MLPV1T")
    feature_importance(model=mlp_v1t, shap_values=shap_values_v1, X_test=X_fold2_scaled, name="MLPV1")
    evaluate_model(mlp_v1, X_fold2, Y_fold2, "MLPV1")

    logging.info("=== 11. Meta MLP Analysis ===")
    X_meta_f2_scaled = scaler_meta.transform(X_meta_f2)
    X_meta_f3_scaled = scaler_meta.transform(X_meta_f3)

    shap_values_v2 = shap_explain(model=mlp_v2t, X_test=X_meta_f3_scaled, X_train=X_meta_f2_scaled, name="MLPV2T")
    feature_importance(model=mlp_v2t, shap_values=shap_values_v2, X_test=X_meta_f3_scaled, name="MLPV2")
    evaluate_model(mlp_v2, X_meta_f3, Y_fold3, "MLPV2")

    logging.info("=== 12. Meta-filtered signal generation ===")
    filtered_signals = filter_signals_with_meta_model(
        daily_signals=daily_signals,
        clf=mlp_v2,
        X_test=X_meta_f3,
        threshold=config.META_PROBA_THRESHOLD,
        min_gap=config.MIN_GAP,
    )
    logging.info("=== 13. Probability Weighting / Vol Targeting ===")
    (
        filtered_mom_returns,
        filtered_mom_returns_costs,
        weights_mom,
        mom_turnover,
        costs,
    ) = compute_probability_weighted_returns(
        clf=mlp_v2,
        filtered_signals=filtered_signals,
        X_test=X_meta_f3,
        returns=returns.loc[config.FOLD3_START :],
        logic=config.LOGIC,
        prob_weighting=config.PROB_WEIGHTING,
        target_vol=config.TARGET_VOL,
        leverage_cap=config.LEVERAGE_CAP,
    )

    logging.info("=== 14. Backtest meta-filtered strategy ===")
    summary = backtest_strategy(
        strategy_returns=filtered_mom_returns,
        strategy_returns_w_costs=filtered_mom_returns_costs,
        turnover=mom_turnover,
        bench_spy=spy_returns.loc[config.FOLD3_START :],
        bench_mom=mom_returns_lo.loc[config.FOLD3_START :],
        bench_mom_ls=mom_returns_ls.loc[config.FOLD3_START :],
        filtered_signals=filtered_signals,
        Y=Y_fold3,
        weights_df=weights_mom,
        name="Stacked MLP Meta-Labeling",
        plot=True,
        save=True,
    )

    logging.info(f"=== Performance Summary === \n {summary[summary.columns[0]]}")

    end = time()

    send_notification(
        message=f"ML training complete! \nLabeling factors: {config.PT_SL_FACTOR} \n",
        topic=NTFY_SERVER,
        duration_seconds=end - start,
        title="Training Job Done",
    )


if __name__ == "__main__":
    main()
