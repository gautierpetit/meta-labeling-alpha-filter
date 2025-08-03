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
    evaluate_model,
    scale_features,
    split_train_test,
    train_meta_model,
)
from notifications import send_notification
from shap_analysis import explain_model
from signals import filter_signals_with_meta_model
from sizing import compute_probability_weighted_returns
from strategy import compute_momentum, get_daily_signals

warnings.simplefilter(action="ignore", category=FutureWarning)

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

    logging.info("=== 2. Generate base strategy signals ===")
    daily_signals_lo, signal_dates_lo = get_daily_signals(
        prices, monthly_prices, long_only=True
    )
    mom_returns_lo = compute_momentum(prices, daily_signals_lo, long_only=True)

    daily_signals_ls, signal_dates_ls = get_daily_signals(
        prices, monthly_prices, long_only=False
    )
    mom_returns_ls = compute_momentum(prices, daily_signals_ls, long_only=False)

    logging.info("=== 3. Build and save features ===")
    build_and_save_features()

    logging.info("=== 4. Load and prepare meta-modeling data ===")
    X = load_features()
    Y = load_labels().squeeze()

    X_fold1, X_fold2, X_fold3, Y_fold1, Y_fold2, Y_fold3 = split_train_test(X, Y)

    X_fold1_scaled = scale_features(X_fold1)
    X_fold2_scaled = scale_features(X_fold2)

    logging.info("=== 5. Classifier Training and calibration ===")
    clf = train_meta_model(X_fold1, Y_fold1, "Bayesian")
    # clf = joblib.load(config.CLF_PATH)

    clf_cal = calibrate_model(clf, X_fold1, Y_fold1)
    # clf_cal = joblib.load(config.CLF_CAL_PATH)

    logging.info("=== 6. Classifier Analysis ===")
    explain_model(model=clf, X_test=X_fold2, name="CLF")

    acc_clf, auc_clf, logloss_clf = evaluate_model(clf_cal, X_fold2, Y_fold2, "CLF")
    logging.info(f"Test Accuracy: {acc_clf:.4f}")
    logging.info(f"Test ROC AUC: {auc_clf:.4f}")
    logging.info(f"Test Log loss: {logloss_clf:.4f}")

    logging.info("=== 7. MLP training ===")
    mlp_v1t, best_fold_hp_v1t, acc_mlp_v1t, auc_mlp_v1t, ll_mlp_v1t, hparams_v1t = (
        mlp_nested_cv(X_fold1_scaled, Y_fold1, "Bayesian", "mlpv1t")
    )
    # mlp_v1t = load_model(config.MLPV1T)

    mlp_v1 = KerasSoftmaxWrapper(mlp_v1t, label_map=config.LABEL_MAP)

    joblib.dump(mlp_v1, config.MLPV1)
    # mlp_v1 = joblib.load(config.MLPV1)

    logging.info("=== 8. MLP Analysis ===")
    explain_model(mlp_v1t, X_fold2_scaled, name="MLPV1T", X_train=X_fold1_scaled)

    acc_mlp_v1, auc_mlp_v1, ll_mlp_v1 = evaluate_model(
        mlp_v1, X_fold2_scaled, Y_fold2, "MLPV1"
    )
    logging.info(f"Test Accuracy: {acc_mlp_v1:.4f}")
    logging.info(f"Test ROC AUC: {auc_mlp_v1:.4f}")
    logging.info(f"Test Log loss: {ll_mlp_v1:.4f}")

    logging.info("=== 9. Generate Meta-features ===")

    X_meta_f2_scaled = build_meta_features(X_fold2, clf_cal, mlp_v1, scale=True)

    X_meta_f3_scaled = build_meta_features(X_fold3, clf_cal, mlp_v1, scale=True)

    logging.info("=== 10. Stacking Ensemble ===")

    mlp_v2t, best_fold_hp_v2t, acc_mlp_v2t, auc_mlp_v2t, ll_mlp_v2t, hparams_v2t = (
        mlp_nested_cv(X_meta_f2_scaled, Y_fold2, "Bayesian", "mlpv2t")
    )
    # mlp_v2t = load_model(config.MLPV2T)

    mlp_v2 = KerasSoftmaxWrapper(mlp_v2t, label_map=config.LABEL_MAP)

    joblib.dump(mlp_v2, config.MLPV2)
    # mlp_v2 = joblib.load(config.MLPV2)

    logging.info("=== 11. Meta MLP Analysis ===")
    explain_model(mlp_v2t, X_meta_f3_scaled, name="MLPV2T", X_train=X_meta_f2_scaled)

    acc_mlp_v2, auc_mlp_v2, ll_mlp_v2 = evaluate_model(
        mlp_v2, X_meta_f3_scaled, Y_fold3, "MLPV2"
    )
    logging.info(f"Test Accuracy: {acc_mlp_v2:.4f}")
    logging.info(f"Test ROC AUC: {auc_mlp_v2:.4f}")
    logging.info(f"Test Log loss: {ll_mlp_v2:.4f}")

    logging.info("=== 12. Meta-filtered signal generation ===")
    filtered_signals = filter_signals_with_meta_model(
        daily_signals=daily_signals_lo if config.LONG_ONLY else daily_signals_ls,
        clf=mlp_v2,
        X_test=X_meta_f3_scaled,
        threshold=config.META_PROBA_THRESHOLD,
    )
    logging.info("=== 13. Probability Weighting / Vol Targeting ===")
    (
        filtered_mom_returns,
        filtered_mom_returns_costs,
        weights_mom,
        mom_turnover,
        costs
    ) = compute_probability_weighted_returns(
        clf=mlp_v2,
        X_test=X_meta_f3_scaled,
        returns=returns,
        threshold=config.META_PROBA_THRESHOLD,
        target_vol=config.TARGET_VOL,
        vol_span=config.VOL_SPAN,
        normalize=False,
        max_leverage=config.MAX_LEVERAGE,
    )

    logging.info("=== 14. Backtest meta-filtered strategy ===")
    summary = backtest_strategy(
        strategy_returns=filtered_mom_returns.loc[config.FOLD3_START:],
        strategy_returns_w_costs=filtered_mom_returns_costs.loc[config.FOLD3_START:],
        turnover=mom_turnover.loc[config.FOLD3_START:],
        bench_spy=spy_returns.loc[config.FOLD3_START:],
        bench_mom=mom_returns_lo.loc[config.FOLD3_START:],
        bench_mom_ls=mom_returns_ls.loc[config.FOLD3_START:],
        filtered_signals=filtered_signals.loc[config.FOLD3_START:],
        Y=Y_fold3.loc[config.FOLD3_START:],
        weights_df=weights_mom.loc[config.FOLD3_START:],
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
