import logging
from time import time

import joblib
import pandas as pd
import numpy as np

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
from features import main as build_and_save_features
from modeling import (
    calibrate_model,
    evaluate_model,
    scale_features,
    split_train_test,
    train_meta_model,
)
from notifications import send_notification
from shap_analysis import explain_model, plot_shap_beeswarm, save_shap_values,explain_mlp
from signals import filter_signals_with_meta_model
from sizing import compute_probability_weighted_returns
from strategy import compute_momentum, get_daily_signals,get_daily_signals_long_short, compute_momentum_long_short
from mlp_modeling import mlp_nested_cv,KerasCalibrationCV
from tensorflow.keras.models import load_model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    start = time()

    logging.info("=== 1. Load raw data ===")
    prices = load_prices()
    monthly_prices = load_monthly_prices()
    returns = load_returns()
    spy_returns = load_spy_returns()

    logging.info("=== 2. Generate base strategy signals ===")
    daily_signals_lo, signal_dates = get_daily_signals(prices, monthly_prices)
    mom_returns_lo = compute_momentum(prices, daily_signals_lo)
    
    daily_signals_ls, signal_dates_ls = get_daily_signals_long_short(prices, monthly_prices)
    mom_returns_ls = compute_momentum_long_short(prices, daily_signals_ls)
    

    logging.info("=== 3. Build and save features ===")
    build_and_save_features()

    logging.info("=== 4. Load and prepare meta-modeling data ===")
    X = load_features()
    Y = load_labels().squeeze()
    # X_scaled = scale_features(X)
    X_train, Y_train, X_test, Y_test = split_train_test(X, Y)

    X_train_scaled = scale_features(X_train)
    X_test_scaled = scale_features(X_test)

    logging.info("=== 5. Classifier Training and calibration ===")
    clf = train_meta_model(X_train, Y_train, "Bayesian")
    # clf = joblib.load(config.BEST_MODEL_PATH)

    calibrated_clf = calibrate_model(clf, X_train, Y_train)
    # calibrated_clf = joblib.load(config.BEST_CAL_PATH)


    logging.info("=== 6. Classifier Analysis ===")
    # CLF Feature explanation
    shap_values = explain_model(
        model=clf, X_test=X_test, type="clf"
    )

    # CLF fit analysis
    acc, auc, logloss = evaluate_model(calibrated_clf, X_test, Y_test, "LightGBM")
    logging.info(f"Test Accuracy: {acc:.4f}")
    logging.info(f"Test ROC AUC: {auc:.4f}")
    logging.info(f"Test Log loss: {logloss:.4f}")

      
    logging.info("=== MPL training and calibration ===")
    mlp_nested, best_fold_hp, acc_mlp,auc_mlp,ll_mlp, hparams = mlp_nested_cv(X_train_scaled, Y_train, "Bayesian" )
    # mlp_nested = load_model(config.BEST_MLP)

    mlp_calibrated = KerasCalibrationCV(mlp_nested,method="sigmoid")
    mlp_calibrated.fit(X_train_scaled,Y_train)
    joblib.dump(mlp_calibrated,config.MLP_CAL)
    # mlp_calibrated = joblib.load(config.MLP_CAL)

    logging.info("=== MLP Analysis ===")
    # MLP Feature explanation
    shap_values_nn = explain_model(
        mlp_nested, X_test_scaled, "mlp",X_train_scaled
    )

    # MLP fit analysis
    acc_mlp, auc_mlp, ll_mlp = evaluate_model(mlp_calibrated, X_test_scaled, Y_test, "MLP")
    logging.info(f"Test Accuracy: {acc_mlp:.4f}")
    logging.info(f"Test ROC AUC: {auc_mlp:.4f}")
    logging.info(f"Test Log loss: {ll_mlp:.4f}")
 
    logging.info("=== Stacking Ensemble ===")











    logging.info("=== 7. Meta-filtered signal generation ===")
    filtered_signals = filter_signals_with_meta_model(
        daily_signals=daily_signals_ls,
        clf=mlp_calibrated,
        X_test=X_test_scaled,
        threshold=config.META_PROBA_THRESHOLD,
    )

    (
        filtered_mom_returns,
        filtered_mom_returns_costs,
        weights_mom,
        mom_turnover,
    ) = compute_probability_weighted_returns(
        clf=mlp_calibrated,
        X_test=X_test_scaled,
        returns=returns,
        threshold=config.META_PROBA_THRESHOLD,
        tc=config.TRANSACTION_COSTS,
        target_vol=0.3,  # 0.3
        vol_span=20,
        normalize=False,
        max_leverage=4,
        long_only=False
    )

    logging.info("=== 8. Backtest meta-filtered strategy ===")
    summary = backtest_strategy(
        strategy_returns=filtered_mom_returns[config.TEST_START_DATE:],
        strategy_returns_w_costs=filtered_mom_returns_costs[config.TEST_START_DATE:],
        turnover=mom_turnover[config.TEST_START_DATE:],
        bench_spy=spy_returns[config.TEST_START_DATE:],
        bench_mom=mom_returns_lo[config.TEST_START_DATE:],
        filtered_signals=filtered_signals[config.TEST_START_DATE:],
        Y=Y,
        weights_df=weights_mom[config.TEST_START_DATE:],
        name="Meta-Filtered Momentum",
        plot=True,
        save=True,
    )

    logging.info(f"=== Performance Summary === \n {summary}")

    end = time()

    send_notification(
        message="ML training complete! \n"
        f"Labeling factors: {config.PT_SL_FACTOR} \n"
        f"Accuracy: {acc:.2%} \n"
        f"ROC AUC: {auc:.2%} \n"
        f"Test Log loss: {logloss:.4f}",
        topic=NTFY_SERVER,
        duration_seconds=end - start,
        title="Training Job Done",
    )


if __name__ == "__main__":
    main()
