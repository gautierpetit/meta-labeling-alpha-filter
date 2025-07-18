import logging
from time import time

import joblib
import pandas as pd

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
from shap_analysis import explain_model, plot_shap_beeswarm, save_shap_values
from signals import filter_signals_with_meta_model
from sizing import compute_probability_weighted_returns
from strategy import compute_momentum, get_daily_signals

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
    daily_signals, signal_dates = get_daily_signals(prices, monthly_prices)
    mom_returns = compute_momentum(prices, daily_signals)

    logging.info("=== 3. Build and save features ===")
    build_and_save_features()

    logging.info("=== 4. Load and prepare meta-modeling data ===")
    X = load_features()
    Y = load_labels().squeeze()
    # X_scaled = scale_features(X)
    X_train, Y_train, X_test, Y_test = split_train_test(X, Y)

    logging.info("=== 5. Train and calibrate model ===")
    clf = train_meta_model(X_train, Y_train, "Bayesian")

    # clf = joblib.load(config.BEST_MODEL_PATH)

    calibrated_clf = calibrate_model(clf, X_train, Y_train)

    # calibrated_clf = joblib.load(config.BEST_CAL_PATH)

    acc, auc, logloss = evaluate_model(calibrated_clf, X_test, Y_test)
    logging.info(f"Test Accuracy: {acc:.4f}")
    logging.info(f"Test ROC AUC: {auc:.4f}")
    logging.info(f"Test Log loss: {logloss:.4f}")

    logging.info("=== 6. SHAP analysis ===")
    shap_values = explain_model(
        clf, X_test
    )  # Shap cannot explain calibrated classifier
    save_shap_values(shap_values, X_test)

    logging.info("=== 7. Meta-filtered signal generation ===")
    filtered_signals = filter_signals_with_meta_model(
        daily_signals=daily_signals,
        clf=calibrated_clf,
        X_test=X_test,
        threshold=config.META_PROBA_THRESHOLD,
    )

    (
        filtered_mom_returns,
        filtered_mom_returns_costs,
        weights_mom,
        proba_mom,
        mom_turnover,
    ) = compute_probability_weighted_returns(
        clf=calibrated_clf,
        X_test=X_test,
        returns=returns,
        threshold=config.META_PROBA_THRESHOLD,
        tc=config.TRANSACTION_COSTS,
        target_vol=0.3,  # 0.3
        vol_span=20,
        normalize=False,
    )

    logging.info("=== 8. Backtest meta-filtered strategy ===")
    summary = backtest_strategy(
        strategy_returns=filtered_mom_returns,
        strategy_returns_w_costs=filtered_mom_returns_costs,
        turnover=mom_turnover,
        bench_spy=spy_returns,
        bench_mom=mom_returns,
        filtered_signals=filtered_signals,
        Y=Y,
        weights_df=weights_mom,
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
