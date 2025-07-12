import logging

import pandas as pd

import config
from data_loader import (load_features, load_labels, load_monthly_prices,
                         load_prices, load_returns, load_spy_returns)
from features import main as build_and_save_features
from signals import filter_signals_with_meta_model
from evaluation import backtest_strategy
from modeling import (calibrate_model, evaluate_model, scale_features,
                      split_train_test, train_meta_model)
from shap_analysis import explain_model, plot_shap_beeswarm, save_shap_values
from strategy import compute_momentum, get_daily_signals

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
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
    X_scaled = scale_features(X)
    X_train, Y_train, X_test, Y_test = split_train_test(X_scaled, Y)

    logging.info("=== 5. Train and calibrate model ===")
    clf = train_meta_model(X_train, Y_train)
    calibrated_clf = calibrate_model(clf, X_train, Y_train)

    acc, auc = evaluate_model(calibrated_clf, X_test, Y_test)
    logging.info(f"Test Accuracy: {acc:.4f}")
    logging.info(f"Test ROC AUC: {auc:.4f}")

    logging.info("=== 6. SHAP analysis ===")
    shap_values = explain_model(clf, X_test) # Shap cannot explain calibrated classifier
    plot_shap_beeswarm(shap_values)
    save_shap_values(shap_values, X_test)

    logging.info("=== 7. Meta-filtered signal generation ===")
    filtered_signals = filter_signals_with_meta_model(
        daily_signals=daily_signals,
        clf=calibrated_clf,
        X_test=X_test,
        threshold=config.META_PROBA_THRESHOLD,
    )

    filtered_returns = returns * filtered_signals
    n_positions = filtered_signals.sum(axis=1).replace(0, pd.NA)
    filtered_mom_returns = filtered_returns.sum(axis=1) / n_positions #EW position sizing

    logging.info("=== 8. Backtest meta-filtered strategy ===")
    filtered_idx = filtered_signals.stack()[filtered_signals.stack() == 1].index
    filtered_outcomes = Y.loc[Y.index.isin(filtered_idx)]
    trade_count = len(filtered_outcomes)
    win_rate = filtered_outcomes.mean()

    summary_meta, summary_spy, summary_mom = backtest_strategy(
        strategy_returns=filtered_mom_returns,
        bench_spy=spy_returns,
        bench_mom=mom_returns,
        name="Meta-Filtered Momentum",
        trade_count=trade_count,
        win_rate=win_rate,
        plot=True,
    )

    summary_meta["Trade Count"] = trade_count
    summary_meta["Win Rate"] = f"{win_rate:.2%}"

    summary = pd.concat([summary_mom, summary_spy, summary_meta], axis=1)
    summary.columns = ["Standard Momentum", "SPY", "Meta-Filtered Momentum"]
    summary.to_excel(config.PERFORMANCE_SUMMARY_XLSX)
    logging.info(f"Backtest summary saved to: {config.PERFORMANCE_SUMMARY_XLSX}")


if __name__ == "__main__":
    main()
