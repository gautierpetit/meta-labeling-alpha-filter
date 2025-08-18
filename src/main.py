"""
Main script for running the meta-labeling alpha filter pipeline.

This script orchestrates the entire workflow, including data loading, feature engineering, model training, evaluation, and backtesting.

Modules:
- Data loading
- Feature engineering
- Model training and evaluation
- Strategy backtesting

Author: Gautier Petit
Date: August 2, 2025
"""

import json
import logging
import sys
import warnings
from time import time

import joblib
import numpy as np
import tensorflow as tf

import src.config as config
from src.analysis import (
    append_ablation_row,
    evaluate_model,
    feature_importance,
    meta_vs_base_diagnostics,
    shap_explain,
)
from src.config_private import NTFY_SERVER
from src.data_loader import (
    load_features,
    load_labels,
    load_monthly_prices,
    load_prices,
    load_returns,
    load_spy_returns,
)
from src.evaluation import backtest_strategy
from src.features import build_meta_features, build_meta_features_lean
from src.features import main as build_and_save_features
from src.labeling import scan_holding_period_range, scan_tp_sl_grid
from src.mlp_modeling import (
    Bundle,
    ClasswiseConvexBlender,
    ConvexProbabilityBlender,
    MetaLogit,
    RollingVectorScaledSoftmax,
    VectorScaledSoftmax,
    _safe_transform,
    mlp_nested_cv,
)
from src.modeling import (
    RollingVectorScaledSoftmaxLGBM,
    VectorScaledSoftmaxLGBM,
    split_train_test,
    train_model,
)
from src.notifications import send_notification
from src.signals import filter_signals_with_meta_model
from src.sizing import compute_probability_weighted_returns
from src.strategy import compute_momentum, get_daily_signals
from src.utils import (
    class_priors,
    index_fingerprint,
    make_run_id,
    md5_columns,
    mirror_tree,
    parse_args,
    safe_git_sha,
    setup_json_logging,
    write_json,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
tf.get_logger().setLevel("ERROR")
lightgb_logger = logging.getLogger("LightGBM")
lightgb_logger.setLevel(logging.ERROR)


def main() -> None:
    """
    Main function to execute the meta-labeling alpha filter pipeline.
    """
    args = parse_args()
    run_id = make_run_id(args.run_tag)
    git_sha = safe_git_sha()

    run_dir = config.RUNS_DIR / run_id
    (fig_dir := run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (res_dir := run_dir / "results").mkdir(parents=True, exist_ok=True)
    (mod_dir := run_dir / "models").mkdir(parents=True, exist_ok=True)

    logger = setup_json_logging(run_dir, run_id=run_id, git_sha=git_sha)
    logger.info("Pipeline started [RUN] %s", run_id)
    cfg = {k: getattr(config, k) for k in dir(config) if k.isupper() and not k.startswith("_")}
    (run_dir / "config_snapshot.json").write_text(json.dumps(cfg, indent=2))

    # Redirect global output dirs to this run capsule (no function signatures change)
    config.FIGURES_DIR = fig_dir
    config.RESULTS_DIR = res_dir
    config.MODELS_DIR = mod_dir
    config.SHAP_VALUES_DIR = run_dir / "shap"
    config.SHAP_VALUES_DIR.mkdir(parents=True, exist_ok=True)

    # keep sub-model dirs coherent under this run
    config.MLPV1_DIR = config.MODELS_DIR / "mlpv1"
    config.MLPV2_DIR = config.MODELS_DIR / "mlpv2"
    config.MLPV1_DIR.mkdir(parents=True, exist_ok=True)
    config.MLPV2_DIR.mkdir(parents=True, exist_ok=True)
    config.CLF_DIR = config.MODELS_DIR / "clf"
    config.CLF_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=== 1. Load raw data ===")
    prices = load_prices()
    monthly_prices = load_monthly_prices()
    returns = load_returns()
    spy_returns = load_spy_returns()
    volatility = returns.rolling(63).std()

    logger.info("=== 2. Generate base strategy signals ===")
    daily_signals_lo = get_daily_signals(prices, monthly_prices, long_only=True)
    mom_returns_lo = compute_momentum(prices, daily_signals_lo)

    daily_signals_ls = get_daily_signals(prices, monthly_prices, long_only=False)
    mom_returns_ls = compute_momentum(prices, daily_signals_ls)

    logger.info("=== 3. Build and save features ===")
    scan_tp_sl_grid(prices, daily_signals_ls, volatility, tp_range=(2, 8), sl_range=(2, 8))

    scan_holding_period_range(
        prices,
        daily_signals_ls,
        volatility,
        tp_sl_factor=(5, 5),
        holding_period_range=(21, 126),
    )

    build_and_save_features()

    logger.info("=== 4. Load and prepare meta-modeling data ===")
    X = load_features()
    Y = load_labels().squeeze()

    X_fold1, X_fold2, X_fold3, Y_fold1, Y_fold2, Y_fold3 = split_train_test(X, Y)

    fingerprints = {
        "fold1": index_fingerprint(X_fold1.index),
        "fold2": index_fingerprint(X_fold2.index),
        "fold3": index_fingerprint(X_fold3.index),
    }
    with open(run_dir / "fold_fingerprints.json", "w", encoding="utf-8") as f:
        json.dump(fingerprints, f, indent=2)

    manifest = {
        "run_id": run_id,
        "git_sha": git_sha,
        "python": sys.version.split()[0],
        "tensorflow": tf.__version__,
        "random_state": config.RANDOM_STATE,
        "label_spec": {
            "pt_sl_factor": tuple(config.PT_SL_FACTOR),
            "max_holding_period": config.MAX_HOLDING_PERIOD,
            "label_map": config.LABEL_MAP,
        },
        "fold_dates": {
            "fold1": (config.FOLD1_START, config.FOLD1_END),
            "fold2": (config.FOLD2_START, config.FOLD2_END),
            "fold3": (config.FOLD3_START, config.FOLD3_END),
        },
        "features": {
            "count": X.shape[1],
            "md5_columns": md5_columns(X),
            "ordered_columns": list(map(str, X.columns)),
        },
    }
    write_json(run_dir / "manifest.json", manifest)

    logger.info("=== 5. Base Models Training ===")
    BUNDLE_CLASS_LABELS = np.array([-1, 0, 1], dtype=int)

    clf = train_model(X_fold1, Y_fold1, "Bayesian")
    # Load model
    clf = joblib.load(config.CLF_DIR / "clf.pkl")

    (
        mlp_v1t,
        scaler,
        best_fold_hp_v1t,
        acc_mlp_v1t,
        auc_mlp_v1t,
        ll_mlp_v1t,
        hparams_v1t,
    ) = mlp_nested_cv(X_fold1, Y_fold1, "Bayesian", "mlpv1t")

    # Save model:
    Bundle(mlp_v1t, scaler, BUNDLE_CLASS_LABELS).save(config.MLPV1_DIR)

    # Load model:
    bundle_v1 = Bundle.load(config.MLPV1_DIR, compile=False)
    mlp_v1t, scaler = bundle_v1.model, bundle_v1.scaler

    logger.info("=== 6. Models calibration ===")
    y_fold2_int = Y_fold2.map(config.LABEL_MAP).values

    # 1) Rolling OOS calibrator for Fold-2 (to build X_meta_f2 leakage-free)
    clf_oosF2 = RollingVectorScaledSoftmaxLGBM.from_fold2(
        clf,
        config.LABEL_MAP,
        X_fold2,
        y_fold2_int,
        n_splits=3,
        embargo=5,
        reg=1e-3,
        lr=0.05,
    )

    mlp_v1_oosF2 = RollingVectorScaledSoftmax.from_fold2(
        mlp_v1t,
        config.LABEL_MAP,
        scaler,
        X_fold2,
        y_fold2_int,
        n_splits=3,
        embargo=5,
        reg=1e-3,
        lr=0.05,
        max_iter=1000,
        tol=1e-7,
        patience=20,
    )

    # 2) A single final calibrator for Fold-3 (most-recent scaling)
    clf_F3 = VectorScaledSoftmaxLGBM.from_validation(
        clf, config.LABEL_MAP, X_fold2, y_fold2_int, reg=1e-3, lr=0.05
    )

    mlp_v1_F3 = VectorScaledSoftmax.from_validation(
        mlp_v1t,
        config.LABEL_MAP,
        scaler,
        X_fold2,
        y_fold2_int,
        reg=1e-3,
        lr=0.05,
        max_iter=1000,
        tol=1e-7,
        patience=20,
    )

    logger.info("=== 7. Generate Meta-features ===")

    daily_signals = daily_signals_lo if config.LONG_ONLY else daily_signals_ls
    # Fold-2 (meta-train): fully OOS-calibrated features
    X_meta_f2 = build_meta_features(X_fold2, clf_oosF2, mlp_v1_oosF2)
    meta_cols = X_meta_f2.columns.tolist()

    # Fold-3 (meta-test): calibrated on Fold-2
    X_meta_f3 = build_meta_features(X_fold3, clf_F3, mlp_v1_F3)

    X_meta_f2_lean = build_meta_features_lean(X_fold2, clf_oosF2, mlp_v1_oosF2)
    X_meta_f3_lean = build_meta_features_lean(X_fold3, clf_F3, mlp_v1_F3)

    logger.info("=== 8. Meta Model Training ===")

    (
        mlp_v2t,
        scaler_meta,
        best_fold_hp_v2t,
        acc_mlp_v2t,
        auc_mlp_v2t,
        ll_mlp_v2t,
        hparams_v2t,
    ) = mlp_nested_cv(X_meta_f2, Y_fold2, "Bayesian", "mlpv2t")

    meta_cols = X_meta_f2.columns.tolist()
    X_meta_f3 = X_meta_f3.reindex(columns=meta_cols)
    assert list(X_meta_f3.columns) == meta_cols
    assert list(scaler_meta.feature_names_in_) == meta_cols

    # Save model:
    Bundle(mlp_v2t, scaler_meta, BUNDLE_CLASS_LABELS).save(config.MLPV2_DIR)

    # Load model:
    bundle_v2 = Bundle.load(config.MLPV2_DIR, compile=False)
    mlp_v2t, scaler_meta = bundle_v2.model, bundle_v2.scaler

    logger.info("=== EXPERIMENTAL STACKER TESTS ===")

    ##### MLPV2 on full features #####
    mlp_v2_F3 = VectorScaledSoftmax.from_validation(
        mlp_v2t,  # trained on Fold-2
        config.LABEL_MAP,
        scaler_meta,
        X_meta_f2,  # calibrate on Fold-2 features
        y_fold2_int,  # Fold-2 labels ONLY
        reg=2e-2,
        lr=0.05,
        max_iter=1000,
        tol=1e-7,
        patience=20,
    )

    X_meta_f2_scaled = _safe_transform(scaler_meta, X_meta_f2)
    X_meta_f3_scaled = _safe_transform(scaler_meta, X_meta_f3)

    shap_values_v2 = shap_explain(
        model=mlp_v2t, X_test=X_meta_f3_scaled, X_train=X_meta_f2_scaled, name="MLPV2T"
    )
    feature_importance(
        model=mlp_v2t, shap_values=shap_values_v2, X_test=X_meta_f3_scaled, name="MLPV2"
    )
    evaluate_model(mlp_v2_F3, X_meta_f3, Y_fold3, "MLPV2_FULL")

    ##### MLPV2 on lean features #####
    (
        mlp_v2t_lean,
        scaler_meta_lean,
        best_fold_hp_v2t_lean,
        _,
        _,
        _,
        _,
    ) = mlp_nested_cv(X_meta_f2_lean, Y_fold2, "Bayesian", "mlpv2t")

    mlp_v2_F3_lean = VectorScaledSoftmax.from_validation(
        mlp_v2t_lean,
        config.LABEL_MAP,
        scaler_meta_lean,
        X_meta_f2_lean,
        y_fold2_int,
        reg=2e-2,
        lr=0.05,
        max_iter=1000,
        tol=1e-7,
        patience=20,
    )

    X_meta_f2_scaled_lean = _safe_transform(scaler_meta_lean, X_meta_f2_lean)
    X_meta_f3_scaled_lean = _safe_transform(scaler_meta_lean, X_meta_f3_lean)

    shap_values_v2 = shap_explain(
        model=mlp_v2t, X_test=X_meta_f3_scaled_lean, X_train=X_meta_f2_scaled_lean, name="MLPV2T"
    )
    feature_importance(
        model=mlp_v2t, shap_values=shap_values_v2, X_test=X_meta_f3_scaled_lean, name="MLPV2"
    )

    evaluate_model(mlp_v2_F3_lean, X_meta_f3_lean, Y_fold3, "MLPV2_LEAN")

    ##### Simple blender #####
    blender = ConvexProbabilityBlender.from_fold2(config.LABEL_MAP, X_meta_f2_lean, y_fold2_int)
    logger.info("Convex blender weight w=%.3f", blender.w)

    evaluate_model(blender, X_meta_f3_lean, Y_fold3, "BLENDER_LEAN")

    ##### Linear regression #####
    meta_logit = MetaLogit(config.LABEL_MAP, C=0.5).fit(X_meta_f2_lean, Y_fold2)

    meta_logit_cal = VectorScaledSoftmax.from_validation(
        model=meta_logit,
        label_map=config.LABEL_MAP,
        scaler=None,  # <-- not needed now
        X_val=X_meta_f2_lean,
        y_val_int=y_fold2_int,
        reg=1e-4,
        lr=0.05,
        max_iter=1000,
        tol=1e-7,
        patience=20,
    )

    evaluate_model(meta_logit_cal, X_meta_f3_lean, Y_fold3, "META_LOGIT_CAL")

    ##### Class-wise blender #####
    blender_fitter = ClasswiseConvexBlender(clf_oosF2, mlp_v1_oosF2, config.LABEL_MAP)
    blender_fitter.fit(
        X_fold2, y_fold2_int, lr=0.05, reg=1e-2, max_iter=2000, patience=50, tol=1e-7
    )

    # Save the trained weights from Fold-2 fit
    blender_fitter.save(run_dir / "models" / "BLENDER_CW")

    # Later / reload path
    blender_fitter = ClasswiseConvexBlender.load(
        run_dir / "models" / "BLENDER_CW", clf_F3, mlp_v1_F3, config.LABEL_MAP
    )
    # Create an inference copy that uses the Fold-3 calibrators
    blender_F3 = blender_fitter.with_inference_models(clf_F3, mlp_v1_F3)

    evaluate_model(blender_F3, X_fold3, Y_fold3, "BLENDER_CW")

    ##### END OF EXPERIMENTAL STACKER TESTS #####

    # Class priors by fold (int labels)
    y1 = Y_fold1.map(config.LABEL_MAP).values
    y2 = Y_fold2.map(config.LABEL_MAP).values
    y3 = Y_fold3.map(config.LABEL_MAP).values

    manifest.update(
        {
            "class_priors": {
                "fold1": class_priors(y1, 3),
                "fold2": class_priors(y2, 3),
                "fold3": class_priors(y3, 3),
            },
            "best_hparams": {
                "clf": clf.get_params(),
                "mlpv1": best_fold_hp_v1t.values,
            },
            "calibration": {
                "clf_oosF2": {
                    "type": "RollingVectorScaledSoftmaxLGBM",
                    "reg": 1e-3,
                    "splits": 3,
                    "embargo": 5,
                },
                "mlp_v1_oosF2": {
                    "type": "RollingVectorScaledSoftmax",
                    "reg": 1e-3,
                    "splits": 3,
                    "embargo": 5,
                },
                "blender_F3": {
                    "type": "ClasswiseConvexBlender",
                    "reg": 1e-2,
                },
                "clf_F3": {"type": "VectorScaledSoftmaxLGBM", "reg": 1e-3},
                "mlp_v1_F3": {"type": "VectorScaledSoftmax", "reg": 1e-3},
            },
        }
    )
    write_json(run_dir / "manifest.json", manifest)

    logger.info("=== 9. Classifier Analysis ===")
    shap_explain(model=clf, X_test=X_fold2, name="CLF")
    feature_importance(model=clf, shap_values=None, X_test=X_fold2, name="CLF")
    evaluate_model(clf_F3, X_fold3, Y_fold3, "CLF")

    logger.info("=== 10. MLP Analysis ===")
    X_fold1_scaled = _safe_transform(scaler, X_fold1)
    X_fold2_scaled = _safe_transform(scaler, X_fold2)

    shap_values_v1 = shap_explain(
        model=mlp_v1t, X_test=X_fold2_scaled, X_train=X_fold1_scaled, name="MLPV1T"
    )
    feature_importance(
        model=mlp_v1t, shap_values=shap_values_v1, X_test=X_fold2_scaled, name="MLPV1"
    )
    evaluate_model(mlp_v1_F3, X_fold3, Y_fold3, "MLPV1")

    logger.info("=== 11. Meta Stacker Analysis ===")
    evaluate_model(blender_F3, X_fold3, Y_fold3, "BLENDER_CW")

    logger.info("=== 12. Meta VS Base ===")

    proba_clf_F3 = clf_F3.predict_proba(X_fold3)
    proba_mlp_F3 = mlp_v1_F3.predict_proba(X_fold3)
    proba_blender_F3 = blender_F3.predict_proba(X_fold3)

    meta_vs_base_diagnostics(
        y_true=Y_fold3,
        proba_clf=proba_clf_F3,
        proba_mlp=proba_mlp_F3,
        proba_meta=proba_blender_F3,
        label_map=config.LABEL_MAP,
        outdir=fig_dir,
        prefix="fold3",
    )

    ablation_csv = res_dir / "ablation_fold3.csv"
    append_ablation_row(ablation_csv, "CLF_cal", Y_fold3, proba_clf_F3, config.LABEL_MAP)
    append_ablation_row(ablation_csv, "MLPv1_cal", Y_fold3, proba_mlp_F3, config.LABEL_MAP)
    append_ablation_row(ablation_csv, "Meta_Blend", Y_fold3, proba_blender_F3, config.LABEL_MAP)

    logger.info("=== 13. Meta-filtered signal generation ===")
    filtered_signals = filter_signals_with_meta_model(
        daily_signals=daily_signals.loc[config.FOLD3_START :],
        clf=blender_F3,
        X_test=X_fold3,
        min_gap=config.MIN_GAP,
    )
    logger.info("=== 14. Probability Weighting / Vol Targeting ===")
    (
        filtered_mom_returns,
        filtered_mom_returns_costs,
        weights_mom,
        mom_turnover,
        costs,
    ) = compute_probability_weighted_returns(
        clf=blender_F3,
        filtered_signals=filtered_signals,
        X_test=X_fold3,
        returns=returns.loc[config.FOLD3_START :],
        prob_weighting=config.PROB_WEIGHTING,
        target_vol=config.TARGET_VOL,
        leverage_cap=config.LEVERAGE_CAP,
    )

    logger.info("=== 15. Backtest meta-filtered strategy ===")
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
        name="Meta-Labeling",
        start=config.FOLD3_START,
        plot=True,
        save=True,
    )

    logger.info("=== Performance Summary === \n%s", summary[summary.columns[0]])

    if args.mirror_latest:
        mirror_tree(config.FIGURES_DIR, config.ROOT_DIR / "figures")
        mirror_tree(config.RESULTS_DIR, config.ROOT_DIR / "results")
        mirror_tree(config.SHAP_VALUES_DIR, config.ROOT_DIR / "shap")
        mirror_tree(config.MODELS_DIR, config.ROOT_DIR / "models")


if __name__ == "__main__":
    start = time()
    try:
        main()
        end = time()
        send_notification(
            message=f"ML training complete! \nLabeling factors: {config.PT_SL_FACTOR} \n",
            topic=NTFY_SERVER,
            duration_seconds=end - start,
            title="Pipeline successfully completed",
        )
        logging.getLogger(__name__).info("Pipeline completed in %.2f seconds.", end - start)
    except Exception as e:
        end = time()
        logging.getLogger(__name__).error("Error occurred during pipeline: %s", e)
        send_notification(
            message=f"A critical error stopped the pipeline \nError: {e}\nCheck the logs for more details.",
            topic=NTFY_SERVER,
            duration_seconds=end - start,
            title="Critical Error Occurred",
        )
