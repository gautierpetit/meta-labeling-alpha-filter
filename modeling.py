import logging
from typing import Literal, Tuple

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV

import config
from config import (
    FOLD1_END,
    FOLD1_START,
    FOLD2_END,
    FOLD2_START,
    FOLD3_END,
    FOLD3_START,
)

logger = logging.getLogger(__name__)


def scale_features(X: pd.DataFrame, return_scaler=False) -> pd.DataFrame:
    """
    Standardize features using StandardScaler.

    Args:
        X (pd.DataFrame): Feature matrix to be scaled.

    Returns:
        pd.DataFrame: Scaled feature matrix with same index and columns.
        scaker (StandardScaler): Fitted scaler object if return_scaler is True.
    """
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
    return (X_scaled, scaler) if return_scaler else X_scaled


def split_train_test(
    X: pd.DataFrame, Y: pd.Series
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split feature matrix and labels into 3 folds using config-defined date ranges.

    Args:
        X (pd.DataFrame): Feature matrix.
        Y (pd.Series): Labels.

    Returns:
        Tuple: Feature and label splits for 3 folds.
    """
    X_fold1 = X.loc[FOLD1_START:FOLD1_END]
    Y_fold1 = Y.loc[FOLD1_START:FOLD1_END]

    X_fold2 = X.loc[FOLD2_START:FOLD2_END]
    Y_fold2 = Y.loc[FOLD2_START:FOLD2_END]

    X_fold3 = X.loc[FOLD3_START:FOLD3_END]
    Y_fold3 = Y.loc[FOLD3_START:FOLD3_END]

    return X_fold1, X_fold2, X_fold3, Y_fold1, Y_fold2, Y_fold3


def train_meta_model(
    X: pd.DataFrame, Y: pd.Series, estimation: Literal["Random", "Bayesian"] = "Random"
) -> BaseEstimator:
    """
    Train a LightGBM model using either Randomized Search or Bayesian SearchCV
    for hyperparameter tuning.

    Args:
        X (pd.DataFrame): Training features.
        Y (pd.Series): Training labels.
        estimation (str): Either "Random" for RandomizedSearchCV or
                          "Bayesian" for BayesSearchCV (default: "Random").

    Returns:
        BaseEstimator: The best estimator found by the search strategy.
    """
    logger.info(f"Training meta model with {estimation} search...")

    lgbm = LGBMClassifier(
        device_type="gpu",
        gpu_platform_id=0,
        gpu_device_id=0,
        objective="multiclass",
        random_state=config.RANDOM_STATE,
        n_jobs=config.N_JOBS,
        verbose=-1
    )

    cv = TimeSeriesSplit(n_splits=config.CV_N_SPLITS, gap=config.MAX_HOLDING_PERIOD)

    if estimation == "Random":
        search = RandomizedSearchCV(
            estimator=lgbm,
            param_distributions=config.HYPERPARAM_RANDOM,
            n_iter=config.RANDOM_SEARCH_ITER,
            cv=cv,
            scoring=config.CV_SCORING,
            n_jobs=config.N_JOBS,
            verbose=1,
            random_state=config.RANDOM_STATE,
        )
    elif estimation == "Bayesian":
        search = BayesSearchCV(
            estimator=lgbm,
            search_spaces=config.HYPERPARAM_BAYESIAN,
            n_iter=config.RANDOM_SEARCH_ITER,
            scoring=config.CV_SCORING,
            cv=cv,
            n_jobs=config.N_JOBS,
            verbose=1,
            random_state=config.RANDOM_STATE,
        )
    else:
        raise ValueError("Invalid estimation method. Choose 'Random' or 'Bayesian'.")

    search.fit(X, Y)

    best_model = search.best_estimator_
    logger.info(f"Best estimator: {best_model}")

    joblib.dump(best_model, config.CLF_PATH)

    return best_model


def calibrate_model(
    clf: BaseEstimator,
    X: pd.DataFrame,
    Y: pd.Series,
    method: Literal["sigmoid", "isotonic"] = "sigmoid",
) -> CalibratedClassifierCV:
    """
    Calibrate classifier probabilities using sigmoid calibration.

    Args:
        clf (BaseEstimator): Trained classifier.
        X (pd.DataFrame): Training features.
        Y (pd.Series): Training labels.

    Returns:
        CalibratedClassifierCV: Calibrated classifier.
    """
    calibrated_clf = CalibratedClassifierCV(
        estimator=clf,
        method=method,
        cv=TimeSeriesSplit(n_splits=config.CV_N_SPLITS, gap=config.MAX_HOLDING_PERIOD),
    )
    calibrated_clf.fit(X, Y)

    joblib.dump(calibrated_clf, config.CLF_CAL_PATH)
    return calibrated_clf


def evaluate_model(
    clf: BaseEstimator,
    X_test: pd.DataFrame,
    Y_test: pd.Series,
    name: str = "Test Model",
) -> tuple[float, float, float]:
    """
    Evaluate multiclass classifier on test data with label remapping for consistency.

    Args:
        clf (BaseEstimator): Trained classifier.
        X_test (pd.DataFrame): Test feature matrix.
        Y_test (pd.Series): True labels (with -1, 0, 1 format).

    Returns:
        Tuple: (Accuracy, ROC AUC [macro], Log Loss)
    """

    y_true = Y_test.map(config.LABEL_MAP).values

    # Predict
    y_pred = clf.predict(X_test)
    y_pred = (
        pd.Series(y_pred).map(config.LABEL_MAP).values
    )  # remap predictions if needed

    y_proba = clf.predict_proba(X_test)

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    logloss = log_loss(y_true, y_proba)

    # Use fixed label order [0, 1, 2] corresponding to [-1, 0, 1]
    fixed_labels = [0, 1, 2]
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=fixed_labels),
        display_labels=["-1", "0", "1"],
    )
    disp.plot()
    plt.savefig(config.FIGURES_DIR / f"confusion_matrix_{name}.png")
    plt.close()

    logger.info("=== Classification Report ===")
    logger.info(classification_report(y_true, y_pred, target_names=["-1", "0", "1"]))

    for class_idx, class_label in enumerate(fixed_labels):
        prob_true, prob_pred = calibration_curve(
            (y_true == class_label).astype(int),
            y_proba[:, class_idx],
            n_bins=10,
            strategy="quantile",
        )

        plt.figure(figsize=(6, 6))
        plt.plot(
            prob_pred,
            prob_true,
            marker="o",
            label=f"Class {class_label} ({['-1', '0', '1'][class_idx]})",
        )
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.title(f"Calibration Curve {name} – Class {['-1', '0', '1'][class_idx]}")
        plt.xlabel("Predicted Probability")
        plt.ylabel("True Proportion")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            config.FIGURES_DIR
            / f"calibration_curve_{name}_class_{['-1', '0', '1'][class_idx]}.png"
        )
        plt.close()

        print(
            f"Label {['-1', '0', '1'][class_idx]} | Mean predicted prob: {y_proba[:, class_idx].mean():.4f}"
        )
        print(f"Actual proportion in Y_test: {(y_true == class_label).mean():.4f}")

    return acc, auc, logloss
