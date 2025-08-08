
import logging
from typing import Literal, Tuple

import joblib

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
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
        verbose=-1,
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


