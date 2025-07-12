import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

import config


def scale_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize features using StandardScaler.

    Args:
        X (pd.DataFrame): Feature matrix to be scaled.

    Returns:
        pd.DataFrame: Scaled feature matrix with same index and columns.
    """
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
    return X_scaled


def split_train_test(
    X: pd.DataFrame, Y: pd.Series
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split feature matrix and labels into train and test sets using config-defined dates.

    Args:
        X (pd.DataFrame): Feature matrix.
        Y (pd.Series): Labels.

    Returns:
        Tuple: (X_train, Y_train, X_test, Y_test)
    """
    X_train = X.loc[:config.TRAIN_END_DATE]
    Y_train = Y.loc[:config.TRAIN_END_DATE]
    X_test = X.loc[config.TEST_START_DATE:]
    Y_test = Y.loc[config.TEST_START_DATE:]
    return X_train, Y_train, X_test, Y_test


def train_meta_model(X: pd.DataFrame, Y: pd.Series) -> BaseEstimator:
    """
    Train a LightGBM model using randomized search over hyperparameters.

    Args:
        X (pd.DataFrame): Training features.
        Y (pd.Series): Training labels.

    Returns:
        BaseEstimator: Best estimator found by randomized search.
    """
    lgbm = LGBMClassifier(
        device_type="gpu",
        gpu_platform_id=0,
        gpu_device_id=0,
        random_state=config.RANDOM_STATE,
        class_weight="balanced",
        n_jobs=config.N_JOBS,
    )

    search = RandomizedSearchCV(
        estimator=lgbm,
        param_distributions=config.HYPERPARAM_SPACE,
        n_iter=config.RANDOM_SEARCH_ITER,
        cv=TimeSeriesSplit(n_splits=config.CV_N_SPLITS),
        scoring=config.CV_SCORING,
        n_jobs=config.N_JOBS,
        verbose=2,
        random_state=config.RANDOM_STATE,
    )

    search.fit(X, Y)
    best_model = search.best_estimator_
    joblib.dump(best_model, config.BEST_MODEL_PATH)
    return best_model


def calibrate_model(
    clf: BaseEstimator, X: pd.DataFrame, Y: pd.Series
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
    calibrated_clf = CalibratedClassifierCV(estimator=clf, method="isotonic", cv=TimeSeriesSplit(n_splits=config.CV_N_SPLITS))
    calibrated_clf.fit(X, Y)
    return calibrated_clf


def evaluate_model(
    clf: BaseEstimator, X_test: pd.DataFrame, Y_test: pd.Series
) -> tuple[float, float]:
    """
    Evaluate classifier on test data and plot calibration curve.

    Args:
        clf (BaseEstimator): Trained (and optionally calibrated) classifier.
        X_test (pd.DataFrame): Test feature matrix.
        Y_test (pd.Series): True labels.

    Returns:
        Tuple: (Accuracy, ROC AUC score)
    """
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(Y_test, y_pred)
    auc = roc_auc_score(Y_test, y_proba)

    # --- Calibration curve plot ---
    prob_true, prob_pred = calibration_curve(Y_test, y_proba, n_bins=20, strategy="uniform")

    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.title("Calibration Curve (Reliability Diagram)")
    plt.xlabel("Predicted Probability")
    plt.ylabel("True Proportion")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "calibration_curve.png")
    plt.close()

    return acc, auc

