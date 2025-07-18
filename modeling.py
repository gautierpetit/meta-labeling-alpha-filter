import logging
from typing import Literal

import joblib
import matplotlib.pyplot as plt
import numpy as np
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
from data_loader import load_features, load_labels

logger = logging.getLogger(__name__)


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
    X_train = X.loc[: config.TRAIN_END_DATE]
    Y_train = Y.loc[: config.TRAIN_END_DATE]
    X_test = X.loc[config.TEST_START_DATE :]
    Y_test = Y.loc[config.TEST_START_DATE :]
    return X_train, Y_train, X_test, Y_test


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
    lgbm = LGBMClassifier(
        device_type="gpu",
        gpu_platform_id=0,
        gpu_device_id=0,
        objective="multiclass",
        random_state=config.RANDOM_STATE,
        # class_weight="balanced",
        n_jobs=config.N_JOBS,
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

    search.fit(
        X,
        Y,
    )

    best_model = search.best_estimator_
    logging.info(f"Best estimator: {best_model}")

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
    calibrated_clf = CalibratedClassifierCV(
        estimator=clf,
        method="sigmoid",
        cv=TimeSeriesSplit(n_splits=config.CV_N_SPLITS, gap=config.MAX_HOLDING_PERIOD),
    )
    calibrated_clf.fit(X, Y)

    joblib.dump(calibrated_clf, config.BEST_CAL_PATH)
    return calibrated_clf


def evaluate_model(
    clf: BaseEstimator, X_test: pd.DataFrame, Y_test: pd.Series
) -> tuple[float, float, float]:
    """
    Evaluate multiclass classifier on test data.

    Args:
        clf (BaseEstimator): Trained classifier.
        X_test (pd.DataFrame): Test feature matrix.
        Y_test (pd.Series): True labels.

    Returns:
        Tuple: (Accuracy, ROC AUC [macro], Log Loss)
    """

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    acc = accuracy_score(Y_test, y_pred)
    auc = roc_auc_score(Y_test, y_proba, multi_class="ovr", average="macro")
    logloss = log_loss(Y_test, y_proba)

    cm = confusion_matrix(Y_test, y_pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot()
    plt.savefig(config.FIGURES_DIR / "confusion_matrix.png")
    plt.close()

    logging.info("=== Classification Report ===")
    logging.info(classification_report(Y_test, y_pred))

    for class_idx, class_label in enumerate(clf.classes_):
        prob_true, prob_pred = calibration_curve(
            (Y_test == class_label).astype(int),
            y_proba[:, class_idx],
            n_bins=10,
            strategy="quantile",
        )

        plt.figure(figsize=(6, 6))
        plt.plot(prob_pred, prob_true, marker="o", label=f"Class {class_idx}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.title(f"Calibration Curve – Class {class_idx}")
        plt.xlabel("Predicted Probability")
        plt.ylabel("True Proportion")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(config.FIGURES_DIR / f"calibration_curve_class_{class_idx}.png")
        plt.close()

        print(
            f"Label {class_label} | Mean predicted prob: {y_proba[:, class_idx].mean():.4f}"
        )
        print(f"Actual proportion in Y_test: {(Y_test == class_label).mean():.4f}")

    return acc, auc, logloss


def main():
    logging.info("=== 4. Load and prepare meta-modeling data ===")
    X = load_features()
    Y = load_labels().squeeze()
    X_train, Y_train, X_test, Y_test = split_train_test(X, Y)
    logging.info("=== 5. Train and calibrate model ===")
    clf = train_meta_model(X_train, Y_train, "Bayesian")
    calibrated_clf = calibrate_model(clf, X_train, Y_train)
    acc, auc, logloss = evaluate_model(calibrated_clf, X_test, Y_test)
    logging.info(f"Test Accuracy: {acc:.4f}")
    logging.info(f"Test ROC AUC: {auc:.4f}")
    logging.info(f"Test Log loss: {logloss:.4f}")


if __name__ == "__main__":
    main()
