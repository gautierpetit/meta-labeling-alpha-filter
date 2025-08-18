"""Model helpers: feature scaling and LightGBM training utilities.

This module contains helpers used across the
project for scaling features, splitting folds and training LightGBM
classifiers. The classes at the bottom provide lightweight wrappers
for post-hoc vector-scaling calibration for LightGBM models.
"""

import logging
from collections.abc import Sequence
from typing import Any, Literal

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from skopt import BayesSearchCV

import src.config as config
from src.config import FOLD1_END, FOLD1_START, FOLD2_END, FOLD2_START, FOLD3_END, FOLD3_START
from src.utils import _rolling_windows

logger = logging.getLogger(__name__)


def scale_features(
    X: pd.DataFrame, return_scaler: bool = False
) -> pd.DataFrame | tuple[pd.DataFrame, StandardScaler]:
    """
    Standardize features using StandardScaler.

    Args:
        X (pd.DataFrame): Feature matrix to be scaled.

    Returns:
        pd.DataFrame: Scaled feature matrix with same index and columns.
        scaler (StandardScaler): Fitted scaler object if return_scaler is True.
    """

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
    return (X_scaled, scaler) if return_scaler else X_scaled


def split_train_test(
    X: pd.DataFrame, Y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
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


def train_model(
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
    y_raw = np.asarray(Y).ravel()
    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)  # maps -1, 0, 1 → 0, 1, 2

    # Get class weights in encoded space
    w_arr = compute_class_weight(class_weight="balanced", classes=np.unique(y_enc), y=y_enc)

    # Map weights back to original labels {-1, 0, 1}
    class_weight = {cls: w for cls, w in zip(le.classes_, w_arr, strict=False)}

    lgbm = LGBMClassifier(
        device_type="gpu",
        gpu_platform_id=0,
        gpu_device_id=0,
        objective="multiclass",
        class_weight=class_weight,
        random_state=config.RANDOM_STATE,
        n_jobs=config.N_JOBS,
        verbose=-1,
    )

    cv = TimeSeriesSplit(n_splits=config.CV_N_SPLITS, gap=config.CV_GAP)

    if estimation == "Random":
        search = RandomizedSearchCV(
            estimator=lgbm,
            param_distributions=config.HYPERPARAM_RANDOM,
            n_iter=config.RANDOM_SEARCH_ITER,
            cv=cv,
            scoring="neg_log_loss",
            n_jobs=config.N_JOBS,
            verbose=1,
            random_state=config.RANDOM_STATE,
        )
    elif estimation == "Bayesian":
        search = BayesSearchCV(
            estimator=lgbm,
            search_spaces=config.HYPERPARAM_BAYESIAN,
            n_iter=config.RANDOM_SEARCH_ITER,
            scoring="neg_log_loss",
            cv=cv,
            n_jobs=config.N_JOBS,
            verbose=1,
            random_state=config.RANDOM_STATE,
        )
    else:
        raise ValueError("Invalid estimation method. Choose 'Random' or 'Bayesian'.")

    search.fit(X, Y)

    best_params = search.best_params_
    logger.info(f"Best estimator: {search.best_estimator_}")
    logger.info(f"Best params: {best_params}")

    clf_final = LGBMClassifier(
        device_type="gpu",
        gpu_platform_id=0,
        gpu_device_id=0,
        objective="multiclass",
        max_depth=-1,
        class_weight=class_weight,
        random_state=config.RANDOM_STATE,
        n_jobs=config.N_JOBS,
        **best_params,
    )

    cut = int(0.8 * len(X))
    Xtr, Xva = X.iloc[:cut], X.iloc[cut:]
    ytr, yva = y_raw[:cut], y_raw[cut:]

    clf_final.fit(
        Xtr,
        ytr,
        eval_set=[(Xva, yva)],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)],
    )

    joblib.dump(clf_final, config.CLF_DIR / "clf.pkl")
    return clf_final


class VectorScaledSoftmaxLGBM:
    """Post-hoc vector-scaling calibration for LightGBM multiclass models.

    The wrapper accepts a trained LightGBM `LGBMClassifier` and applies per-class
    affine rescaling of logits z' = a * z + b followed by a softmax. It exposes
    `predict_proba` and `predict` compatible with sklearn-like estimators.
    """

    def __init__(
        self,
        lgbm_model: LGBMClassifier,
        class_labels: Sequence[Any],
        a: np.ndarray,
        b: np.ndarray,
    ) -> None:
        """Initialize the calibrated wrapper.

        Args:
            lgbm_model: A fitted LightGBM `LGBMClassifier` instance.
            class_labels: Iterable of original class labels ordered to match
                the columns of the LightGBM raw scores (logits).
            a: Per-class scaling coefficients (shape (K,)).
            b: Per-class bias terms (shape (K,)).
        """
        # store provided objects; do not mutate inputs
        self.model = lgbm_model
        self.class_labels_ = list(class_labels)
        self.a = np.asarray(a, dtype=np.float64)
        self.b = np.asarray(b, dtype=np.float64)
        self.K = len(self.class_labels_)

    @staticmethod
    def _softmax_np(z: np.ndarray) -> np.ndarray:
        """Numerically-stable softmax over rows of z."""
        z = z - np.max(z, axis=1, keepdims=True)
        ez = np.exp(z)
        return ez / np.sum(ez, axis=1, keepdims=True)

    @staticmethod
    def _fit_vs(
        logits: np.ndarray,
        y_int: np.ndarray,
        K: int,
        reg: float = 1e-3,
        lr: float = 0.05,
        max_iter: int = 1000,
        tol: float = 1e-7,
        patience: int = 20,
    ) -> tuple[np.ndarray, np.ndarray]:
        # simple GD on (a,b) per class with L2 regularization
        a = np.ones(K, dtype=np.float64)
        b = np.zeros(K, dtype=np.float64)
        y_onehot = np.eye(K, dtype=np.float64)[y_int]
        best_loss, no_improve = np.inf, 0
        for _ in range(max_iter):
            z = logits * a[np.newaxis, :] + b[np.newaxis, :]
            p = VectorScaledSoftmaxLGBM._softmax_np(z)
            # cross-entropy + reg
            loss = -np.sum(y_onehot * np.log(p + 1e-12)) / len(y_int) + reg * (
                np.sum(a * a) + np.sum(b * b)
            )
            if loss + tol < best_loss:
                best_loss = loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
            # gradients
            g = (p - y_onehot) / len(y_int)
            grad_a = np.sum(g * logits, axis=0) + 2 * reg * a
            grad_b = np.sum(g, axis=0) + 2 * reg * b
            a -= lr * grad_a
            b -= lr * grad_b
        return a, b

    @classmethod
    def from_validation(
        cls,
        lgbm_model: LGBMClassifier,
        label_map: dict[Any, int],
        X_val: pd.DataFrame,
        y_val_int: np.ndarray,
        reg: float = 1e-3,
        lr: float = 0.05,
        max_iter: int = 1000,
        tol: float = 1e-7,
        patience: int = 20,
    ) -> "VectorScaledSoftmaxLGBM":
        logits = lgbm_model.predict(X_val, raw_score=True)
        # LightGBM returns shape (n, K) for multiclass raw scores
        class_labels = sorted(label_map.keys(), key=lambda k: label_map[k])
        a, b = cls._fit_vs(
            logits,
            y_val_int,
            K=len(class_labels),
            reg=reg,
            lr=lr,
            max_iter=max_iter,
            tol=tol,
            patience=patience,
        )
        return cls(lgbm_model, class_labels, a, b)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated class probabilities for rows in X.

        This method expects X to be the same feature matrix used to obtain the
        LightGBM raw scores (no additional scaling is applied here).
        """
        logits = self.model.predict(X, raw_score=True)
        z = logits * self.a[np.newaxis, :] + self.b[np.newaxis, :]
        return self._softmax_np(z)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return the predicted class labels for rows in X.

        The result is an ndarray of the original class labels (e.g. -1, 0, 1).
        """
        P = self.predict_proba(X)
        idx = np.argmax(P, axis=1)
        return np.array([self.class_labels_[i] for i in idx])


class RollingVectorScaledSoftmaxLGBM:
    """Rolling out-of-sample vector-scaling for LightGBM.

    This class builds per-segment (time window) vector-scaling parameters
    (a, b) computed on calibration slices and applied to prediction windows.
    """

    def __init__(
        self,
        lgbm_model: LGBMClassifier,
        class_labels: Sequence[Any],
        segments: Sequence[tuple[Any, Any, np.ndarray, np.ndarray]],
        default_ab: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Create a rolling calibration object.

        Args:
            lgbm_model: Fitted LightGBM classifier producing raw scores.
            class_labels: Ordered class labels matching logits columns.
            segments: Sequence of tuples (start_label, end_label, a, b)
                where `a` and `b` are ndarray per-class parameters.
            default_ab: Fallback (a, b) used for timestamps not covered by a
                segment.
        """
        self.model = lgbm_model
        self.class_labels_ = list(class_labels)
        self.K = len(self.class_labels_)
        self.segments = list(segments)
        self.default_a, self.default_b = default_ab

    @classmethod
    def from_fold2(
        cls,
        lgbm_model,
        label_map,
        X_fold2,
        y_fold2_int,
        n_splits=3,
        embargo=5,
        reg=1e-3,
        lr=0.05,
        max_iter=1000,
        tol=1e-7,
        patience=20,
    ):
        n = len(X_fold2)
        idx = X_fold2.index
        class_labels = sorted(label_map.keys(), key=lambda k: label_map[k])
        segments = []
        last_a = np.ones(len(class_labels))
        last_b = np.zeros(len(class_labels))

        for cal_end, pred_start, pred_end in _rolling_windows(n, n_splits, embargo):
            logits_cal = lgbm_model.predict(X_fold2.iloc[:cal_end], raw_score=True)
            a, b = VectorScaledSoftmaxLGBM._fit_vs(
                logits_cal,
                y_fold2_int[:cal_end],
                K=len(class_labels),
                reg=reg,
                lr=lr,
                max_iter=max_iter,
                tol=tol,
                patience=patience,
            )
            last_a, last_b = a.copy(), b.copy()
            start_label = idx[pred_start]
            end_label = idx[pred_end - 1]
            segments.append((start_label, end_label, last_a, last_b))
        return cls(lgbm_model, class_labels, segments, (last_a, last_b))

    @staticmethod
    def _softmax_np(z: np.ndarray) -> np.ndarray:
        z = z - np.max(z, axis=1, keepdims=True)
        ez = np.exp(z)
        return ez / np.sum(ez, axis=1, keepdims=True)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        logits = self.model.predict(X, raw_score=True)
        out = np.full((len(X), self.K), np.nan, dtype=np.float64)
        for start_label, end_label, a, b in self.segments:
            mask = (X.index >= start_label) & (X.index <= end_label)
            if not np.any(mask):
                continue
            z = logits[mask] * a[np.newaxis, :] + b[np.newaxis, :]
            out[mask] = self._softmax_np(z)
        missing = np.isnan(out).any(axis=1)
        if np.any(missing):
            z = logits[missing] * self.default_a[np.newaxis, :] + self.default_b[np.newaxis, :]
            out[missing] = self._softmax_np(z)
        return out

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted original class labels for rows in X."""
        P = self.predict_proba(X)
        idx = np.argmax(P, axis=1)
        return np.array([self.class_labels_[i] for i in idx])
