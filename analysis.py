import json
import logging
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import Model
from sklearn.base import BaseEstimator
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)

import config
import shap

logger = logging.getLogger(__name__)


def shap_explain(
    model: Union[BaseEstimator, Model],
    X_test: pd.DataFrame,
    name: str = "model",
    X_train: Optional[pd.DataFrame] = None,
) -> None:
    """
    Generate SHAP explanations for a given model and test dataset.

    Args:
        model (Union[BaseEstimator, Model]): The model to explain (e.g., sklearn or Keras model).
        X_test (pd.DataFrame): Test dataset for which SHAP values are computed.
        name (str): Name of the model for saving outputs (default: "model").
        X_train (Optional[pd.DataFrame]): Training dataset for background data (required for deep models).

    Returns:
        None

    Raises:
        ValueError: If `X_train` is not provided for deep models.
    """
    X_explain = X_test.sample(1000, random_state=config.RANDOM_STATE)
    
    if hasattr(model, "classes_"):  # sklearn-like model
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_explain.values)
    else:  # Deep learning model
        if X_train is None:
            raise ValueError("Background dataset required for deep models.")
        background = X_train.sample(100, random_state=config.RANDOM_STATE)
        explainer = shap.DeepExplainer(model, background.values)
        shap_values = explainer.shap_values(X_explain.values, check_additivity=False)

    # Save SHAP values and generate summary plots
    for class_idx in range(shap_values.shape[2]):
        # Shap summary plot
        shap.summary_plot(
            shap_values[:, :, class_idx],
            X_explain,
            show=False,
            class_names=[f"Class {class_idx}"],
        )
        plt.title(f"SHAP Summary Plot for {name} - Class {class_idx}")
        plt.gcf().tight_layout()
        output_path = config.SHAP_VALUES_DIR / "summaries" / f"summary_{name}_class_{class_idx}.png"
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        logger.info(f"SHAP summary plot saved to: {output_path}")

        # Save SHAP values to parquet
        class_df = pd.DataFrame(
            shap_values[:, :, class_idx],
            columns=X_test.columns,
            index=X_explain.index,
        )
        output_path = (
            config.SHAP_VALUES_DIR / "values" / f"values_{name}_class_{class_idx}.parquet"
        )
        class_df.to_parquet(output_path)
        logger.info(f"SHAP values saved to: {output_path}")

    return shap_values


def feature_importance(
    model: Union[BaseEstimator, Model],
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
    name: str,
) -> None:
    """
    Calculate and save feature importance based on SHAP values or model attributes.

    Args:
        model (Union[BaseEstimator, Model]): Trained model.
        shap_values (ndarray): SHAP values of shape (n_samples, n_features, n_classes).
        X_test (pd.DataFrame): Test dataset.
        name (str): Name of the model for saving outputs.

    Returns:
        None
    """
    if hasattr(model, "feature_importances_"):
        raw_importance = pd.Series(
            data=model.feature_importances_, index=X_test.columns
        )
    else:
        avg_across_classes = np.mean(np.abs(shap_values), axis=2)
        mean_abs_shap = np.mean(avg_across_classes, axis=0)
        raw_importance = pd.Series(mean_abs_shap, index=X_test.columns)

    importance = raw_importance / raw_importance.sum()
    importance = importance.sort_values(ascending=False)
    logger.info(f"Feature importance: \n{importance * 100}")

    # Save to Excel
    path = config.RESULTS_DIR / "feature_importance" / f"feature_importance_{name}.xlsx"
    importance.to_excel(path, header=["importance"])
    logger.info(f"[{name}] Feature importance saved to: {path}")




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
        name (str): Name of the model for saving outputs (default: "Test Model").

    Returns:
        tuple[float, float, float]: Accuracy, ROC AUC (macro), and Log Loss.
    """
    y_true = Y_test.map(config.LABEL_MAP).values

    # Predict
    y_pred = clf.predict(X_test)
    y_pred = pd.Series(y_pred).map(config.LABEL_MAP).values
    y_proba = clf.predict_proba(X_test)

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    logloss = log_loss(y_true, y_proba)

    # Use fixed label order [0, 1, 2] corresponding to [-1, 0, 1]
    fixed_labels = [0, 1, 2]

    # Confusion matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=fixed_labels),
        display_labels=["-1", "0", "1"],
    )
    disp.plot()
    plt.savefig(config.FIGURES_DIR / "confusion_matrices" / f"confusion_matrix_{name}.png")
    plt.close()

    # Save classification report to JSON
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=["-1", "0", "1"],
        output_dict=True,
        zero_division=0,
    )
    report_dict["accuracy"] = acc
    report_dict["roc_auc"] = auc
    report_dict["log_loss"] = logloss
    json_path = config.RESULTS_DIR / "classification_reports" / f"classification_report_{name}.json"
    with open(json_path, "w") as f:
        json.dump(report_dict, f, indent=4)

    logger.info(f"Saved classification report to: {json_path}")
    logger.info("=== Classification Report ===")
    logger.info(report_dict)
    logger.info(f"Accuracy: {acc:.4f}, AUC: {auc:.4f}, Log Loss: {logloss:.4f}")

    # Calibration curve for each class
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
        path = config.FIGURES_DIR / "calibration_curves" / f"calibration_curve_{name}_class_{['-1', '0', '1'][class_idx]}.png"
        plt.savefig(path)
        plt.close()

        logger.info(
            f"Label {['-1', '0', '1'][class_idx]} | Mean predicted prob: {y_proba[:, class_idx].mean():.4f}"
        )
        logger.info(
            f"Actual proportion in Y_test: {(y_true == class_label).mean():.4f}"
        )



def plot_learning_curve(history, name: str, save: bool = True):
    """
    Plot training/validation loss over epochs from a Keras History.
    """
    if history is None or not hasattr(history, "history") or not history.history:
        return

    hist = pd.DataFrame(history.history)
    plt.figure(figsize=(8, 5))
    plt.plot(hist["loss"], label="loss")
    if "val_loss" in hist:
        plt.plot(hist["val_loss"], label="val_loss")
    plt.title(f"Learning Curve – {name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    if save:
        path = config.FIGURES_DIR / "learning_curves" / f"learning_curve_{name}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

def save_history(history, name: str):
    if history is None or not hasattr(history, "history") or not history.history:
        return
    df = pd.DataFrame(history.history)
    path = config.FIGURES_DIR / "learning_curves" / f"learning_curve_{name}.xlsx"
    df.to_excel(path, index=False)
