"""
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

Utilities for model analysis, plots and reports.

This module provides convenience functions to generate SHAP explanations,
feature importance tables, evaluation metrics and diagnostic plots.

The functions favor robustness (safe sampling, path creation), explicit
typing and JSON-serializable outputs for downstream CI/storage.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import Model
from sklearn.base import BaseEstimator
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)

import shap
import src.config as config
from src.utils import _per_sample_nll, _to_3d_shap

logger = logging.getLogger(__name__)


def shap_explain(
    model: BaseEstimator | Model,
    X_test: pd.DataFrame,
    name: str = "model",
    X_train: pd.DataFrame | None = None,
) -> np.ndarray:
    """Compute SHAP values and write summary plots and per-class values.

    The function handles both tree-based sklearn estimators and deep Keras
    models. For deep models a small background dataset (`X_train`) is
    required.

    Args:
        model: Trained estimator (sklearn or Keras).
        X_test: Test set (DataFrame) used to compute explanations. A random
            sample is taken for speed.
        name: Identifier used when saving plots/values.
        X_train: Optional background dataset required for deep models.

    Returns:
        np.ndarray: SHAP values reshaped to (n_samples, n_features, n_classes).

    Raises:
        ValueError: If `X_train` is required but not provided.
    """

    n_samples = min(len(X_test), 1000)
    if n_samples == 0:
        raise ValueError("X_test must contain at least one row")

    X_explain = X_test.sample(n_samples, random_state=config.RANDOM_STATE)

    # Choose suitable explainer
    if hasattr(model, "classes_"):
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_explain.values)
    else:
        if X_train is None:
            raise ValueError("Background dataset required for deep models.")
        n_background = min(len(X_train), 100)
        background = X_train.sample(n_background, random_state=config.RANDOM_STATE)
        explainer = shap.DeepExplainer(model, background.values)
        sv = explainer.shap_values(X_explain.values, check_additivity=False)

    sv3 = _to_3d_shap(sv)

    # Persist summary plots and per-class values
    for class_idx in range(sv3.shape[2]):
        try:
            shap.summary_plot(sv3[:, :, class_idx], X_explain, show=False)
            plt.title(f"SHAP Summary Plot for {name} - Class {class_idx}")
            plt.gcf().tight_layout()
            output_path = (
                Path(config.SHAP_VALUES_DIR) / "summaries" / f"summary_{name}_class_{class_idx}.png"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, bbox_inches="tight")
            plt.close()
            logger.info("SHAP summary plot saved to: %s", output_path)
        except Exception:
            logger.exception("Failed to create SHAP summary plot for class %s", class_idx)

        class_df = pd.DataFrame(sv3[:, :, class_idx], columns=X_test.columns, index=X_explain.index)
        output_path2 = (
            Path(config.SHAP_VALUES_DIR) / "values" / f"values_{name}_class_{class_idx}.parquet"
        )
        output_path2.parent.mkdir(parents=True, exist_ok=True)
        class_df.to_parquet(output_path2)

    return sv3


def feature_importance(
    model: BaseEstimator | Model,
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
    name: str,
) -> pd.Series:
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
        raw_importance = pd.Series(data=model.feature_importances_, index=X_test.columns)
    else:
        if shap_values.ndim != 3:
            raise ValueError(
                "shap_values should be 3-dimensional (n_samples, n_features, n_classes)"
            )
        avg_across_classes = np.mean(np.abs(shap_values), axis=2)
        mean_abs_shap = np.mean(avg_across_classes, axis=0)
        raw_importance = pd.Series(mean_abs_shap, index=X_test.columns)

    importance = raw_importance / raw_importance.sum()
    importance = importance.sort_values(ascending=False) * 100
    logger.info("Feature importance for %s:\n%s", name, importance)

    # Save to CSV
    path = Path(config.RESULTS_DIR) / "feature_importance" / f"feature_importance_{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(path)
    logger.info("[%s] Feature importance saved to: %s", name, path)

    return importance


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

    acc = float(accuracy_score(y_true, y_pred))
    auc = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
    logloss = float(log_loss(y_true, y_proba))

    # Use fixed label order [0, 1, 2] corresponding to [-1, 0, 1]
    fixed_labels = [0, 1, 2]

    # Confusion matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=fixed_labels),
        display_labels=["-1", "0", "1"],
    )
    disp.plot()
    path = Path(config.FIGURES_DIR) / "confusion_matrices" / f"confusion_matrix_{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()

    # Save classification report to JSON
    report_dict = classification_report(
        y_true, y_pred, target_names=["-1", "0", "1"], output_dict=True, zero_division=0
    )
    report_dict["accuracy"] = acc
    report_dict["roc_auc"] = auc
    report_dict["log_loss"] = logloss

    # Calibration curve for each class
    for class_idx, class_label in enumerate(fixed_labels):
        prob_true, prob_pred = calibration_curve(
            (y_true == class_label).astype(int),
            y_proba[:, class_idx],
            n_bins=10,
            strategy="quantile",
        )

        ece = np.abs(prob_true - prob_pred).mean()  # ECE calculation
        brier = brier_score_loss((y_true == class_label).astype(int), y_proba[:, class_idx])

        # Add to report
        report_dict[f"class_{class_label}_ece"] = ece
        report_dict[f"class_{class_label}_brier"] = brier

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
        path = (
            Path(config.FIGURES_DIR)
            / "calibration_curves"
            / f"calibration_curve_{name}_class_{['-1', '0', '1'][class_idx]}.png"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path)
        plt.close()

        logger.info(
            "Label %s | Mean predicted prob: %.4f",
            ["-1", "0", "1"][class_idx],
            float(y_proba[:, class_idx].mean()),
        )
        logger.info(
            "Actual proportion in Y_test: %.4f",
            float((y_true == class_label).mean()),
        )

    json_path = (
        Path(config.RESULTS_DIR) / "classification_reports" / f"classification_report_{name}.json"
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure JSON-serializable (convert numpy scalars)
    def _json_fallback(o: Any) -> Any:
        if isinstance(o, np.floating | np.integer):
            return o.item()
        raise TypeError(f"Object of type {type(o)} not JSON serializable")

    with open(json_path, "w") as f:
        json.dump(report_dict, f, indent=4, default=_json_fallback)

    logger.info("Saved classification report to: %s", json_path)
    logger.info("=== Classification Report ===")
    logger.info(report_dict)
    logger.info(f"Accuracy: {acc:.4f}, AUC: {auc:.4f}, Log Loss: {logloss:.4f}")
    return acc, auc, logloss


def plot_learning_curve(history: Any, name: str, save: bool = True) -> None:
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
        path.parent.mkdir(parents=True, exist_ok=True)  # Ensure folder exists
        plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def save_history(history: Any, name: str) -> None:
    if history is None or not hasattr(history, "history") or not history.history:
        return
    df = pd.DataFrame(history.history)
    path = Path(config.FIGURES_DIR) / "learning_curves" / f"learning_curve_{name}.xlsx"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=False)


def meta_vs_base_diagnostics(
    y_true: pd.Series | np.ndarray,
    proba_clf: np.ndarray,
    proba_mlp: np.ndarray,
    proba_meta: np.ndarray,
    label_map: dict,
    outdir: str | Path,
    prefix: str = "fold3",
) -> None:
    # integer labels & the index for class +1
    y_int = np.asarray([label_map[int(y)] for y in y_true])
    k_pos = label_map[1]

    # 1) Δ log-loss histogram (meta minus best base) – per-sample
    nll_clf = _per_sample_nll(y_int, proba_clf)
    nll_mlp = _per_sample_nll(y_int, proba_mlp)
    nll_meta = _per_sample_nll(y_int, proba_meta)
    delta = nll_meta - np.minimum(nll_clf, nll_mlp)

    plt.figure(figsize=(8, 5))
    plt.hist(delta, bins=60)
    plt.axvline(delta.mean(), ls="--")
    plt.title(
        f"Δ per-sample NLL (meta - best base) | mean={delta.mean():.4f} | improved={(delta < 0).mean():.1%}"
    )
    plt.xlabel("Δ NLL")
    plt.ylabel("count")
    plt.tight_layout()
    outdir = Path(outdir)
    path = outdir / "meta_vs_base" / f"{prefix}_delta_nll_hist.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()

    # 2) P(+1) scatter (MLPv1 vs Meta), colored by true class with legend
    plt.figure(figsize=(8, 5))
    x = proba_mlp[:, k_pos]
    y = proba_meta[:, k_pos]
    # Colors for mapped labels 0,1,2 which correspond to true labels -1,0,+1
    class_colors = ["tab:orange", "tab:gray", "tab:green"]
    class_labels = ["-1", "0", "+1"]

    # Plot one scatter per true-class so legend entries are created
    for idx, lbl in enumerate(class_labels):
        mask = y_int == idx
        if mask.any():
            plt.scatter(x[mask], y[mask], s=6, alpha=0.25, c=class_colors[idx], label=lbl)

    lim = (0, 1)
    plt.plot(lim, lim, "k--", lw=1)
    plt.xlim(lim)
    plt.ylim(lim)
    plt.xlabel("MLPv1  P(+1)")
    plt.ylabel("Meta  P(+1)")
    plt.title("P(+1): MLPv1 vs Meta (Fold-3)")
    plt.legend(title="True label", markerscale=2, fontsize="small")
    plt.tight_layout()
    path = outdir / "meta_vs_base" / f"{prefix}_p1_scatter_mlp_vs_meta.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()

    # 3) Decile lift on P(+1): empirical freq of class +1
    df = pd.DataFrame({"p1": proba_meta[:, k_pos], "y": y_int})
    df["decile"] = pd.qcut(df["p1"], 10, labels=False, duplicates="drop")
    lift = df.groupby("decile").apply(lambda g: (g["y"] == k_pos).mean())
    plt.figure(figsize=(8, 5))
    lift.plot(marker="o")
    plt.title("Empirical P(y=+1) by P(+1) decile (Meta, Fold-3)")
    plt.xlabel("decile (low→high P(+1))")
    plt.ylabel("empirical freq of +1")
    plt.tight_layout()
    path = outdir / "meta_vs_base" / f"{prefix}_decile_lift_meta.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def append_ablation_row(
    out_csv: str | Path, name: str, y_true: pd.Series, proba: np.ndarray, label_map: dict
) -> None:
    """Append a single-row summary (logloss, ROC AUC) to the ablation CSV.

    The CSV is created if it does not exist. The function is robust to both
    string and Path inputs for `out_csv`.
    """
    out_csv = Path(out_csv)
    y_int = np.asarray([label_map[int(y)] for y in y_true])
    y_pred = proba.argmax(axis=1)
    row = {
        "model": name,
        "log_loss": float(log_loss(y_int, proba, labels=[0, 1, 2])),
        "roc_auc_ovr": float(roc_auc_score(y_int, proba, multi_class="ovr")),
        "accuracy": float(accuracy_score(y_int, y_pred)),
    }

    df = pd.DataFrame([row])
    if out_csv.exists():
        df0 = pd.read_csv(out_csv)
        df = pd.concat([df0, df], ignore_index=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
