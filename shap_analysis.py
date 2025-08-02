import logging
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import Model
from sklearn.base import BaseEstimator

import config
import shap


def calculate_feature_importance(raw_importance: pd.Series) -> pd.Series:
    """
    Calculate normalized feature importance.

    Args:
        raw_importance (pd.Series): Raw feature importance values.

    Returns:
        pd.Series: Normalized and sorted feature importance values.
    """
    importance = raw_importance / raw_importance.sum()
    importance = importance.sort_values(ascending=False)
    return importance


def save_shap_values(shap_values: list, X_test: pd.DataFrame, name: str) -> None:
    """
    Save SHAP values for each class to parquet files.

    Args:
        shap_values (list): SHAP values for each class.
        X_test (pd.DataFrame): Test dataset.
        name (str): Name of the model for saving outputs.

    Returns:
        None
    """
    for class_idx in range(len(shap_values)):
        class_df = pd.DataFrame(
            shap_values[class_idx],
            columns=X_test.columns,
            index=X_test.index,
        )
        class_df.to_parquet(
            config.SHAP_VALUES_DIR / f"values_{name}_class_{class_idx}.parquet"
        )


def explain_model(
    model: Union[BaseEstimator, Model],
    X_test: pd.DataFrame,
    name: str = "model",
    X_train: Optional[pd.DataFrame] = None,
    class_plot: int = 2,
) -> Union[shap.Explanation, list]:
    """
    Generate SHAP explanations for a given model and test dataset.

    Args:
        model (Union[BaseEstimator, Model]): The model to explain (e.g., sklearn or Keras model).
        X_test (pd.DataFrame): Test dataset for which SHAP values are computed.
        name (str): Name of the model for saving outputs (default: "model").
        X_train (Optional[pd.DataFrame]): Training dataset for background data (required for deep models).
        class_plot (int): Class index for which to generate the SHAP summary plot (default: 2).

    Returns:
        None

    Raises:
        ValueError: If `X_train` is not provided for deep models.

    Example:
        shap_values = explain_model(model, X_test, name="my_model", X_train=X_train)
    """
    if hasattr(model, "classes_"):  # sklearn-like model
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test.values)

        # Feature importance if available
        if hasattr(model, "feature_importances_"):
            raw_importance = pd.Series(
                data=model.feature_importances_, index=X_test.columns
            )
            importance = calculate_feature_importance(raw_importance)
            logging.info(f"Feature importance: \n{importance * 100}")

    else:  # Deep learning model
        if X_train is None:
            raise ValueError("Background dataset required for deep models.")

        background = X_train.sample(100, random_state=config.RANDOM_STATE)
        explainer = shap.DeepExplainer(model, background.values)
        shap_values = explainer.shap_values(X_test.values)

        avg_across_classes = np.mean(np.abs(shap_values), axis=2)
        mean_abs_shap = np.mean(avg_across_classes, axis=0)
        raw_importance = pd.Series(mean_abs_shap, index=X_test.columns)
        importance = calculate_feature_importance(raw_importance)
        logging.info(f"Feature importance: \n{importance * 100}")

    # Save SHAP values for each class
    save_shap_values(shap_values, X_test, name)

    # Generate SHAP summary plot
    shap.summary_plot(shap_values[class_plot], X_test, show=False)
    plt.gcf().tight_layout()
    plt.savefig(config.SHAP_VALUES_DIR / f"summary_{name}.png", bbox_inches="tight")
    plt.close()
