import logging

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import BaseEstimator

import config
import shap


def explain_model(clf: BaseEstimator, X_test: pd.DataFrame) -> shap.Explanation:
    """
    Generate SHAP values for a given classifier and test set.

    Args:
        clf (BaseEstimator): Trained classifier.
        X_test (pd.DataFrame): Feature matrix for the test set.

    Returns:
        shap.Explanation: SHAP values explaining model predictions.
    """
    explainer = shap.Explainer(clf)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test)
    plt.savefig(config.SHAP_VALUES_DIR / "shap_beeswarm.png")
    plt.close()

    importance = pd.Series(
        data=clf.feature_importances_, index=X_test.columns
    ).sort_values(ascending=False)
    logging.info(f"Feature importance: {importance}")

    return shap_values


def plot_shap_beeswarm(shap_values: shap.Explanation) -> None:
    """
    Plot a SHAP beeswarm summary plot and save to file.

    Args:
        shap_values (shap.Explanation): SHAP values to visualize.

    Returns:
        None
    """
    shap.plots.beeswarm(shap_values, show=False)
    plt.savefig(config.SHAP_VALUES_DIR / "shap_beeswarm.png")
    plt.close()


def save_shap_values(shap_values: shap.Explanation, X_test: pd.DataFrame) -> None:
    """
    Save SHAP values for multiclass classification, one file per class.
    """
    for class_idx in range(shap_values.values.shape[2]):
        class_df = pd.DataFrame(
            shap_values.values[:, :, class_idx],
            columns=X_test.columns,
            index=X_test.index,
        )
        class_df.to_parquet(
            config.SHAP_VALUES_DIR / f"shap_values_class_{class_idx}.parquet"
        )
