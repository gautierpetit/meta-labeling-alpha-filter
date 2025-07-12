import shap
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator

import config


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
    plt.savefig(config.FIGURES_DIR / "shap_beeswarm.png")


def save_shap_values(shap_values: shap.Explanation, X_test: pd.DataFrame) -> None:
    """
    Save SHAP values to a Parquet file for later analysis.

    Args:
        shap_values (shap.Explanation): SHAP values to save.
        X_test (pd.DataFrame): Feature matrix (used to label columns and index).

    Returns:
        None
    """
    df = pd.DataFrame(
        shap_values.values, columns=X_test.columns, index=X_test.index
    )
    df.to_parquet(config.SHAP_VALUES_PARQUET)