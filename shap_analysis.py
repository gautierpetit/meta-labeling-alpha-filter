import logging
from keras import Model
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import BaseEstimator
import numpy as np
import tensorflow as tf
from typing import Literal, Union, Optional

import config
import shap

def explain_model(
    model: Union[BaseEstimator, Model],
    X_test: pd.DataFrame,
    type: Literal["clf", "mlp"],
    X_train: Optional[pd.DataFrame] = None,
    class_plot: int = 2,
) -> Union[shap.Explanation, list]:

    if type == "clf":
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test.values)
        
        # Feature importance if available
        
        raw_importance = pd.Series(
            data=model.feature_importances_, index=X_test.columns
        )
        importance = raw_importance / raw_importance.sum()
        importance = importance.sort_values(ascending=False)
        logging.info(f"Feature importance: \n{importance*100}")
        
    
    elif type == "mlp":
        if X_train is None:
            raise ValueError("Background data set required for mlp")

        background = X_train.sample(100, random_state=config.RANDOM_STATE)
        explainer = shap.DeepExplainer(model, background.values)
        shap_values = explainer.shap_values(X_test.values)

        avg_across_classes = np.mean(np.abs(shap_values), axis=2)
        mean_abs_shap = np.mean(avg_across_classes, axis=0)
        importance = pd.Series(mean_abs_shap, index=X_test.columns).sort_values(ascending=False)
        logging.info(f"Feature Importance: \n{importance*100}")

        # SHAP summary beeswarm for one class
    
    else:
        raise ValueError("Wrong type. Choose 'clf' or 'mlp'")
        
    for class_idx in range(shap_values.shape[2]):
            class_df = pd.DataFrame(
                shap_values[:, :, class_idx],
                columns=X_test.columns,
                index=X_test.index,
            )
            class_df.to_parquet(
                config.SHAP_VALUES_DIR / f"values_{type}_class_{class_idx}.parquet"
            )   

    

    shap.summary_plot(shap_values[:, :, class_plot], X_test, show=False)
    plt.gcf().tight_layout()  # Optional: adjust layout
    plt.savefig(config.SHAP_VALUES_DIR / f"summary_{type}.png", bbox_inches="tight")
    plt.close()


    return shap_values




