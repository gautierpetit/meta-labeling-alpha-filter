import logging
import os
import random
from typing import Literal

import keras_tuner as kt

import numpy as np
import pandas as pd
import tensorflow as tf
from keras_tuner import Objective
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    roc_auc_score,
)



from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from modeling import evaluate_model




import config
from data_loader import load_features, load_labels
from modeling import scale_features, split_train_test

logger = logging.getLogger(__name__)



# Set random seeds
random.seed(config.RANDOM_STATE)
np.random.seed(config.RANDOM_STATE)
tf.random.set_seed(config.RANDOM_STATE)

# Configure TensorFlow to use deterministic operations
os.environ["TF_DETERMINISTIC_OPS"] = "1"

# Early stopping callback
early_stop = EarlyStopping(
    monitor="loss",
    patience=config.NN_TRAINING_PARAMS["early_stopping_patience"],
    min_delta=config.NN_TRAINING_PARAMS["early_stopping_min_delta"],
    restore_best_weights=True,
)


def build_model(hp, input_dim: int) -> Sequential:
    """
    Builds a Keras MLP model with hyperparameter tuning support.
    """
    n_hidden = hp.Int("n_hidden", **config.NN_HP_SPACE["n_hidden"])
    d = hp.Float("dropout", **config.NN_HP_SPACE["dropout"])
    activation = hp.Choice("activation", config.NN_HP_SPACE["activation"])
    weight_decay = hp.Choice("l2_reg", [1e-6, 1e-5, 1e-4])

    model = Sequential()

    for i in range(n_hidden):
        units = hp.Choice(
            f"units{i+1}", config.NN_HP_SPACE[f"units{i+1}"]
        )
        # OPTIONAL per layer activation [increases seach space]
        # activation = hp.Choice(f"activation{i+1}", config.NN_HP_SPACE["activation"])

        if i == 0:
            model.add(Dense(
                units,
                input_shape=(input_dim,),
                kernel_regularizer=l2(weight_decay)
            ))
        else:
            model.add(Dense(
                units,
                kernel_regularizer=l2(weight_decay)
            ))
            
        if activation != "selu":
            model.add(BatchNormalization())
        
        model.add(Activation(activation))
        model.add(Dropout(d))

    model.add(Dense(3))

    model.compile(
        optimizer=Adam(
            hp.Float("learning_rate", **config.NN_HP_SPACE["learning_rate"])
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        
    )
    model.compiled = True
    return model





def mlp_nested_cv(
    X: pd.DataFrame,
    Y: pd.Series,
    estimation: Literal["Random", "Bayesian"] = "Random",
    project_name = "mlp"
):
    y_int = Y.map(config.LABEL_MAP).values
    input_dim = X.shape[1]

    outer_cv = TimeSeriesSplit(
        n_splits=config.CV_N_SPLITS, gap=config.MAX_HOLDING_PERIOD
    )
    acc_scores = []
    auc_scores = []
    ll_scores = []
    fold_hparams = []

    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X), 1):
        logger.info(f"\nOuter Fold {fold}/{config.CV_N_SPLITS}")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train_int, y_val_int = y_int[train_idx], y_int[val_idx]


        def model_builder(hp):
            return build_model(hp, input_dim)

        tuner_cls = (
            kt.BayesianOptimization if estimation == "Bayesian" else kt.RandomSearch
        )

        tuner = tuner_cls(
            model_builder,
            objective=Objective("val_loss",direction="min"),
            max_trials=config.NN_TRAINING_PARAMS["max_trials"],
            executions_per_trial=1,
            directory=f"kt/fold_{fold}",
            project_name=project_name,
            overwrite=True,
            seed=config.RANDOM_STATE,
        )

        tuner.search(
            X_train,
            y_train_int,
            epochs=config.NN_TRAINING_PARAMS["epochs"],
            validation_split=0.1,
            batch_size=config.NN_TRAINING_PARAMS["batch_size"],
            verbose=0,
            use_multiprocessing=True,
            workers=os.cpu_count(),
        )

        best_hp = tuner.get_best_hyperparameters(1)[0]
        best_model = tuner.hypermodel.build(best_hp)
        fold_hparams.append(best_hp)

        best_model.fit(
            X_train,
            y_train_int,
            validation_data=(X_val, y_val_int),
            epochs=config.NN_TRAINING_PARAMS["epochs"],
            batch_size=config.NN_TRAINING_PARAMS["batch_size"],
            callbacks=[early_stop],
            verbose=0,
        )

        logits = best_model.predict(X_val, verbose=0)
        y_proba = tf.nn.softmax(logits, axis=1).numpy()

        y_pred = np.argmax(y_proba, axis=1)
        acc = accuracy_score(y_val_int, y_pred)
        auc = roc_auc_score(y_val_int, y_proba,multi_class="ovr", average="macro")
        ll = log_loss(y_val_int, y_proba)
        acc_scores.append(acc)
        auc_scores.append(auc)
        ll_scores.append(ll)
        logger.info(
            f"\nBest model metrics for fold {fold}:"
            f"\nAccuracy: {acc:.4f}"
            f"\nROC AUC: {auc:.4f}"
            f"\nLog Loss: {ll:.4f}"
        )
        logger.info(f"\nBest HPs: {best_hp.values}")

    

    best_fold = np.argmax(auc_scores)
    best_fold_hp = fold_hparams[best_fold]

    model = train_MLP(X, Y, best_fold_hp, input_dim)
    filename = config.MODELS_DIR / f"{project_name}.pkl"
    model.save(filename)

    logger.info(
        f"\nAverage metrics of nested cross-validation tuning over {config.CV_N_SPLITS} folds:"
        f"\nMean Accuracy: {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}"
        f"\nMean ROC AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}"
        f"\nMean Log Loss: {np.mean(ll_scores):.4f} ± {np.std(ll_scores):.4f}"
        f"\nBest Fold selected is: {best_fold} with AUC {auc_scores[best_fold]:.4f}"
        f"\nTraining model on best parameters: {best_fold_hp.values}"
        f"\nModel saved to: {filename}"
    )

    return model, best_fold_hp, acc_scores,auc_scores,ll_scores, fold_hparams


def train_MLP(X: pd.DataFrame, Y: pd.Series, best_hp: kt.HyperParameters, input_dim: int) -> Sequential:
    y_int = Y.map(config.LABEL_MAP).values
    

    model = build_model(best_hp, input_dim)
    model.fit(
        X,
        y_int,
        epochs=config.NN_TRAINING_PARAMS["epochs"],
        batch_size=config.NN_TRAINING_PARAMS["batch_size"],
        callbacks=[early_stop],
        verbose=0,
    )

    return model






class KerasSoftmaxWrapper:
    def __init__(self, model, label_map: dict):
        self.model = model
        self.class_labels_ = sorted(label_map.keys(), key=lambda k: label_map[k])  # e.g., [-1, 0, 1]
        self.class_to_index = label_map

    def predict_proba(self, X):
        logits = self.model.predict(X)
        return tf.nn.softmax(logits, axis=1).numpy()

    def predict(self, X):
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return np.array([self.class_labels_[i] for i in indices])





def main():
    logger.info("=== 4. Load and prepare meta-modeling data ===")
    X_scaled = scale_features(load_features())
    Y = load_labels().squeeze()
    X_train_scaled, Y_train, X_test_scaled, Y_test = split_train_test(X_scaled, Y)

    
    logger.info("=== MPL training, hyperparm tuning with nested CV ===")
    clf, best_fold_hp, scores, hparams = mlp_nested_cv(
            X_train_scaled, Y_train, "Bayesian"
    )
    acc, auc, ll = evaluate_model(
        clf, X_test_scaled, Y_test
    )
    logger.info(
        f"\nTest Accuracy: {acc:.4f}"
        "\nTest ROC AUC: {auc:.4f}"
        "\nTest Log loss: {ll:.4f}"
    )


if __name__ == "__main__":
    main()
