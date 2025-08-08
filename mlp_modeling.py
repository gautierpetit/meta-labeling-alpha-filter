import logging
import os
import random
from typing import Literal

import keras_tuner as kt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import Activation, BatchNormalization, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from keras_tuner import Objective
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

import config
from modeling import scale_features

logger = logging.getLogger(__name__)

# Set random seeds
random.seed(config.RANDOM_STATE)
np.random.seed(config.RANDOM_STATE)
tf.random.set_seed(config.RANDOM_STATE)

# Configure TensorFlow to use deterministic operations
os.environ["TF_DETERMINISTIC_OPS"] = "1"

# Early stopping callback
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=config.NN_TRAINING_PARAMS["early_stopping_patience"],
    min_delta=config.NN_TRAINING_PARAMS["early_stopping_min_delta"],
    restore_best_weights=True,
    verbose=0,
)


def make_dataset(
    X: np.ndarray, y: np.ndarray, *, batch_size: int,) -> tf.data.Dataset:
    """
    Create a TensorFlow dataset from features and labels.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Label vector.
        batch_size (int): Batch size for the dataset.


    Returns:
        tf.data.Dataset: Prepared TensorFlow dataset.
    """
    ds = tf.data.Dataset.from_tensor_slices((X.astype(np.float32), y.astype(np.int32)))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(hp: kt.HyperParameters, input_dim: int, project_name="mlp") -> Sequential:
    """
    Build a Keras MLP model with hyperparameter tuning support.

    Args:
        hp (kt.HyperParameters): Hyperparameter tuning object.
        input_dim (int): Number of input features.

    Returns:
        Sequential: Compiled Keras model.
    """

    hp_space = config.MLPV1_HP_SPACE if project_name == "mlpv1t" else config.MLPV2_HP_SPACE

    n_hidden = hp.Int("n_hidden", **hp_space["n_hidden"])
    dropout_rate = hp.Float("dropout", **hp_space["dropout"])
    activation = hp.Choice("activation", hp_space["activation"])
    weight_decay = hp.Choice("l2_reg", hp_space["l2_reg"])

    model = Sequential()

    for i in range(n_hidden):
        units = hp.Choice(f"units{i + 1}", hp_space[f"units{i + 1}"])

        if i == 0:
            model.add(
                Dense(
                    units, input_shape=(input_dim,), kernel_regularizer=l2(weight_decay)
                )
            )
        else:
            model.add(Dense(units, kernel_regularizer=l2(weight_decay)))

        if hp_space["batch_norm"]:
            model.add(BatchNormalization())

        model.add(Activation(activation))
        model.add(Dropout(dropout_rate))

    model.add(Dense(3))

    model.compile(
        optimizer=Adam(
            hp.Float("learning_rate", **hp_space["learning_rate"])
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )
    return model


def mlp_nested_cv(
    X: pd.DataFrame,
    Y: pd.Series,
    estimation: Literal["Random", "Bayesian"] = "Random",
    project_name="mlp",
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

    hp_space = config.MLPV1_HP_SPACE if project_name == "mlpv1t" else config.MLPV2_HP_SPACE

    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X), 1):
        logger.info(f"\nOuter Fold {fold}/{config.CV_N_SPLITS}")

        X_train_raw, X_val_raw = X.iloc[train_idx], X.iloc[val_idx]
        X_train, fold_scaler = scale_features(X_train_raw, return_scaler=True)
        X_val = pd.DataFrame(fold_scaler.transform(X_val_raw), index=X_val_raw.index, columns=X_val_raw.columns)

        y_train_int, y_val_int = y_int[train_idx], y_int[val_idx]

        def model_builder(hp):
            return build_model(hp, input_dim, project_name=project_name)

        tuner_cls = (
            kt.BayesianOptimization if estimation == "Bayesian" else kt.RandomSearch
        )

        tuner = tuner_cls(
            model_builder,
            objective=Objective("val_loss", direction="min"),
            max_trials=hp_space["max_trials"],
            executions_per_trial=1,
            directory=f"kt/fold_{fold}",
            project_name=project_name,
            overwrite=False,
            seed=config.RANDOM_STATE,
        )

        train_ds = make_dataset(
            X_train.values,
            y_train_int,
            batch_size=hp_space["batch_size"],
        )
        val_ds = make_dataset(
            X_val.values,
            y_val_int,
            batch_size=hp_space["batch_size"],
            )
        tuner.search(
            train_ds,
            validation_data=val_ds,  # Provide validation dataset here
            epochs=hp_space["epochs"],
            verbose=0,
        )

        best_hp = tuner.get_best_hyperparameters(1)[0]
        best_model = tuner.hypermodel.build(best_hp)
        fold_hparams.append(best_hp)

        early_stop = EarlyStopping(
            monitor="loss",
            patience=config.NN_TRAINING_PARAMS["early_stopping_patience"],
            min_delta=config.NN_TRAINING_PARAMS["early_stopping_min_delta"],
            restore_best_weights=True,
            verbose=0,
        )

        best_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=hp_space["epochs"],
            callbacks=[early_stop],
            verbose=0,
        )

        logits = best_model.predict(X_val, verbose=0)
        y_proba = tf.nn.softmax(logits, axis=1).numpy()

        y_pred = np.argmax(y_proba, axis=1)
        acc = accuracy_score(y_val_int, y_pred)
        auc = roc_auc_score(y_val_int, y_proba, multi_class="ovr", average="macro")
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

    model, final_scaler = train_MLP(X, Y, best_fold_hp, input_dim, project_name=project_name)
    filename = config.MODELS_DIR / f"{project_name}.pkl"
    model.save(filename)

    logger.info(
        f"\nAverage metrics of nested cross-validation tuning over {config.CV_N_SPLITS} folds:"
        f"\nMean Accuracy: {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}"
        f"\nMean ROC AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}"
        f"\nMean Log Loss: {np.mean(ll_scores):.4f} ± {np.std(ll_scores):.4f}"
        f"\nBest Fold selected is: {best_fold+1} with AUC {auc_scores[best_fold]:.4f}"
        f"\nTraining model on best parameters: {best_fold_hp.values}"
        f"\nModel saved to: {filename}"
    )

    return model, final_scaler, best_fold_hp, acc_scores, auc_scores, ll_scores, fold_hparams


def train_MLP(
    X: pd.DataFrame, Y: pd.Series, best_hp: kt.HyperParameters, input_dim: int, project_name="mlp"
) -> Sequential:
    y_int = Y.map(config.LABEL_MAP).values  
    hp_space = config.MLPV1_HP_SPACE if project_name == "mlpv1t" else config.MLPV2_HP_SPACE

    X_scaled, final_scaler = scale_features(X, return_scaler=True)

    model = build_model(best_hp, input_dim)
    train_ds = make_dataset(
        X.values, y_int, batch_size=hp_space["batch_size"]
    )
    model.fit(
        train_ds,
        epochs=hp_space["epochs"],
        callbacks=[early_stop],
        verbose=0,
    )

    return model, final_scaler


class KerasSoftmaxWrapper:
    def __init__(self, model, label_map: dict, scaler=None):
        self.model = model
        self.class_labels_ = sorted(
            label_map.keys(), key=lambda k: label_map[k]
        )  # e.g., [-1, 0, 1]
        self.class_to_index = label_map
        self.scaler = scaler

    def predict_proba(self, X):
        if self.scaler is None:
            raise ValueError("Scaler not found. Ensure model was trained with consistent scaling.")
        X = pd.DataFrame(self.scaler.transform(X), index=X.index, columns=X.columns)
        logits = self.model.predict(X)
        return tf.nn.softmax(logits, axis=1).numpy()

    def predict(self, X):
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return np.array([self.class_labels_[i] for i in indices])

