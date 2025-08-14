
import json
import logging
import math
import os
import random
from dataclasses import dataclass
from typing import Literal

import joblib
import keras_tuner as kt
import numpy
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Activation, BatchNormalization, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from keras_tuner import Objective
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import src.config as config
from src.analysis import plot_learning_curve, save_history
from src.modeling import scale_features
from src.utils import _rolling_windows

logger = logging.getLogger(__name__)

# Set random seeds
random.seed(config.RANDOM_STATE)
np.random.seed(config.RANDOM_STATE)
tf.random.set_seed(config.RANDOM_STATE)

# Configure TensorFlow to use deterministic operations
os.environ["TF_DETERMINISTIC_OPS"] = "1"


# put this near your other helpers
def build_callbacks(monitor="val_loss"):
    return [
        EarlyStopping(
            monitor=monitor,
            mode="min",
            patience=config.NN_TRAINING_PARAMS["early_stopping_patience"],
            min_delta=config.NN_TRAINING_PARAMS["early_stopping_min_delta"],
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor=monitor,
            mode="min",
            factor=config.NN_TRAINING_PARAMS["reduce_lr_factor"],
            patience=config.NN_TRAINING_PARAMS["reduce_lr_patience"],
            min_lr=config.NN_TRAINING_PARAMS["reduce_lr_min_lr"],
            verbose=0,
        ),
    ]


def _safe_transform(scaler, X):
    if hasattr(scaler, "feature_names_in_"):
        means = pd.Series(scaler.mean_, index=scaler.feature_names_in_)
        X = X.reindex(columns=scaler.feature_names_in_).fillna(means)
    Xt = scaler.transform(X)
    Xt = np.clip(Xt, -8, 8)  # guards against the Amihud-type spikes
    return pd.DataFrame(Xt, index=X.index, columns=X.columns)


def steps_per_epoch_for(y_len: int, batch_size: int) -> int:
    return max(1, math.ceil(y_len / batch_size))


def make_dataset(
    X: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    sample_weight: np.ndarray | None = None,
) -> tf.data.Dataset:
    """
    Create a TensorFlow dataset from features and labels.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Label vector.
        batch_size (int): Batch size for the dataset.


    Returns:
        tf.data.Dataset: Prepared TensorFlow dataset.
    """

    X = X.astype(np.float32)
    y = y.reshape(-1).astype(np.int32)
    if sample_weight is None:
        ds = tf.data.Dataset.from_tensor_slices((X, y))
    else:
        sw = sample_weight.astype(np.float32)
        ds = tf.data.Dataset.from_tensor_slices((X, y, sw))
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def make_balanced_dataset(
    X: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    seed: int = 42,
) -> tf.data.Dataset:
    """Create a balanced tf.data pipeline with ~equal class sampling."""
    X = X.astype(np.float32)
    y = y.reshape(-1).astype(np.int32)
    base = tf.data.Dataset.from_tensor_slices((X, y))

    # split by class indices (assuming LABEL_MAP maps {-1,0,1} -> {0,1,2})
    ds0 = base.filter(lambda x, t: tf.equal(t, 0)).repeat()
    ds1 = base.filter(lambda x, t: tf.equal(t, 1)).repeat()
    ds2 = base.filter(lambda x, t: tf.equal(t, 2)).repeat()

    ds_bal = tf.data.Dataset.sample_from_datasets(
        [ds0, ds1, ds2], weights=[1 / 3, 1 / 3, 1 / 3], seed=seed
    )
    return ds_bal.batch(batch_size).prefetch(tf.data.AUTOTUNE)


@tf.keras.utils.register_keras_serializable(package="custom")
class SparseCCELabelSmoothing(tf.keras.losses.Loss):
    def __init__(
        self, n_classes, label_smoothing=0.0, from_logits=True, name="sparse_cce_ls"
    ):
        # Make the outer loss produce a scalar per batch
        super().__init__(
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name=name
        )
        self.n_classes = int(n_classes)
        self.from_logits = bool(from_logits)
        self.label_smoothing = float(label_smoothing)
        # Inner CE stays unreduced -> returns a vector of per-example losses
        self._ce = tf.keras.losses.CategoricalCrossentropy(
            from_logits=self.from_logits,
            label_smoothing=self.label_smoothing,
            reduction=tf.keras.losses.Reduction.NONE,
        )

    def call(self, y_true, y_pred):
        # ensure 1D labels before one-hot
        y_true = tf.cast(tf.squeeze(y_true), tf.int32)
        y_true_oh = tf.one_hot(y_true, depth=self.n_classes)
        per_example = self._ce(y_true_oh, y_pred)   # shape (batch,)
        return per_example  # outer Loss (SUM_OVER_BATCH_SIZE) reduces to scalar


    def get_config(self):
        return {
            "n_classes": self.n_classes,
            "label_smoothing": self.label_smoothing,
            "from_logits": self.from_logits,
            "name": self.name,
        }


def build_model(
    hp: kt.HyperParameters, input_dim: int, bias_init: str, project_name="mlp"
) -> Sequential:
    """
    Build a Keras MLP model with hyperparameter tuning support.

    Args:
        hp (kt.HyperParameters): Hyperparameter tuning object.
        input_dim (int): Number of input features.

    Returns:
        Sequential: Compiled Keras model.
    """

    hp_space = (
        config.MLPV1_HP_SPACE if project_name == "mlpv1t" else config.MLPV2_HP_SPACE
    )

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

    model.add(Dense(3, bias_initializer=bias_init))

    n_classes = len(config.LABEL_MAP)  # 3
    loss_fn = SparseCCELabelSmoothing(
        n_classes=n_classes,
        label_smoothing=hp_space["label_smoothing"],
        from_logits=True,
    )
    model.compile(
        optimizer=Adam(
            hp.Float("learning_rate", **hp_space["learning_rate"]), clipnorm=1.0
        ),
        loss=loss_fn,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    return model


def train_MLP(
    X: pd.DataFrame,
    Y: pd.Series,
    best_hp: kt.HyperParameters,
    input_dim: int,
    project_name="mlp",
) -> Sequential:
    y_int = Y.map(config.LABEL_MAP).values
    hp_space = (
        config.MLPV1_HP_SPACE if project_name == "mlpv1t" else config.MLPV2_HP_SPACE
    )

    X_scaled, final_scaler = scale_features(X, return_scaler=True)

    counts = np.bincount(y_int, minlength=3).astype(float)
    eps = 1e-3
    priors = (counts + eps) / (counts.sum() + 3 * eps)
    bias_init = tf.keras.initializers.Constant(np.log(priors))

    model = build_model(
        best_hp, input_dim, bias_init=bias_init, project_name=project_name
    )

    cut = int(0.85 * len(X_scaled))
    Xtr, ytr = X_scaled.iloc[:cut], y_int[:cut]
    Xva, yva = X_scaled.iloc[cut:], y_int[cut:]

    train_ds = make_balanced_dataset(
        Xtr.values, ytr, batch_size=hp_space["batch_size"], seed=config.RANDOM_STATE
    )
    val_ds = make_dataset(Xva.values, yva, batch_size=hp_space["batch_size"])

    bsize = hp_space["batch_size"]
    train_steps = steps_per_epoch_for(len(ytr), bsize)
    val_steps = steps_per_epoch_for(len(yva), bsize)

    cb_final = build_callbacks()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=hp_space["epochs"],
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=cb_final,
        verbose=0,
    )

    model_tag = f"{project_name}_final"
    plot_learning_curve(history, name=model_tag)
    save_history(history, name=model_tag)

    return model, final_scaler


def mlp_nested_cv(
    X: pd.DataFrame,
    Y: pd.Series,
    estimation: Literal["Random", "Bayesian"] = "Random",
    project_name="mlp",
):
    y_int = Y.map(config.LABEL_MAP).values
    input_dim = X.shape[1]

    outer_cv = TimeSeriesSplit(n_splits=config.CV_N_SPLITS, gap=config.CV_GAP)
    acc_scores, auc_scores, ll_scores, hparams_all = [], [], [], []

    hp_space = (
        config.MLPV1_HP_SPACE if project_name == "mlpv1t" else config.MLPV2_HP_SPACE
    )

    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X), 1):
        logger.info(f"\nOuter Fold {fold}/{config.CV_N_SPLITS}")

        X_train_raw, X_val_raw = X.iloc[train_idx], X.iloc[val_idx]
        y_train_int, y_val_int = y_int[train_idx], y_int[val_idx]

        # Fit scaler on training split only
        X_train_scaled, fold_scaler = scale_features(X_train_raw, return_scaler=True)
        X_val_scaled = _safe_transform(fold_scaler, X_val_raw)

        # Initialize bias for the output layer based on class priors
        counts = np.bincount(y_train_int, minlength=3).astype(float)
        eps = 1e-3
        priors = (counts + eps) / (counts.sum() + 3 * eps)
        bias_init = tf.keras.initializers.Constant(np.log(priors))

        def model_builder(hp):
            return build_model(
                hp, input_dim, bias_init=bias_init, project_name=project_name
            )

        tuner_cls = (
            kt.BayesianOptimization if estimation == "Bayesian" else kt.RandomSearch
        )

        tuner = tuner_cls(
            model_builder,
            objective=Objective("val_loss", direction="min"),
            max_trials=hp_space["max_trials"],
            executions_per_trial=1,
            directory=f"{config.MODELS_DIR}/kt/fold_{fold}",
            project_name=project_name,
            overwrite=True,
            seed=config.RANDOM_STATE,
        )

        train_ds = make_balanced_dataset(
            X_train_scaled.values,
            y_train_int,
            batch_size=hp_space["batch_size"],
            seed=config.RANDOM_STATE,
        )
        val_ds = make_dataset(
            X_val_scaled.values,
            y_val_int,
            batch_size=hp_space["batch_size"],
            sample_weight=None,
        )

        bsize = hp_space["batch_size"]
        train_steps = steps_per_epoch_for(len(y_train_int), bsize)
        val_steps = steps_per_epoch_for(len(y_val_int), bsize)

        cb_search = build_callbacks()
        tuner.search(
            train_ds,
            validation_data=val_ds,
            epochs=hp_space["epochs"],
            steps_per_epoch=train_steps,
            validation_steps=val_steps,
            callbacks=cb_search,
            verbose=0,
        )

        best_hp = tuner.get_best_hyperparameters(1)[0]
        best_model = tuner.hypermodel.build(best_hp)
        hparams_all.append(best_hp)

        cb_refit = build_callbacks()
        history = best_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=hp_space["epochs"],
            steps_per_epoch=train_steps,
            validation_steps=val_steps,
            callbacks=cb_refit,
            verbose=0,
        )

        fold_tag = f"{project_name}_fold{fold}"
        plot_learning_curve(history, name=fold_tag)
        save_history(history, name=fold_tag)

        logits = best_model.predict(X_val_scaled.values, verbose=0)
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

    ll = np.array(ll_scores)
    auc = np.array(auc_scores)

    # Ranks: smaller ll is better; larger auc is better
    rank_ll  = np.argsort(np.argsort(ll))                   # 0 = best
    rank_auc = np.argsort(np.argsort(-auc))                 # 0 = best

    # Heavily weight log-loss, use AUC as tie-breaker
    rank_sum = rank_ll + rank_auc
    best_fold = int(np.argmin(rank_sum))
    best_fold_hp = hparams_all[best_fold]

    model, final_scaler = train_MLP(
        X, Y, best_fold_hp, input_dim, project_name=project_name
    )
    filename = config.MODELS_DIR / f"{project_name}.keras"
    model.save(filename)

    logger.info(
        f"\nAverage metrics of nested cross-validation tuning over {config.CV_N_SPLITS} folds:"
        f"\nMean Accuracy: {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}"
        f"\nMean ROC AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}"
        f"\nMean Log Loss: {np.mean(ll_scores):.4f} ± {np.std(ll_scores):.4f}"
        f"\nBest Fold selected is: {best_fold+1} "
        f"(selector = 2*rank(NLL)+rank(AUC); "
        f"NLL={ll[best_fold]:.4f}, AUC={auc[best_fold]:.4f})"
        f"\nTraining model on best parameters: {best_fold_hp.values}"
        f"\nModel saved to: {filename}"
    )

    return (
        model,
        final_scaler,
        best_fold_hp,
        acc_scores,
        auc_scores,
        ll_scores,
        hparams_all,
    )



@dataclass
class Bundle:
    model: tf.keras.Model
    scaler: object  # e.g., StandardScaler
    class_labels: np.ndarray  # e.g., np.array([-1, 0, 1])

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        # SavedModel so we don't need custom_objects at load time
        self.model.save(os.path.join(path, "model"), include_optimizer=False)
        joblib.dump(self.scaler, os.path.join(path, "scaler.pkl"))
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump({"class_labels": self.class_labels.tolist()}, f)

    @classmethod
    def load(cls, path: str, compile: bool = False):
        model = tf.keras.models.load_model(os.path.join(path, "model"), compile=compile)
        scaler = joblib.load(os.path.join(path, "scaler.pkl"))
        with open(os.path.join(path, "meta.json"), "r") as f:
            meta = json.load(f)
        return cls(
            model=model, scaler=scaler, class_labels=np.array(meta["class_labels"])
        )



class VectorScaledSoftmax:
    """
    Post-hoc multiclass calibration:
      z'_k = a_k * z_k + b_k,   p = softmax(z')
    Compatible with TF 2.10. Trains a,b on a held-out validation set
    by minimizing multinomial NLL (with small L2 on (a-1,b)).
    """

    def __init__(
        self,
        model,
        label_map: dict,
        scaler=None,
        a: np.ndarray | None = None,
        b: np.ndarray | None = None,
    ):
        self.model = model
        # order classes by model index: e.g. {-1:0, 0:1, 1:2}
        self.class_labels_ = sorted(label_map.keys(), key=lambda k: label_map[k])
        self.class_to_index = label_map
        self.scaler = scaler
        K = len(self.class_labels_)
        self.a = (
            np.ones(K, dtype=np.float64)
            if a is None
            else np.asarray(a, dtype=np.float64)
        )
        self.b = (
            np.zeros(K, dtype=np.float64)
            if b is None
            else np.asarray(b, dtype=np.float64)
        )

    @staticmethod
    def _softmax_np(z: np.ndarray) -> np.ndarray:
        z = z - np.max(z, axis=1, keepdims=True)
        ez = np.exp(z)
        return ez / np.sum(ez, axis=1, keepdims=True)

    @classmethod
    def from_validation(
        cls,
        model,
        label_map,
        scaler,
        X_val: pd.DataFrame,
        y_val_int: np.ndarray,
        reg: float = 1e-3,
        lr: float = 0.05,
        max_iter: int = 1000,
        tol: float = 1e-7,
        patience: int = 20,
    ):
        """
        Fit a,b on a validation fold and return a ready-to-use calibrator.
        y_val_int must be integer-coded according to label_map values (0..K-1).
        """
        if hasattr(model, "predict_logits"):
            # e.g. MetaLogit: it handles its own scaling/columns
            logits = model.predict_logits(X_val)  # (n, K)
        else:
            if scaler is None:
                raise ValueError(
                    "Scaler required unless model exposes `predict_logits`."
                )
            Xt = _safe_transform(scaler, X_val)
            logits = model.predict(Xt.values, verbose=0)

        n, K = logits.shape

        # ensure labels are 0..K-1
        y_val_int = np.asarray(y_val_int)
        if y_val_int.min() < 0 or y_val_int.max() >= logits.shape[1]:
            raise ValueError(
                "y_val_int must be integer-coded in [0..K-1]. "
                "Did you forget to map with config.LABEL_MAP?"
            )

        # TF variables
        a = tf.Variable(np.ones(K, dtype=np.float32))
        b = tf.Variable(np.zeros(K, dtype=np.float32))
        y_tf = tf.convert_to_tensor(y_val_int, tf.int32)
        z_tf = tf.convert_to_tensor(logits, tf.float32)

        opt = tf.keras.optimizers.Adam(learning_rate=lr)

        best = np.inf
        stall = 0

        @tf.function
        def step():
            with tf.GradientTape() as tape:
                # z' = a*z + b
                z_prime = z_tf * a[tf.newaxis, :] + b[tf.newaxis, :]
                # NLL
                loss = tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(
                        y_tf, z_prime, from_logits=True
                    )
                )
                # small L2 on (a-1, b) to prevent extreme values
                loss += reg * (tf.reduce_sum((a - 1.0) ** 2) + tf.reduce_sum(b**2))
            grads = tape.gradient(loss, [a, b])
            opt.apply_gradients(zip(grads, [a, b]))
            return loss

        for _ in range(max_iter):
            loss = float(step().numpy())
            if best - loss > tol:
                best = loss
                stall = 0
            else:
                stall += 1
                if stall >= patience:
                    break

        return cls(
            model=model,
            label_map=label_map,
            scaler=scaler,
            a=a.numpy().astype(np.float64),
            b=b.numpy().astype(np.float64),
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if hasattr(self.model, "predict_logits"):
            logits = self.model.predict_logits(X)
        else:
            if self.scaler is None:
                raise ValueError(
                    "Scaler not found. Pass a scaler or provide `predict_logits`."
                )
            Xt = _safe_transform(self.scaler, X)
            logits = self.model.predict(Xt.values, verbose=0)

        z_prime = logits * self.a[np.newaxis, :] + self.b[np.newaxis, :]
        return self._softmax_np(z_prime)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return np.array([self.class_labels_[i] for i in idx])

    # Optional helpers to persist calibration params
    def get_params(self) -> dict:
        return {
            "a": self.a.copy(),
            "b": self.b.copy(),
            "class_labels": self.class_labels_,
        }

    def set_params(self, a: np.ndarray, b: np.ndarray):
        self.a = np.asarray(a, dtype=np.float64)
        self.b = np.asarray(b, dtype=np.float64)



class RollingVectorScaledSoftmax:
    """
    Leakage-free post-hoc calibration for a single base model over time.
    - Inside Fold-2: split chronologically; for each segment j, fit VectorScaling
      on an earlier slice and use it to predict the next slice (OOS).
    - Outside Fold-2 (e.g., Fold-3): use the last fitted (most recent) scaling.

    """

    def __init__(self, model, label_map, scaler, segments, default_ab):
        self.model = model
        self.scaler = scaler
        self.class_labels_ = sorted(label_map.keys(), key=lambda k: label_map[k])
        self.class_to_index = label_map
        self.K = len(self.class_labels_)
        # segments: list of (start_label, end_label, a_vec, b_vec)
        self.segments = segments
        # default (a,b) used outside all segments
        self.default_a, self.default_b = default_ab

    @staticmethod
    def _softmax_np(z):
        z = z - np.max(z, axis=1, keepdims=True)
        ez = np.exp(z)
        return ez / np.sum(ez, axis=1, keepdims=True)

    @classmethod
    def from_fold2(
        cls,
        model,
        label_map,
        scaler,
        X_fold2: pd.DataFrame,
        y_fold2_int: np.ndarray,
        n_splits: int = 3,
        embargo: int = 5,
        *,
        reg=1e-3,
        lr=0.05,
        max_iter=1000,
        tol=1e-7,
        patience=20,
    ):
        """
        Build a leakage-free rolling calibrator over Fold-2 using n_splits equal slices
        with an embargo (in rows) between fit and predict slices.
        """
        n = len(X_fold2)
        idx = X_fold2.index
        segments = []
        last_a = np.ones(len(label_map))
        last_b = np.zeros(len(label_map))

        for cal_end, pred_start, pred_end in _rolling_windows(n, n_splits, embargo):
            vs = VectorScaledSoftmax.from_validation(
                model,
                label_map,
                scaler,
                X_fold2.iloc[:cal_end],
                y_fold2_int[:cal_end],
                reg=reg,
                lr=lr,
                max_iter=max_iter,
                tol=tol,
                patience=patience,
            )
            last_a, last_b = vs.a.copy(), vs.b.copy()
            start_label = idx[pred_start]
            end_label = idx[pred_end - 1]
            segments.append((start_label, end_label, last_a, last_b))

        return cls(model, label_map, scaler, segments, (last_a, last_b))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.scaler is None:
            raise ValueError(
                "Scaler not found. Ensure model was trained with consistent scaling."
            )
        if hasattr(self.scaler, "feature_names_in_"):
            exp = list(self.scaler.feature_names_in_)
            got = list(X.columns)
            if exp != got:
                raise ValueError(
                    "Meta feature schema mismatch for calibrator.\n"
                    f"Expected (from scaler): {exp[:5]}... (len={len(exp)})\n"
                    f"Got: {got[:5]}... (len={len(got)})"
                )
        Xt = _safe_transform(self.scaler, X)
        logits = self.model.predict(Xt.values, verbose=0)
        out = np.full((len(X), self.K), np.nan, dtype=np.float64)

        # apply segment-specific (a,b) inside Fold-2 ranges
        for start_label, end_label, a, b in self.segments:
            mask = (X.index >= start_label) & (X.index <= end_label)
            if not np.any(mask):
                continue
            z_prime = logits[mask] * a[np.newaxis, :] + b[np.newaxis, :]
            out[mask] = self._softmax_np(z_prime)

        # anything not covered (e.g., Fold-3) uses the most recent (a,b)
        missing = np.isnan(out).any(axis=1)
        if np.any(missing):
            z_prime = (
                logits[missing] * self.default_a[np.newaxis, :]
                + self.default_b[np.newaxis, :]
            )
            out[missing] = self._softmax_np(z_prime)

        return out

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return np.array([self.class_labels_[i] for i in idx])









class ConvexProbabilityBlender:
    """
    Learns a single weight w in [0,1] to blend calibrated proba from
    the two base models present in X_meta columns:
      p_blend = w * p_mlp + (1-w) * p_clf
    Expects columns: proba_clf_-1, proba_clf_0, proba_clf_1,
                     proba_mlp_-1, proba_mlp_0, proba_mlp_1
    """

    def __init__(self, label_map, w=0.5):
        self.class_labels_ = sorted(label_map.keys(), key=lambda k: label_map[k])
        self.class_to_index = label_map
        self.w = float(w)

    @staticmethod
    def _nll(y_true_int, proba, eps=1e-12):
        p = np.clip(proba, eps, 1 - eps)
        p /= p.sum(axis=1, keepdims=True)
        return -np.mean(np.log(p[np.arange(len(y_true_int)), y_true_int]))

    @classmethod
    def from_fold2(cls, label_map, X_meta_f2, y_fold2_int):
        # grid-search w in [0,1]
        grid = np.linspace(0.0, 1.0, 21)
        proba_clf = X_meta_f2[["proba_clf_-1", "proba_clf_0", "proba_clf_1"]].values
        proba_mlp = X_meta_f2[["proba_mlp_-1", "proba_mlp_0", "proba_mlp_1"]].values

        best_w, best_nll = 0.5, np.inf
        for w in grid:
            p = w * proba_mlp + (1.0 - w) * proba_clf
            nll = cls._nll(y_fold2_int, p)
            if nll < best_nll:
                best_nll, best_w = nll, w
        return cls(label_map, w=best_w)

    # keep interface similar to your other "model" wrappers
    def predict_proba(self, X_meta: pd.DataFrame) -> np.ndarray:
        proba_clf = X_meta[["proba_clf_-1", "proba_clf_0", "proba_clf_1"]].values
        proba_mlp = X_meta[["proba_mlp_-1", "proba_mlp_0", "proba_mlp_1"]].values
        p = self.w * proba_mlp + (1.0 - self.w) * proba_clf
        # normalize just in case
        p = np.clip(p, 1e-12, 1.0)
        p /= p.sum(axis=1, keepdims=True)
        return p

    def predict(self, X_meta: pd.DataFrame) -> np.ndarray:
        p = self.predict_proba(X_meta)
        idx = np.argmax(p, axis=1)
        return np.array([self.class_labels_[i] for i in idx])





class MetaLogit:
    """Multinomial logistic regression meta-learner on X_meta."""

    def __init__(self, label_map, C=1.0):
        self.class_labels_ = sorted(label_map.keys(), key=lambda k: label_map[k])
        self.class_to_index = label_map
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            C=C,
            class_weight="balanced",
            max_iter=1000,
            n_jobs=-1,
        )
        self.selected_cols = None

    def fit(self, X_meta: pd.DataFrame, y):
        # Select robust columns
        cols = [
            "proba_clf_-1",
            "proba_clf_0",
            "proba_clf_1",
            "proba_mlp_-1",
            "proba_mlp_0",
            "proba_mlp_1",
            "proba_gap_clf",
            "proba_gap_mlp",
            "confidence_mean",
            "confidence_agreement",
            "model_agreement",
            "logodds_clf",
            "logodds_mlp",
            "margin_clf",
            "margin_mlp",
        ]
        self.selected_cols = [c for c in cols if c in X_meta.columns]
        Xs = self.scaler.fit_transform(X_meta[self.selected_cols].values)
        y_int = pd.Series(y).map(self.class_to_index).values
        self.model.fit(Xs, y_int)
        return self

    def predict_proba(self, X_meta: pd.DataFrame) -> np.ndarray:
        Xs = self.scaler.transform(X_meta[self.selected_cols].values)
        proba = self.model.predict_proba(Xs)
        # scikit uses class order [0,1,2] already
        return proba

    def predict(self, X_meta: pd.DataFrame) -> np.ndarray:
        idx = self.model.predict(
            self.scaler.transform(X_meta[self.selected_cols].values)
        )
        return np.array([self.class_labels_[i] for i in idx])

    def predict_logits(self, X_meta: pd.DataFrame) -> np.ndarray:
        # use the same selected_cols you fit on
        Xs = self.scaler.transform(X_meta[self.selected_cols].values)
        # multinomial LR returns class-wise linear scores (pre-softmax)
        return self.model.decision_function(Xs)  # shape (n_samples, K)








class ClasswiseConvexBlender:
    """
    p_k = w_k * p_clf_k + (1 - w_k) * p_mlp_k,  w_k in (0,1) learned on Fold-2.
    Holds references to two *calibrated* base model wrappers that expose predict_proba(X).
    """

    def __init__(self, model_clf, model_mlp, label_map: dict):
        # base models used during *fit* (Fold-2)
        self.model_clf = model_clf
        self.model_mlp = model_mlp
        # order classes by model index: e.g. {-1:0,0:1,1:2}
        self.class_labels_ = np.array(
            sorted(label_map.keys(), key=lambda k: label_map[k])
        )
        self.class_to_index = label_map
        self.w = np.full(len(self.class_labels_), 0.5, dtype=np.float64)  # start at 0.5

    def _blend(self, p_clf, p_mlp, w_sig):
        p = w_sig[np.newaxis, :] * p_clf + (1.0 - w_sig[np.newaxis, :]) * p_mlp
        p = np.clip(p, 1e-9, 1.0)
        p /= p.sum(axis=1, keepdims=True)
        return p

    def fit(
        self,
        X_fold2,
        y_fold2_int,
        *,
        lr=0.05,
        reg=1e-4,
        max_iter=2000,
        patience=50,
        tol=1e-7,
        verbose=False,
    ):
        """Learn w_k on Fold-2 using calibrated base models passed at __init__."""
        p1 = self.model_clf.predict_proba(X_fold2)  # (N, K)
        p2 = self.model_mlp.predict_proba(X_fold2)  # (N, K)

        # tensors
        w = tf.Variable(self.w.astype(np.float32))
        y = tf.convert_to_tensor(y_fold2_int.astype(np.int32))
        P1 = tf.convert_to_tensor(p1.astype(np.float32))
        P2 = tf.convert_to_tensor(p2.astype(np.float32))
        opt = tf.keras.optimizers.Adam(lr)

        @tf.function
        def step():
            # unconstrained w -> (0,1) via sigmoid
            w_sig = tf.clip_by_value(tf.sigmoid(w), 1e-3, 1.0 - 1e-3)
            P = w_sig[tf.newaxis, :] * P1 + (1.0 - w_sig[tf.newaxis, :]) * P2
            loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(y, P, from_logits=False)
            )
            # tiny shrinkage toward fairness (0.5) to reduce variance
            loss += reg * tf.reduce_sum((w_sig - 0.5) ** 2)
            grads = tf.gradients(loss, [w])[0]
            opt.apply_gradients([(grads, w)])
            return loss

        best = np.inf
        stall = 0
        for _ in range(max_iter):
            loss = float(step().numpy())
            if best - loss > tol:
                best, stall = loss, 0
            else:
                stall += 1
                if stall >= patience:
                    break

        self.w = tf.sigmoid(w).numpy().astype(np.float64)
        if verbose:
            print(
                f"Classwise blender weights (by class order {self.class_labels_.tolist()}): {self.w}"
            )
        return self

    # --- inference wiring ---
    def with_inference_models(self, model_clf, model_mlp):
        """Return a copy that reuses learned weights but swaps the base models (e.g., Fold-3 calibrators)."""
        clone = ClasswiseConvexBlender(model_clf, model_mlp, self.class_to_index)
        clone.w = self.w.copy()
        return clone

    # evaluate_model expects predict_proba(X) / predict(X)
    def predict_proba(self, X):
        p1 = self.model_clf.predict_proba(X)
        p2 = self.model_mlp.predict_proba(X)
        p = self._blend(p1, p2, self.w)
        return p

    def predict(self, X):
        p = self.predict_proba(X)
        idx = p.argmax(axis=1)
        return self.class_labels_[idx]

    
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        payload = {
            "class_labels": self.class_labels_.tolist(),
            "w": self.w.tolist(),
        }
        with open(os.path.join(path, "blender_cw.json"), "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str, model_clf, model_mlp, label_map: dict):
        with open(os.path.join(path, "blender_cw.json"), "r") as f:
            payload = json.load(f)
        obj = cls(model_clf, model_mlp, label_map)
        obj.class_labels_ = np.array(payload["class_labels"])
        obj.w = np.array(payload["w"], dtype=np.float64)
        # (optional) sanity checks
        assert len(obj.w) == len(obj.class_labels_), "w/class mismatch"
        return obj