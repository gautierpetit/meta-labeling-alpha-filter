import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
import argparse
import joblib
import numpy as np
import tensorflow as tf
from sklearn.base import ClassifierMixin
import shutil
from pathlib import Path


def get_class_to_index(clf: ClassifierMixin) -> dict:
    """
    Retrieve a mapping of class labels to indices from the classifier.

    Args:
        clf (ClassifierMixin): The classifier object.

    Returns:
        dict: Mapping of class labels to indices.

    Raises:
        ValueError: If the classifier does not define `class_labels_` or `classes_`.
    """
    if hasattr(clf, "class_labels_"):
        return {label: i for i, label in enumerate(clf.class_labels_)}
    elif hasattr(clf, "classes_"):
        return {cls: i for i, cls in enumerate(clf.classes_)}
    else:
        raise ValueError("Classifier must define `class_labels_` or `classes_`.")


def _rolling_windows(n: int, n_splits: int = 3, embargo: int = 5):
    """
    Build (cal_end, pred_start, pred_end) triplets in row indices:
      - cal window:   [0 : cal_end)
      - pred window:  [pred_start : pred_end)
      - pred_start = cal_end + embargo
    Returns a list covering Fold-2 sequentially; last partial window is skipped if empty.
    """
    assert n_splits >= 2
    cuts = [0] + [int(round(n * i / n_splits)) for i in range(1, n_splits)] + [n]
    windows = []
    for j in range(1, len(cuts) - 1):
        cal_end = cuts[j]
        pred_start = min(cal_end + max(embargo, 0), n)
        pred_end = cuts[j + 1]
        if pred_end > pred_start:
            windows.append((cal_end, pred_start, pred_end))
    return windows


def _to_3d_shap(sv):
    import numpy as np

    if isinstance(sv, list):  # e.g., TreeExplainer multiclass
        return np.stack(sv, axis=2)  # [(n,f)] -> (n,f,C)
    sv = np.asarray(sv)
    if sv.ndim == 2:  # binary/single-output
        return sv[..., None]
    return sv  # already (n,f,C)


def make_run_id(tag: str | None = None) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{tag}" if tag else ts


def md5_columns(df) -> str:
    s = ",".join(map(str, df.columns))
    h = hashlib.md5(s.encode()).hexdigest()
    return h


def safe_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def write_json(path, obj):
    def convert_keys(o):
        if isinstance(o, dict):
            return {str(k): convert_keys(v) for k, v in o.items()}
        elif isinstance(o, list):
            return [convert_keys(i) for i in o]
        elif isinstance(o, (np.int32, np.int64)):  # Handle numpy integer types
            return int(o)
        elif isinstance(o, (np.float32, np.float64)):  # Handle numpy float types
            return float(o)
        else:
            return o

    with open(path, "w") as f:
        json.dump(convert_keys(obj), f, indent=2, default=str)


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def class_priors(y_int: np.ndarray, n_classes: int) -> list[float]:
    counts = np.bincount(y_int, minlength=n_classes)
    return (counts / counts.sum()).round(6).tolist()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run-tag", type=str, default="default", help="Run tag for this execution."
    )
    p.add_argument(
        "--mirror-latest",
        action="store_true",
        help="Copy this run's artifacts to top-level figures/results/shap/models",
    )
    args, unknown = p.parse_known_args()
    return args


def _per_sample_nll(y_int, proba):
    eps = 1e-15
    p = np.clip(proba[np.arange(len(y_int)), y_int], eps, 1 - eps)
    return -np.log(p)


def mirror_tree(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for root, _, files in os.walk(src):
        rel = Path(root).relative_to(src)
        out = dst / rel
        out.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(Path(root) / f, out / f)
