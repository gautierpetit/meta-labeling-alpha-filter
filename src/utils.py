"""
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

Small project utilities: I/O, ids, and lightweight numeric helpers.

This module intentionally contains compact helper functions used across
the repository: mapping classifier classes, building rolling-window
splits, small serialisation helpers and a couple of numerical helpers
used by analysis and modeling code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin


def get_class_to_index(clf: ClassifierMixin) -> dict[Any, int]:
    """Return a mapping from class label -> integer index for a classifier.

    The function supports two common attribute conventions used by the
    project's wrappers: ``class_labels_`` (custom wrappers) and
    ``classes_`` (scikit-learn estimators).
    """
    if hasattr(clf, "class_labels_"):
        return {label: i for i, label in enumerate(clf.class_labels_)}
    if hasattr(clf, "classes_"):
        return {cls: i for i, cls in enumerate(clf.classes_)}
    raise ValueError("Classifier must define `class_labels_` or `classes_`.")


def _rolling_windows(n: int, n_splits: int = 3, embargo: int = 5) -> list[tuple[int, int, int]]:
    """Create chronologically ordered calibration/prediction windows.

    Returns a list of (cal_end, pred_start, pred_end) integer triplets
    suitable for time-series rolling calibration (Fold-2 style) usage.
    """
    assert n_splits >= 2
    cuts = [0] + [int(round(n * i / n_splits)) for i in range(1, n_splits)] + [n]
    windows: list[tuple[int, int, int]] = []
    for j in range(1, len(cuts) - 1):
        cal_end = cuts[j]
        pred_start = min(cal_end + max(embargo, 0), n)
        pred_end = cuts[j + 1]
        if pred_end > pred_start:
            windows.append((cal_end, pred_start, pred_end))
    return windows


def _to_3d_shap(sv: list[np.ndarray] | np.ndarray) -> np.ndarray:
    """Normalize SHAP arrays to a (n_samples, n_features, n_classes) array.

    Many SHAP APIs return either a 2D array (n_samples, n_features) for
    binary tasks or a list of arrays per class. This helper normalizes
    both shapes into a 3D array where the last axis indexes classes.
    """
    if isinstance(sv, list):
        return np.stack(sv, axis=2)
    sv = np.asarray(sv)
    if sv.ndim == 2:
        return sv[..., None]
    return sv


def make_run_id(tag: str | None = None) -> str:
    """Return a short run id using timestamp and optional tag.

    Examples
    --------
    >>> make_run_id('exp1')
    '20250817_123000_exp1'
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{tag}" if tag else ts


def md5_columns(df: pd.DataFrame) -> str:
    """Return MD5 of dataframe column names (stable string representation)."""
    s = ",".join(map(str, df.columns))
    return hashlib.md5(s.encode()).hexdigest()


def safe_git_sha() -> str:
    """Return the short git sha for the current repo or 'unknown'."""
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def write_json(path: str | Path, obj: Any) -> None:
    """Write an object to JSON, converting numpy scalars and keys to strings.

    The function intentionally keeps a permissive ``default=str`` fallback
    to make it easy to persist small metadata objects produced during runs.
    """

    def convert_keys(o: Any) -> Any:
        if isinstance(o, dict):
            return {str(k): convert_keys(v) for k, v in o.items()}
        if isinstance(o, list):
            return [convert_keys(i) for i in o]
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        return o

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf8") as f:
        json.dump(convert_keys(obj), f, indent=2, default=str)


def read_json(path: str | Path) -> Any:
    """Read JSON from `path` and return the deserialized object."""
    with open(path, encoding="utf8") as f:
        return json.load(f)


def class_priors(y_int: np.ndarray, n_classes: int) -> list[float]:
    """Return class prior probabilities from integer-encoded labels."""
    counts = np.bincount(y_int, minlength=n_classes)
    return (counts / counts.sum()).round(6).tolist()


def parse_args() -> argparse.Namespace:
    """Parse a minimal set of CLI flags used by the project's scripts."""
    p = argparse.ArgumentParser()
    p.add_argument("--run-tag", type=str, default="default", help="Run tag for this execution.")
    p.add_argument(
        "--mirror-latest",
        action="store_true",
        help="Copy this run's artifacts to top-level figures/results/shap/models",
    )
    args, _unknown = p.parse_known_args()
    return args


def _per_sample_nll(y_int: np.ndarray, proba: np.ndarray) -> np.ndarray:
    """Return per-sample negative log-likelihood for integer labels and proba array."""
    eps = 1e-15
    p = np.clip(proba[np.arange(len(y_int)), y_int], eps, 1 - eps)
    return -np.log(p)


def mirror_tree(src: Path, dst: Path) -> None:
    """Recursively copy files from `src` to `dst`, creating directories as needed."""
    dst.mkdir(parents=True, exist_ok=True)
    for root, _dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        out = dst / rel
        out.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(Path(root) / f, out / f)


def index_fingerprint(idx) -> dict:
    """Compact, deterministic summary for Index or MultiIndex."""
    # Flatten to strings
    if getattr(idx, "nlevels", 1) > 1:
        ser = idx.map(lambda t: "|".join(map(str, t)))
    else:
        ser = idx.map(str)
    # Hash (sample + full length)
    head = ser[:3].tolist()
    tail = ser[-3:].tolist()
    sample = "|".join(ser[:10].tolist() + ser[-10:].tolist())
    h = hashlib.blake2b(sample.encode("utf-8"), digest_size=12).hexdigest()
    return {
        "length": int(len(idx)),
        "head": head,
        "tail": tail,
        "sample_hash_blake2b": h,
    }


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
        }
        # optional context via LoggerAdapter
        for k in ("run_id", "git_sha"):
            val = getattr(record, k, None)
            if val is not None:
                payload[k] = val
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_json_logging(
    run_dir: Path, run_id: str | None = None, git_sha: str | None = None
) -> logging.LoggerAdapter:
    run_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Console
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        root.addHandler(ch)

    # JSON file (rotating)
    fh = RotatingFileHandler(
        run_dir / "run.log.jsonl", maxBytes=10_000_000, backupCount=3, encoding="utf-8"
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(JsonFormatter())
    root.addHandler(fh)

    # Attach context (run_id, git_sha)
    return logging.LoggerAdapter(root, extra={"run_id": run_id, "git_sha": git_sha})
