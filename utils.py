from sklearn.base import ClassifierMixin
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


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



def compute_balanced_weights(y_int):
    classes = np.unique(y_int)
    w = compute_class_weight(class_weight="balanced", classes=classes, y=y_int)
    return {int(c): float(wi) for c, wi in zip(classes, w)}