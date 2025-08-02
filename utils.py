from sklearn.base import ClassifierMixin


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
