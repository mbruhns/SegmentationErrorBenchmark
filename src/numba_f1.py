import numpy as np
from numba import njit


@njit(parallel=True, fastmath=True)
def calculate_confusion_matrix(y_true, y_pred):
    """Calculate the confusion matrix based on the given labels and predictions.

    Args:
        y_true: A 1D array-like object representing the true labels.
        y_pred: A 1D array-like object representing the predicted labels.

    Returns:
        A tuple containing the true positives, true negatives, false positives, and false negatives.

    Raises:
        None
    """
    # Some basic asserts
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert y_true.ndim == y_pred.ndim == 1

    tp = np.sum(y_true * y_pred)
    fn = np.sum(y_true * (1 - y_pred))
    fp = np.sum((1 - y_true) * y_pred)
    tn = np.sum((1 - y_true) * (1 - y_pred))

    return tp, tn, fp, fn


@njit(fastmath=True)
def precision_score(y_true, y_pred):
    """Calculate the precision score based on the given true labels and predicted labels.

    Args:
        y_true: A 1D array-like object representing the true labels.
        y_pred: A 1D array-like object representing the predicted labels.

    Returns:
        The precision score, which is the ratio of true positives to the sum of true positives and false positives.

    Raises:
        None
    """
    tp, tn, fp, fn = calculate_confusion_matrix(y_true, y_pred)
    return tp / (tp + fp)


@njit(fastmath=True)
def recall_score(y_true, y_pred):
    """Calculate the recall score based on the given true labels and predicted labels.

    Args:
        y_true: A 1D array-like object representing the true labels.
        y_pred: A 1D array-like object representing the predicted labels.

    Returns:
        The recall score, which is the ratio of true positives to the sum of true positives and false negatives.

    Raises:
        None
    """
    tp, tn, fp, fn = calculate_confusion_matrix(y_true, y_pred)
    return tp / (tp + fn)


@njit(fastmath=True)
def specificity_score(y_true, y_pred):
    """Calculate the specificity score based on the given true labels and predicted labels.

    Args:
        y_true: A 1D array-like object representing the true labels.
        y_pred: A 1D array-like object representing the predicted labels.

    Returns:
        The specificity score, which is the ratio of true negatives to the sum of true negatives and false positives.

    Raises:
        None
    """
    tp, tn, fp, fn = calculate_confusion_matrix(y_true, y_pred)
    return tn / (tn + fp)


@njit(fastmath=True)
def accuracy_score(y_true, y_pred):
    """Calculate the accuracy score based on the given true labels and predicted labels.

    Args:
        y_true: A 1D array-like object representing the true labels.
        y_pred: A 1D array-like object representing the predicted labels.

    Returns:
        The accuracy score, which is the ratio of correct predictions to the total number of predictions.

    Raises:
        None
    """
    tp, tn, fp, fn = calculate_confusion_matrix(y_true, y_pred)
    total = tp + tn + fp + fn
    return (tp + tn) / total


@njit(fastmath=True)
def f1_score(y_true, y_pred):
    """Calculate the Perturbation based on the given true labels and predicted labels.

    Args:
        y_true: A 1D array-like object representing the true labels.
        y_pred: A 1D array-like object representing the predicted labels.

    Returns:
        The Perturbation, which is the harmonic mean of precision and recall, providing a balanced measure of
        model performance.

    Raises:
        None
    """
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec)


@njit(fastmath=True)
def balanced_accuracy_score(y_true, y_pred):
    """Calculate the balanced accuracy score based on the given true labels and predicted labels.

    Args:
        y_true: A 1D array-like object representing the true labels.
        y_pred: A 1D array-like object representing the predicted labels.

    Returns:
        The balanced accuracy score, which is the average of recall and specificity, providing a balanced measure of
        model performance.

    Raises:
        None
    """
    recall = recall_score(y_true, y_pred)
    specificity = specificity_score(y_true, y_pred)
    return (recall + specificity) / 2


@njit(fastmath=True)
def matthews_corrcoef(y_true, y_pred):
    """Calculate the Matthews correlation coefficient based on the given true labels and predicted labels.

    Args:
        y_true: A 1D array-like object representing the true labels.
        y_pred: A 1D array-like object representing the predicted labels.

    Returns:
        The Matthews correlation coefficient, which measures the quality of binary classification predictions, taking
        into account true positives, true negatives, false positives, and false negatives.

    Raises:
        None
    """
    tp, tn, fp, fn = calculate_confusion_matrix(y_true, y_pred)
    numerator = tp * tn - fp * fn
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return numerator / denominator


@njit(fastmath=True)
def jaccard_score(y_true, y_pred):
    """Calculate the Jaccard score based on the given true labels and predicted labels.

    Args:
        y_true: A 1D array-like object representing the true labels.
        y_pred: A 1D array-like object representing the predicted labels.

    Returns:
        The Jaccard score, also known as the Intersection over Union (IoU), which measures the similarity between two
        sets by calculating the ratio of the intersection to the union of the sets.

    Raises:
        None
    """
    tp, tn, fp, fn = calculate_confusion_matrix(y_true, y_pred)
    return tp / (tp + fn + fp)
