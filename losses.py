import numpy as np

def cross_entropy(y_true, y_pred, sample_weight=1.0):
    """
    Categorical cross-entropy for one-hot BCI class labels.

    y_true and y_pred are column vectors shaped (n_classes, 1).
    """
    y_pred = np.clip(y_pred, 1e-8, 1.0)
    return -sample_weight * np.sum(y_true * np.log(y_pred))

def cross_entropy_prime(y_true, y_pred, sample_weight=1.0):
    """
    Gradient for softmax + categorical cross-entropy.

    This is the gradient with respect to the logits before softmax, so the
    Softmax layer backward pass should pass it through unchanged.
    """
    return sample_weight * (y_pred - y_true)
