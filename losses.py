import numpy as np

def cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred + 1e-8))
def cross_entropy_prime(y_true, y_pred):
    return -y_true / (y_pred + 1e-8) / np.size(y_true)

