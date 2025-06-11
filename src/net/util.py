"""
Module containing utility functions
"""

import numpy as np

def one_hot(y: np.ndarray):
    """
    One Hot encode the vector y
    """
    d = len(np.unique(y))
    y_onehot = np.zeros(shape=(len(y), d))
    idxs = np.arange(start=0, stop=len(y), step=1)
    y_onehot[idxs, y] = 1

    return y_onehot

def tanh(x: np.ndarray):
    """
    Tanh function
    """
    return np.tanh(x)

def dtanh(x: np.ndarray):
    """
    Tanh derivative
    """
    return 1 - np.tanh(x) ** 2

def cross_entropy(pred: np.ndarray, target: np.ndarray):
    """
    Cross entropy loss
    """
    if pred.shape == 0:
        return None
    elif pred.shape == 1:
        return -np.sum(target * np.log(pred + 1e-8))
    else:
        return -np.sum(target * np.log(pred + 1e-8), axis=1)