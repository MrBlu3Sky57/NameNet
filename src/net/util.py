"""
Utility functions for project
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