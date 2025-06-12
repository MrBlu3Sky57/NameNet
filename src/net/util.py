"""
Module containing utility functions
"""

import numpy as np
from .tensor import Tensor

# Deterministic testing
SEED = 42

def one_hot(y: np.ndarray):
    """
    One Hot encode the vector y
    """
    d = np.max((np.unique(y))) + 1
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

# A caution -- The model fails for XOR with ReLU activation (likely due to dead neurons)
# Might need to adjust initializations based on training data
def relu(x: np.ndarray):
    """
    ReLU function
    """
    return np.maximum(0, x)

def drelu(x: np.ndarray):
    """
    ReLU Derivative
    """
    return (x > 0).astype(float)

def soft_max(x: np.ndarray, temperature: float = 1.0):
    """ 
    If the array is of dimension 1 apply elementwise soft max
    otherwise apply row wise.
    """
    if len(x.shape) == 0:
        return None
    if len(x.shape) == 1:
        x = x - np.max(x) # Stability
        logits = np.exp(x / temperature)
        return logits / np.sum(logits)
    if len(x.shape) > 1:
        x = x - np.max(x, axis=1, keepdims=True) # Stability
        logits = np.exp(x / temperature)
        return logits / np.sum(logits, axis=1, keepdims=True)

def cross_entropy(pred: np.ndarray, target: np.ndarray):
    """
    Cross entropy loss
    """
    if len(pred.shape) == 0:
        return None
    elif len(pred.shape) == 1:
        return -np.sum(target * np.log(pred + 1e-8)) / pred.shape[0]
    else:
        return -np.sum(target * np.log(pred + 1e-8), axis=1) / pred.shape[0]

def clip_grad(tensor: Tensor, max_norm: float):
    """ Clip grad """
    grad_norm = np.linalg.norm(tensor.grad)
    if grad_norm > max_norm:
        tensor.grad *= (max_norm / grad_norm)
