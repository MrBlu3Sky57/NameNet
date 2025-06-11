"""
File containing the functionality for training an MLP
"""

from net import *
import numpy as np

def grad_descent(model: MLP, xs: np.ndarray, y_onehot: np.ndarray, iters: int, epochs: int, batch_size: int, lr: float):
    generator = np.random.default_rng(SEED)
    for epoch in epochs:
        for batch in batch_size:
            pass