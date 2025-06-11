"""
File containing the functionality for training an MLP
"""

import numpy as np
from net import MLP, SEED

def grad_descent(model: MLP, xs: np.ndarray, ys: np.ndarray, iters: int, epochs: int, 
                 batch_size: int, lr: float):
    """
    Function that performs gradient descent on a MLP object given input and output data
    """

    y_onehot = _one_hot(ys)
    generator = np.random.default_rng(SEED)
    for epoch in range(epochs):

        # Shuffle data once per epoch
        idcs = generator.permutation(len(xs))
        x_epoch = xs[idcs]
        y_epoch = y_onehot[idcs]

        # Do backprop for each batch
        for start in range(0, len(xs), batch_size):
            x_batch = x_epoch[start:(start + batch_size)]
            y_batch = y_epoch[start:(start + batch_size)]

            for _ in range(iters):
                # Forward and backward pass
                model.forward(x_batch)
                model.backward(y_batch)

                # Update values then zero gradients
                for w, b in zip(model.weights, model.biases):
                    # Do I need to zero grad??
                    w.increment(lr)
                    b.increment(lr)
        print(f"Epoch: {epoch + 1}/{epochs} done.")

def _one_hot(y: np.ndarray):
    """
    One Hot encode the vector y
    """
    d = len(np.unique(y))
    y_onehot = np.zeros(shape=(len(y), d))
    idxs = np.arange(start=0, stop=len(y), step=1)
    y_onehot[idxs, y] = 1

    return y_onehot