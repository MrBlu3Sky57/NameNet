"""
File containing the functionality for training an MLP
"""

import numpy as np
from net.util import SEED, one_hot
from net.model.mlp import MLP

def grad_descent(model: MLP, xs: np.ndarray, ys: np.ndarray, iters: int, epochs: int,
                 batch_size: int, lr: float):
    """
    Function that performs gradient descent on a MLP object given input and output data
    """

    y_onehot = one_hot(ys)
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

                # Update values
                for w, b in zip(model.weights[1:], model.biases[1:]):
                    # Do I need to zero grad??
                    w.increment(lr)
                    b.increment(lr)
                if model.emb.value is not None:
                    model.emb.increment(lr)
                    model.emb.zero_grad()
