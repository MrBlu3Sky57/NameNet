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

                # Update values then zero gradients
                for w, b in zip(model.weights[1:], model.biases[1:]):
                    # Do I need to zero grad??
                    w.increment(lr)
                    b.increment(lr)
                if model.emb is not None:
                    # Construct embedding gradients on the fly -- Seems a bit clunky right now
                    emb_grad = np.zeros_like(model.emb) # (v, d) -- v is vocab size, d is feature size
                    d = emb_grad.shape[1]
                    c = model.layers[0].value.shape[1] // d
                    n = model.layers[0].value.shape[0]
                    np.add.at(emb_grad, np.reshape(xs, n * c), np.reshape(model.layers[0].grad, n * c, d))
                    
                    # Layers.grad has (n, c * d) --> need to unbind --> (n, c, d) --> (n * c, d) want to index into emb with n * c grads and update
                    # Xs, --> (n, c) --> need to unbind --> (n * c)


                    model.emb -= lr * emb_grad
