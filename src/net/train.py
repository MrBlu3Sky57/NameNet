"""
File containing the functionality for training an MLP
"""

import numpy as np
from net.util import SEED, one_hot, clip_grad, cross_entropy
from net.model.mlp import MLP

def _grad_descent(model: MLP, xs: np.ndarray, ys: np.ndarray, iters: int, epochs: int,
                 batch_size: int, lr: float, l2_lambda: float = None):
    """
    Function that performs gradient descent on a MLP object given input and output data
    -- Old version --
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
                    clip_grad(w, 5.0)
                    clip_grad(b, 5.0)

                    # L2 regularization
                    if l2_lambda is not None:
                        w.grad += l2_lambda * w.value
                    w.increment(lr)
                    b.increment(lr)
                if model.emb.value is not None:
                    clip_grad(model.emb, 5.0)
                    model.emb.grad += l2_lambda * model.emb.value
                    model.emb.increment(lr)
                    model.emb.zero_grad()
        print(f"epoch: {epoch + 1}/{epochs} done.")

def grad_descent(model: MLP, xs: np.ndarray, ys: np.ndarray, steps: int, batch_size: int, lr_start: float, l2_lambda: float=0):
    """
    Function that performs gradient descent on MLP object
    """
    y_onehot = one_hot(ys)
    generator = np.random.default_rng(SEED)
    
    for step in range(steps):
        ix = generator.integers(0, len(xs), batch_size)
        x_batch, y_batch =  xs[ix], y_onehot[ix]

        model.forward(x_batch)
        model.backward(y_batch)

        lr = lr_start if step < steps // 2 else lr_start * 0.1

        # Update values
        for w, b in zip(model.weights[1:], model.biases[1:]):
            clip_grad(w, 5.0)
            clip_grad(b, 5.0)

            # L2 regularization
            w.grad += l2_lambda * w.value
            
            w.increment(lr)
            b.increment(lr)

            # Update emb
            if model.emb.value is not None:
                clip_grad(model.emb, 5.0)
                model.emb.grad += l2_lambda * model.emb.value
                model.emb.increment(lr)
                model.emb.zero_grad()

        if step % 1000 == 0:
            probs = model.layers[-1].value
            loss = np.mean(cross_entropy(probs, y_batch))
            print(f"step {step} | loss: {loss:.4f}")

