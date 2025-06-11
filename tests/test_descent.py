"""
Testing grad descent with dummy dataset.
      --- This test was written by Chat GPT ---
"""

import numpy as np
from net.util import SEED, relu, drelu, one_hot, cross_entropy
from net import grad_descent, MLP


def test_grad_descent_reduces_loss():
    # Create dummy data: 3 classes, 2D input
    np.random.seed(SEED)
    X = np.random.randn(100, 2)
    y = np.random.randint(0, 3, size=100)
    y_onehot = one_hot(y)

    # Build model
    model = MLP(size=(2, 10, 3), sigma=relu, dsigma=drelu)

    # Initial loss
    model.forward(X)
    init_loss = np.mean(cross_entropy(model.layers[-1].value, y_onehot))

    # Train
    grad_descent(model, X, y, iters=1, epochs=5, batch_size=10, lr=0.1)

    # Final loss
    model.forward(X)
    final_loss = np.mean(cross_entropy(model.layers[-1].value, y_onehot))

    # Assertions
    assert final_loss < init_loss, "Loss did not decrease"
    assert not np.allclose(model.weights[1].value, 0), "Weights not updated"
    assert not np.allclose(model.biases[1].value, 0), "Biases not updated"
