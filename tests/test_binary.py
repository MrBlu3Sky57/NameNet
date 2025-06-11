"""
Testing the model on XOR, AND and OR for overfitting
      --- This test was written by Chat GPT ---
"""

import numpy as np
from net.mlp import MLP
from net.train import grad_descent
from net import tanh, dtanh

def test_mlp_overfits_xor():
    # XOR input and labels
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])  # XOR

    # Build small model
    model = MLP(size=(2, 4, 2), sigma=tanh, dsigma=dtanh)

    # Train to overfit
    grad_descent(model, X, y, iters=10, epochs=200, batch_size=4, lr=0.1)

    # Predict
    model.forward(X)
    preds = np.argmax(model.layers[-1].value, axis=1)

    # Assert perfect fit
    assert np.array_equal(preds, y), f"Model failed to overfit XOR: got {preds}, expected {y}"

def test_mlp_overfits_and():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])  # AND

    model = MLP(size=(2, 4, 2), sigma=tanh, dsigma=dtanh)
    grad_descent(model, X, y, iters=10, epochs=200, batch_size=4, lr=0.1)

    model.forward(X)
    preds = np.argmax(model.layers[-1].value, axis=1)
    assert np.array_equal(preds, y), f"AND: got {preds}, expected {y}"

def test_mlp_overfits_or():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])  # OR

    model = MLP(size=(2, 4, 2), sigma=tanh, dsigma=dtanh)
    grad_descent(model, X, y, iters=10, epochs=200, batch_size=4, lr=0.1)

    model.forward(X)
    preds = np.argmax(model.layers[-1].value, axis=1)
    assert np.array_equal(preds, y), f"OR: got {preds}, expected {y}"