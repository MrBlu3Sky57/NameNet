"""
Testing backprop with comparison to finite differences.
      --- This test was written by Chat GPT ---
"""

import numpy as np
from net.util import SEED, tanh, dtanh, cross_entropy
from net import MLP

# Numerical gradient checker
def numerical_grad(loss_fn, param, eps=1e-5):
    grad = np.zeros_like(param)
    for i in range(param.size):
        orig = param.flat[i]

        param.flat[i] = orig + eps
        loss_plus = loss_fn()

        param.flat[i] = orig - eps
        loss_minus = loss_fn()

        param.flat[i] = orig  # restore
        grad.flat[i] = ((loss_plus - loss_minus) / (2 * eps)).item() # Unbox array

    return grad

def test_mlp_backprop_matches_numerical():
    # Set seed for reproducibility
    np.random.seed(SEED)

    # Build small network
    net = MLP((2, 3, 2), tanh, dtanh)

    # Fix weights and biases for reproducibility
    for W in net.weights[1:]:
        W.value = np.random.randn(*W.value.shape)
    for b in net.biases[1:]:
        b.value = np.random.randn(*b.value.shape)

    # Single input and one-hot label
    x = np.array([[0.5, -0.3]])
    y = np.array([[0, 1]])

    def compute_loss():
        net.forward(x)
        return cross_entropy(net.layers[-1].value, y)

    # Run forward and backward to get analytical grad
    net.forward(x)
    net.backward(y)

    # Pick a specific layer to check (output layer)
    i = -1
    analytic_grad = net.weights[i].grad.copy()

    def loss_with_modified_weights():
        return compute_loss()

    # Numerical gradient
    numeric_grad = numerical_grad(loss_with_modified_weights, net.weights[i].value)

    # Compare (relative error)
    rel_error = np.linalg.norm(analytic_grad - numeric_grad) / (np.linalg.norm(analytic_grad) + np.linalg.norm(numeric_grad) + 1e-8)

    print("Analytic grad:\n", analytic_grad)
    print("Numeric grad:\n", numeric_grad)
    print("Relative error:", rel_error)

    assert rel_error < 1e-5, f"Backprop gradient differs from numerical gradient (rel error {rel_error})"
