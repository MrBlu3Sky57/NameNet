"""
Init file of net module
"""

from net.tensor import Tensor
from net.mlp import MLP
from net.train import grad_descent
from net.util import one_hot, tanh, dtanh, cross_entropy

# Deterministic testing
SEED = 42

__all__ = ["Tensor", "MLP", "SEED", "grad_descent", "one_hot", "tanh", "dtanh", "cross_entropy"]
