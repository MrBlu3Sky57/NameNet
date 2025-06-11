"""
Init file of net module
"""

from net.tensor import Tensor
from net.mlp import MLP
from net.util import one_hot

# Deterministic testing
SEED = 42

__all__ = ["Tensor", "MLP", "one_hot", "SEED"]
