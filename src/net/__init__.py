"""
Init file of net module
"""

from net.tensor import Tensor
from net.util import SEED, one_hot, tanh, dtanh, cross_entropy


__all__ = ["Tensor", "SEED", "one_hot", "tanh", "dtanh", "cross_entropy"]
