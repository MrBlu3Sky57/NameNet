"""
Init file of net module
"""

from net.tensor import Tensor
from net.util import SEED, one_hot, tanh, dtanh, relu, drelu, soft_max, cross_entropy


__all__ = ["Tensor", "SEED", "one_hot", "tanh", "dtanh", "relu", "drelu", "soft_max", "cross_entropy"]
