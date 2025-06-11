"""
File containing the MLP class
"""

from net import Tensor
import numpy as np

class MLP():
    """
    Multilayer perceptron class that supports forward and backward pass
    operations
    """
    layers: list[Tensor]
    unact: list[Tensor]
    weights: list[Tensor]
    biases: list[Tensor]
    sigma: callable

    def __init__(self, size: tuple, sigma):
        """ 
        Create an empty network of the required size
        assuming len(size) > 1
        """
        self.sigma = sigma
        self.layers = [Tensor(None)]
        self.unact = [Tensor(np.zeros(size[0]))]
        self.weights = [Tensor(None)]
        self.biases = [Tensor(None)]
        for l1, l2 in zip(size, size[1:]):
            self.layers.append(Tensor(None))
            self.unact.append(Tensor(None))
            self.weights.append(Tensor(np.random.randn(l2, l1)))
            self.biases.append(Tensor(np.random.randn(1, l2))) # Could lead to a bug!!
    
    def forward(self, xs: np.ndarray):
        """
        Apply a forward pass through the MLP given the input data, assuming
        dimensions correspond correctly
        """
        self.layers[0].value = xs
        for i in range(1, len(self.layers)):
            self.unact[i].value = self.layers[i - 1].value @ self.weights[i].value.T + self.biases[i].value
            self.layers[i].value = self.sigma(self.unact[i].value)
    
    def backward(self, dl: callable):
        """
        Run a backward pass on the network
        """
        pass
    
    def _soft_max(self, x: np.ndarray):
        """ 
        If the array is of dimension 1 apply elementwise soft max
        otherwise apply row wise.
        """
        if len(x.shape) == 0:
            raise ValueError
        if len(x.shape) == 1:
            x = x - np.max(x) # Stability
            logits = np.exp(x)
            return logits / np.sum(logits)
        if len(x.shape) > 1:
            x = x - np.max(x, axis=1) # Stability
            logits = np.exp(x)
            return logits / np.sum(logits, axis=1)
        