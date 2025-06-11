"""
File containing the MLP class
"""

import numpy as np
from net import Tensor

class MLP():
    """
    Multilayer perceptron class that supports forward and backward pass
    operations using cross entropy loss and soft max output
    """
    layers: list[Tensor]
    unact: list[Tensor]
    weights: list[Tensor]
    biases: list[Tensor]
    sigma: callable
    dsigma: callable

    def __init__(self, size: tuple, sigma, dsigma):
        """ 
        Create an empty network of the required size
        assuming len(size) > 1
        """
        self.sigma = sigma
        self.dsigma = dsigma
        self.layers = [Tensor(None)]
        self.unact = [Tensor(np.zeros(size[0]))]
        self.weights = [Tensor(None)]
        self.biases = [Tensor(None)]
        for l1, l2 in zip(size, size[1:]):
            self.layers.append(Tensor(None))
            self.unact.append(Tensor(None))
            self.weights.append(Tensor(np.random.randn(l2, l1)))
            self.biases.append(Tensor(np.random.randn(1, l2)))  # Always 2D row vector

    def forward(self, xs: np.ndarray):
        """
        Apply a forward pass through the MLP given the input data, assuming
        dimensions correspond correctly
        """
        self.layers[0].value = xs
        for i in range(1, len(self.layers)):

            self.unact[i].value = self.layers[i - 1].value @ self.weights[i].value.T + self.biases[i].value
            if i != len(self.layers) - 1:
                self.layers[i].value = self.sigma(self.unact[i].value)
        self.layers[-1].value = self._soft_max(self.unact[-1].value)

    def backward(self, y_onehot):
        """
        Run a backward pass on the network (Assuming network gets matrix inputs)
        """
        self.unact[-1].grad = self.layers[-1].value - y_onehot
        for i in range(1, len(self.layers)):
            self.weights[-i].grad = self.unact[-i].grad.T @ self.layers[-(i+1)].value # Sum across batch
            self.biases[-i].grad = np.sum(self.unact[-i].grad, axis=0) # Sum across batch
            self.layers[-(i + 1)].grad = self.unact[-i].grad @ self.weights[-i].value # Get total grad for each example in batch
            self.unact[-(i + 1)].grad = self.layers[-(i+1)].grad * self.dsigma(self.unact[-(i + 1)].value)
        return

    @staticmethod
    def _soft_max(x: np.ndarray):
        """ 
        If the array is of dimension 1 apply elementwise soft max
        otherwise apply row wise.
        """
        if len(x.shape) == 0:
            return None
        if len(x.shape) == 1:
            x = x - np.max(x) # Stability
            logits = np.exp(x)
            return logits / np.sum(logits)
        if len(x.shape) > 1:
            x = x - np.max(x, axis=1) # Stability
            logits = np.exp(x)
            return logits / np.sum(logits, axis=1)
