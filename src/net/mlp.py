"""
File containing the MLP class
"""

import numpy as np
from net import Tensor

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
            self.biases.append(Tensor(np.random.randn(l2))) # Could lead to a bug!!

    def forward(self, xs: np.ndarray):
        """
        Apply a forward pass through the MLP given the input data, assuming
        dimensions correspond correctly
        """
        self.layers[0].value = xs
        for i in range(1, len(self.layers)):
            if len(xs.shape) == 1:
                self.biases.value = np.reshape(self.biases[i].value, shape=len(self.biases[i].value))
                self.unact[i].value = self.layers[i - 1].value @ self.weights[i].value.T + self.biases[i].value
            else:
                self.biases.value = np.reshape(self.biases[i].value, shape=(1, len(self.biases[i].value)))
                self.unact[i].value = self.layers[i - 1].value @ self.weights[i].value.T + self.biases[i].value
                if i != len(self.layers) - 1:
                    self.layers[i].value = self.sigma(self.unact[i].value)
        self.layers[-1].value = self._soft_max(self.layers[-1].value)

    def backward(self, dl: callable):
        """
        Run a backward pass on the network (Assuming network gets matrix inputs)
        """
        self.layers[-1].grad = dl(self.layers[-1].value)
        self.unact[-1].grad = np.sum(self._deriv_soft_max(self.unact[-1].value) * self.layers[-1].grad, axis=0, keepdims=True)
        for i in range(1, len(self.layers)):
            self.weights[-i].grad = np.sum(self.layers[-(i+1)].value, axis=0, keepdims=True).T * self.unact[-(i+1)].grad
            self.biases[-i].grad = (self.unact[-1].grad).flatten()
            self.layers[-(i + 1)].grad = np.sum(self.weights[-(i + 1)].T * self.unact[-1].grad, axis=1, keepdims=True).T
            self.unact[-(i + 1)].grad = self.dsigma(self.layers[-i].grad[-(i + 1)])
        return

    @staticmethod
    def _soft_max(x: np.ndarray):
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
    
    @staticmethod
    def _deriv_soft_max(x: np.ndarray):
        """
        Softmax derivative
        """
        return x * (1 - x)