"""
File containing the MLP class
"""

import numpy as np
from net.tensor import Tensor
from net.util import soft_max

class MLP():
    """
    Multilayer perceptron class that supports forward and backward pass
    operations using cross entropy loss and soft max output
    """
    layers: list[Tensor]
    ins: np.ndarray
    unact: list[Tensor]
    weights: list[Tensor]
    biases: list[Tensor]
    emb: Tensor
    sigma: callable
    dsigma: callable

    def __init__(self, size: tuple, sigma, dsigma, emb: np.ndarray = None):
        """ 
        Create an empty network of the required size
        assuming len(size) > 1 which can support embeddings.
        Size must have the right dimensions based on data shape
        """
        self.ins = None
        self.sigma = sigma
        self.dsigma = dsigma
        self.layers = [Tensor(None)]
        self.unact = [Tensor(np.zeros(size[0]))]
        self.weights = [Tensor(None)]
        self.biases = [Tensor(None)]
        for l1, l2 in zip(size, size[1:]):
            self.layers.append(Tensor(None))
            self.unact.append(Tensor(None))
            scale = np.sqrt(2.0 / l1)  # He initialization for ReLU, or use 1/l1 for tanh
            self.weights.append(Tensor(np.random.randn(l2, l1) * scale))
            self.biases.append(Tensor(np.zeros((1, l2))))  # Always 2D row vector
        self.emb = Tensor(emb)

    def forward(self, xs: np.ndarray):
        """
        Apply a forward pass through the MLP given the input data, assuming
        dimensions correspond correctly
        """
        if self.emb.value is None:
            self.layers[0].value = xs
            self.ins = xs
        else:
            self.ins = xs
            self.layers[0].value = self._embed(xs)
        for i in range(1, len(self.layers)):

            self.unact[i].value = self.layers[i - 1].value @ self.weights[i].value.T + self.biases[i].value
            if i != len(self.layers) - 1:
                self.layers[i].value = self.sigma(self.unact[i].value)
        self.layers[-1].value = soft_max(self.unact[-1].value)

    def backward(self, y_onehot):
        """
        Run a backward pass on the network (Assuming network gets matrix inputs)
        """
        self.unact[-1].grad = self.layers[-1].value - y_onehot # (n, l_-1)
        for i in range(1, len(self.layers)):
            self.weights[-i].grad = self.unact[-i].grad.T @ self.layers[-(i+1)].value # Sum across batch
            self.biases[-i].grad = np.sum(self.unact[-i].grad, axis=0) # Sum across batch
            self.layers[-(i + 1)].grad = self.unact[-i].grad @ self.weights[-i].value # Get total grad for each example in batch (n, l_-(i+1))
            self.unact[-(i + 1)].grad = self.layers[-(i+1)].grad * self.dsigma(self.unact[-(i + 1)].value) # (n, l_-(i+1))
        if self.emb.value is not None:
            # self.layers[0].grad has (n, c * d) --> need to unbind --> (n, c, d) --> (n * c, d) want to index into emb with n * c grads and update
            # ins, --> (n, c) --> need to unbind --> (n * c)
            d = self.emb.grad.shape[1]
            c = self.layers[0].value.shape[1] // d
            n = self.layers[0].value.shape[0]
            np.add.at(self.emb.grad, np.reshape(self.ins, shape=(n * c)), np.reshape(self.layers[0].grad, shape=(n * c, d)))

    def _embed(self, xs: np.ndarray):
        """
        Embed inputs into feature space
        """
        embs = self.emb.value[xs]
        if len(xs.shape) == 1:
            return np.reshape(embs, shape=(embs.shape[0] * embs.shape[1]))
        else:
            return np.reshape(embs, shape=(embs.shape[0], -1))
