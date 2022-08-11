import torch
from torch.nn import Module, Linear, Tanh, Softmax, Softmin, Sigmoid, Identity, Sequential
from typing import Iterator
from .input_layer import LinearStatic


class OutputLayer(Module):

    def __init__(self, n: int, m: int, weights: torch.Tensor = None, trainable: bool = False, transform: str = None,
                 dtype: torch.dtype = torch.float64):

        super().__init__()

        # initialize output weights
        if trainable:
            layer = Linear(n, m, dtype=dtype)
        else:
            if weights is None:
                weights = torch.randn(m, n, dtype=dtype)
            elif weights.dtype != dtype:
                weights = torch.tensor(weights, dtype=dtype)
            layer = LinearStatic(weights)

        # define output function
        if transform is None:
            transform = Identity()
        elif transform == 'tanh':
            transform = Tanh()
        elif transform == 'softmax':
            transform = Softmax()
        elif transform == 'softmin':
            transform = Softmin()
        elif transform == 'sigmoid':
            transform = Sigmoid()

        # define output layer
        self.layer = Sequential(layer, transform)

    def forward(self, x):
        return self.layer(x)

    def parameters(self, recurse: bool = True) -> Iterator:
        return self.layer.parameters(recurse=recurse)
