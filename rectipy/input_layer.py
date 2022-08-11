import torch
from torch.nn import Module, Linear
from typing import Iterator


class InputLayer(Module):

    def __init__(self, n: int, m: int, weights: torch.Tensor = None, trainable: bool = False,
                 dtype: torch.dtype = torch.float64):
        super().__init__()
        if trainable:
            self.layer = Linear(m, n, bias=False, dtype=dtype)
        else:
            if weights is None:
                weights = torch.randn(n, m, dtype=dtype)
            elif weights.dtype != dtype:
                weights = torch.tensor(weights, dtype=dtype)
            if weights.shape[0] != n or weights.shape[1] != m:
                raise ValueError("Shape of the provided weights does not match the input and output dimensions of the"
                                 "input layer.")
            self.layer = LinearStatic(weights)

    def forward(self, x):
        return self.layer(x)

    def parameters(self, recurse: bool = True) -> Iterator:
        return self.layer.parameters(recurse=recurse)


class LinearStatic(Module):

    def __init__(self, weights: torch.Tensor):
        super().__init__()
        if not isinstance(weights, torch.Tensor):
            raise TypeError("Weights provided to the input layer have to be of type `torch.Tensor`.")
        weights.detach()
        self.weights = weights

    def forward(self, x):
        return self.weights @ x

    def parameters(self, recurse: bool = True) -> Iterator:
        for p in []:
            yield p
