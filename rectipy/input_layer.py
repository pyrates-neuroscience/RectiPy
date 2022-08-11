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
            self.layer = LinearStatic(weights)

    def forward(self, x):
        return self.layer(x)

    def parameters(self, recurse: bool = True) -> Iterator:
        return self.layer.parameters(recurse=recurse)


class LinearStatic(Module):

    def __init__(self, weights: torch.Tensor):
        super().__init__()
        weights.detach()
        self.weights = weights

    def forward(self, x):
        return self.weights @ x

    def parameters(self, recurse: bool = True) -> Iterator:
        for p in []:
            yield p
