import torch
from torch.nn import Module, Linear
from typing import Iterator
import numpy as np


class LinearStatic(Module):

    def __init__(self, weights: torch.Tensor):
        super().__init__()
        if not isinstance(weights, torch.Tensor):
            raise TypeError("Weights provided to the input layer have to be of type `torch.Tensor`.")
        weights.detach()
        self.weight = weights

    def forward(self, x):
        return self.weight @ x

    def parameters(self, recurse: bool = True) -> Iterator:
        for p in []:
            yield p


class InputLayer(Linear, LinearStatic):

    def __new__(cls, n: int, m: int, weights: np.ndarray = None, trainable: bool = False,
                 dtype: torch.dtype = torch.float64, **kwargs):
        if trainable:
            return cls._init_linear(m, n, dtype, kwargs.pop('bias', False))
        else:
            if weights is None:
                weights = torch.randn(n, m, dtype=dtype)
            else:
                weights = torch.tensor(weights, dtype=dtype)
            if weights.shape[0] == m and weights.shape[1] == n:
                weights = weights.T
            elif weights.shape[0] != n or weights.shape[1] != m:
                raise ValueError("Shape of the provided weights does not match the input and output dimensions of the"
                                 "layer.")
            return cls._init_static(weights)

    @staticmethod
    def _init_linear(m: int, n: int, dtype: torch.dtype, bias: bool):
        return Linear(m, n, bias=bias, dtype=dtype)

    @staticmethod
    def _init_static(weights: torch.Tensor):
        return LinearStatic(weights)

    # def forward(self, x):
    #     return self.layer(x)
    #
    # def parameters(self, recurse: bool = True) -> Iterator:
    #     return self.layer.parameters(recurse=recurse)

