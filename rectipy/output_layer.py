import torch
from torch.nn import Module, Linear, Tanh, Softmax, Softmin, Sigmoid, Identity, Sequential
from typing import Iterator
from .input_layer import LinearStatic


class OutputLayer(Module):

    def __init__(self, n: int, m: int, weights: torch.Tensor = None, trainable: bool = False,
                 activation_function: str = None, dtype: torch.dtype = torch.float64, **kwargs):

        super().__init__()

        # initialize output weights
        if trainable:
            layer = Linear(n, m, dtype=dtype, **kwargs)
        else:
            if weights is None:
                weights = torch.randn(m, n, dtype=dtype)
            elif weights.dtype != dtype:
                weights = torch.tensor(weights, dtype=dtype)
            if weights.shape[0] != m or weights.shape[1] != n:
                raise ValueError("Shape of the provided weights does not match the input and output dimensions of the"
                                 "output layer.")
            layer = LinearStatic(weights)

        # define output function
        if activation_function is None:
            activation_function = Identity()
        elif activation_function == 'tanh':
            activation_function = Tanh()
        elif activation_function == 'softmax':
            activation_function = Softmax()
        elif activation_function == 'softmin':
            activation_function = Softmin()
        elif activation_function == 'sigmoid':
            activation_function = Sigmoid()
        else:
            raise ValueError(f"Invalid keyword argument `activation_function`: {activation_function} is not a valid "
                             f"option. See the docstring for `Network.add_output_layer` for valid options.")

        # define output layer
        self.layer = Sequential(layer, activation_function)

    def forward(self, x):
        return self.layer(x)

    def parameters(self, recurse: bool = True) -> Iterator:
        return self.layer.parameters(recurse=recurse)
