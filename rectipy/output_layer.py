import torch
from torch.nn import Module, Tanh, Softmax, Softmin, Sigmoid, Identity, Sequential
from .input_layer import InputLayer
import numpy as np


class OutputLayer(Sequential):

    def __new__(cls, n: int, m: int, weights: np.ndarray = None, trainable: bool = False,
                activation_function: str = None, dtype: torch.dtype = torch.float64, **kwargs):

        # initialize linear layer with weights
        layer = InputLayer(m, n, weights=weights, trainable=trainable, dtype=dtype, bias=kwargs.pop('bias', True))

        # define output function
        if activation_function is None:
            activation_function = Identity()
        elif activation_function == 'tanh':
            activation_function = Tanh()
        elif activation_function == 'softmax':
            activation_function = Softmax(dim=0)
        elif activation_function == 'softmin':
            activation_function = Softmin(dim=0)
        elif activation_function == 'sigmoid':
            activation_function = Sigmoid()
        else:
            raise ValueError(f"Invalid keyword argument `activation_function`: {activation_function} is not a valid "
                             f"option. See the docstring for `Network.add_output_layer` for valid options.")

        # define output layer
        return cls._init_sequential(layer, activation_function)

    @staticmethod
    def _init_sequential(layer: InputLayer, func: Module):
        return Sequential(layer, func)
