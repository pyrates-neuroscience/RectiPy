import torch
from torch.nn import Module, Sequential, Tanh, Softmax, Softmin, Sigmoid, Identity
from typing import Iterator, Union
import numpy as np
from .utility import to_device


class Linear(Module):

    _tensors = ["weights"]

    def __init__(self, n_in: int, n_out: int, weights: Union[np.ndarray, torch.Tensor] = None,
                 dtype: torch.dtype = torch.float64, detach: bool = True, **kwargs):

        super().__init__()

        # finalize layer weights
        if weights is None:
            weights = torch.randn(n_out, n_in, dtype=dtype)
        else:
            weights = torch.tensor(weights, dtype=dtype)
        if weights.shape[0] == n_in and weights.shape[1] == n_out:
            weights = weights.T
        elif weights.shape[0] != n_out or weights.shape[1] != n_in:
            raise ValueError("Shape of the provided weights does not match the input and output dimensions of the"
                             "layer.")
        self.weights = weights

        # handle tensor gradient requirements
        self.train_params = []
        if detach:
            self.detach()
        else:
            train_params = kwargs.pop("train_params", self._tensors)
            for key in self._tensors:
                val = getattr(self, key)
                if key in train_params:
                    val.requires_grad = True
                    self.train_params.append(val)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.weights @ x

    def parameters(self, recurse: bool = True) -> Iterator:
        for p in self.train_params:
            yield p

    def to(self, device: str, **kwargs):
        super().to(device=torch.device(device), **kwargs)
        for attr in self._tensors:
            val = getattr(self, attr)
            setattr(self, attr, to_device(val, device=device))
        return self

    def detach(self):
        for key in self._tensors:
            val = getattr(self, key)
            val.detach()


class RLSLayer(Linear):

    _tensors = ["weights", "P"]

    def __init__(self, n_in: int, n_out: int, weights: Union[np.ndarray, torch.Tensor] = None,
                 dtype: torch.dtype = torch.float64, beta: float = 1.0, delta: float = 1.0):
        """General form of the extended recursive least-squares algorithm as described in [1]_. Can be used to implement
        readout weight learning as in

        Parameters
        ----------
        n_in
            Input dimensionality of the layer.
        n_out
            Output dimensionality of the layer.
        weights
            2D array (m x n) with initial set of weights.
        dtype
            Data type for all tensors.
        beta
            Forgetting rate with 0 < beta <= 1. The smaller beta is, the more importance is given to most recent
            observations over past observations.
        delta
            Regularization parameter for the initial state of the state-error correlation matrix `P`.

        References
        ----------

        .. [1] Principe et al. (2011) Kernel Adaptive Filtering: A Comprehensive Introduction. John Wiley & Sons.
        """

        # check inputs for correctness
        if beta > 1 or beta < 0:
            raise ValueError("Parameter beta should be a positive scalar between 0 and 1.")
        if delta < 0:
            raise ValueError("Parameter delta should be a positive scalar.")

        # set RLS-specific attributes
        self.delta = delta
        self.beta_inv = 1.0 / beta
        self.P = delta * torch.eye(n_in, dtype=dtype)
        self.loss = 0

        # call super method
        super().__init__(n_in, n_out, weights=weights, dtype=dtype, detach=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:

        # predict target vector y
        y_pred = super().forward(x)
        if y is None:
            return y_pred

        # calculate current error
        err = y - y_pred

        # calculate gain
        k = self.P @ x * self.beta_inv
        k /= 1.0 + x @ k

        # update the weights
        self.weights.add_(err * k)

        # update the error correlation matrix
        self.P.add_(-(torch.outer(k, x @ self.P)))
        self.P.mul_(self.beta_inv)

        # update loss
        self.loss = torch.inner(err, err)
        return y_pred


class LayerStack(Sequential):

    def __init__(self, layer: Linear, activation_function: str = None):

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

        # call super class
        super().__init__(layer, activation_function)

    def to(self, device: str, **kwargs):
        super().to(device=torch.device(device), **kwargs)
        for layer in self:
            layer.to(device)
        return self

    def parameters(self, recurse: bool = True) -> Iterator:
        for layer in self:
            for p in layer.parameters(recurse):
                yield p