import torch
from torch.nn import Module
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
        elif type(weights) is np.ndarray:
            weights = torch.tensor(weights, dtype=dtype)
        if weights.shape[0] == n_in and weights.shape[1] == n_out:
            weights = weights.T
        elif weights.shape[0] != n_out or weights.shape[1] != n_in:
            raise ValueError("Shape of the provided weights does not match the input and output dimensions of the"
                             "layer.")

        # set public attributes
        self.n_in = n_in
        self.n_out = n_out
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


class RLS(Linear):

    _tensors = ["weights", "P"]

    def __init__(self, n_in: int, n_out: int, weights: Union[np.ndarray, torch.Tensor] = None,
                 dtype: torch.dtype = torch.float64, beta: float = 1.0, alpha: float = 1.0, delta: float = 1.0):
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
        alpha
            Regularization parameter for the initial state of the state-error correlation matrix `P`.
        delta
            Additional scalar > 0 for weighting the updates of the weight matrix and the covariance matrix of

        References
        ----------

        .. [1] Principe et al. (2011) Kernel Adaptive Filtering: A Comprehensive Introduction. John Wiley & Sons.
        """

        # check inputs for correctness
        if beta > 1 or beta < 0:
            raise ValueError("Parameter beta should be a positive scalar between 0 and 1.")
        if alpha < 0:
            raise ValueError("Parameter alpha should be a positive scalar.")

        # set RLS-specific attributes
        self.beta = beta
        self.P = alpha * torch.eye(n_in, dtype=dtype)
        self.loss = 0.0

        # call super method
        super().__init__(n_in, n_out, weights=weights, dtype=dtype, detach=True)

    def update(self, x: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor) -> None:

        # calculate current error
        err = y - y_hat

        # calculate gain
        k = self.P @ x
        k /= self.beta + x @ k

        # update the error correlation matrix
        self.P.add_(-torch.outer(k, x @ self.P))
        self.P /= self.beta

        # update the weights
        self.weights.add_(torch.outer(err, self.P @ x))

        # update loss
        self.loss = torch.inner(err, err)
