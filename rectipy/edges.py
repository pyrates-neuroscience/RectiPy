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
            raise ValueError("Shape of the provided weights does not match the input and output dimensions of the "
                             "source and target nodes.")

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


class LinearMemory(Linear):

    _tensors = ["weights", "buffer", "delays"]

    def __init__(self, n_in: int, n_out: int, delays: Union[np.ndarray, torch.Tensor],
                 weights: Union[np.ndarray, torch.Tensor] = None,
                 intrinsic_weights: Union[np.ndarray, torch.Tensor] = None, dtype: torch.dtype = torch.float64,
                 detach: bool = True, **kwargs):

        # set up delay tensor
        if type(delays) is np.ndarray:
            delays = torch.tensor(delays, dtype=torch.long)
        if len(delays) != n_in:
            raise ValueError("The number of delays must match the number of node inputs.")
        self.delays = delays

        # set up intrinsic buffer
        self.buffer = torch.zeros((n_in, delays.max()+1), dtype=dtype)

        # call super method
        train_params = kwargs.pop("train_params", ["weights"])
        super().__init__(n_in, n_out, weights=weights, dtype=dtype, detach=detach, train_params=train_params, **kwargs)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        self.buffer = self.buffer.roll(-1, 1)
        self.buffer[:, self.delays] = x
        return super().forward(self.buffer[:, 0])


class LinearFilter(Linear):

    _tensors = ["weights", "filter", "y"]

    def __init__(self, n_in: int, n_out: int, filter_weights: Union[np.ndarray, torch.Tensor],
                 weights: Union[np.ndarray, torch.Tensor] = None, dtype: torch.dtype = torch.float64,
                 detach: bool = True, **kwargs):

        # set up intrinsic weights
        if type(filter_weights) is np.ndarray:
            filter_weights = torch.tensor(filter_weights, dtype=dtype)
        if filter_weights.shape[0] != n_in or filter_weights.shape[1] != n_in:
            raise ValueError("Intrinsic weights have to be a square matrix with the number of rows and columns matching"
                             "the number of inputs to the edge.")
        self.filter = filter_weights
        self.y = torch.zeros(n_in, dtype=dtype)

        # call super method
        train_params = kwargs.pop("train_params", ["weights", "filter"])
        super().__init__(n_in, n_out, weights=weights, dtype=dtype, detach=detach, train_params=train_params, **kwargs)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        self.y = self.filter @ self.y + x
        return super().forward(self.y)


class LinearMemoryFilter(LinearMemory):

    _tensors = ["weights", "buffer", "delays", "filter"]

    def __init__(self, n_in: int, n_out: int, delays: Union[np.ndarray, torch.Tensor],
                 filter_weights: Union[np.ndarray, torch.Tensor], weights: Union[np.ndarray, torch.Tensor] = None,
                 dtype: torch.dtype = torch.float64, detach: bool = True, **kwargs):

        # set up intrinsic weights
        if type(filter_weights) is np.ndarray:
            filter_weights = torch.tensor(filter_weights, dtype=dtype)
        if filter_weights.shape[0] != n_in or filter_weights.shape[1] != n_in:
            raise ValueError("Intrinsic weights have to be a square matrix with the number of rows and columns matching"
                             "the number of inputs to the edge.")
        self.filter = filter_weights

        # call super method
        train_params = kwargs.pop("train_params", ["weights", "filter"])
        super().__init__(n_in, n_out, delays=delays, weights=weights, dtype=dtype, detach=detach,
                         train_params=train_params, **kwargs)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        self.buffer = self.filter @ self.buffer.roll(-1, 1)
        self.buffer[:, self.delays] = x
        return super().forward(self.buffer[:, 0])


class LinearMasked(Linear):

    _tensors = ["weights", "mask"]

    def __init__(self, n_in: int, n_out: int, mask: Union[np.ndarray, torch.Tensor],
                 weights: Union[np.ndarray, torch.Tensor] = None, dtype: torch.dtype = torch.float64,
                 detach: bool = True, **kwargs):

        # set mask attribute
        if type(mask) is np.ndarray:
            mask = torch.tensor(mask, dtype=dtype)
        if mask.shape[0] == n_in and mask.shape[1] == n_out:
            mask = mask.T
        elif mask.shape[0] != n_out or mask.shape[1] != n_in:
            raise ValueError("Shape of the provided mask does not match the input and output dimensions of the "
                             "source and target nodes.")
        mask.requires_grad = False
        self.mask = mask

        # call super method
        train_params = kwargs.pop("train_params", ["weights"])
        super().__init__(n_in, n_out, weights=weights, dtype=dtype, detach=detach, train_params=train_params, **kwargs)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return (self.weights * self.mask) @ x


class RLS(Linear):

    _tensors = ["weights", "P"]

    def __init__(self, n_in: int, n_out: int, weights: Union[np.ndarray, torch.Tensor] = None,
                 dtype: torch.dtype = torch.float64, beta: float = 1.0, alpha: float = 1.0):
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

        References
        ----------

        .. [1] Principe et al. (2011) Kernel Adaptive Filtering: A Comprehensive Introduction. John Wiley & Sons.
        """

        # check inputs for correctness
        if beta > 1 or beta < 0:
            raise ValueError("Parameter beta should be a positive scalar between 0 and 1.")
        if alpha < 0:
            raise ValueError("Parameter alpha should be a positive scalar.")

        # initialize zero weights by default
        if weights is None:
            weights = torch.zeros((n_out, n_in), dtype=dtype)

        # set RLS-specific attributes
        self.beta = beta**(-1)
        self.P = alpha * torch.eye(n_in, dtype=dtype)
        self.loss = 0.0

        # call super method
        super().__init__(n_in, n_out, weights=weights, dtype=dtype, detach=True)
        self.train_params = []

    def update(self, x: torch.Tensor, y: torch.Tensor, y_hat: torch.Tensor) -> None:

        z = self.beta * self.P @ x
        k = (1.0 + x @ z) ** (-1)
        error = y - y_hat
        self.weights += torch.outer((y - k * x @ (self.weights + torch.outer(y, z)).T), z)
        self.P -= k * torch.outer(z, z)
        self.loss = torch.inner(error, error)
