"""
Contains classes and functions for model optimization/training
"""
import torch
from torch.nn import Module
import math
from typing import Optional, Callable


class ExtendedRLS(Module):

    def __init__(self, X: torch.Tensor, y: torch.Tensor, A: torch.Tensor, delta: float = 1.0):
        """General form of the extended recursive least-squares algorithm as described in [1]_.

        Parameters
        ----------
        X
            2D array with observations. Each row is an observation, each column is an observed variable.
        y
            2D array with targets. Each row is an observation, each column is a target variable.
            Number of observations must be the same for `X` and `y`.
        A
            2D square transition matrix. Maps from previous observation coefficients to new observation coefficients.
            Numbers of rows/columns correspond to the number of columns of `X`.
        delta
            Regularization parameter for the initial state of the state-error correlation matrix `P`.

        References
        ----------

        .. [1] Principe et al. (2011) Kernel Adaptive Filtering: A Comprehensive Introduction. John Wiley & Sons.
        """

        super().__init__()

        # make sure matrix dimensions check out
        if len(X.shape) < 2:
            raise ValueError("Observation matrix X should be 2-dimensional, where rows are observation samples, "
                             "and columns are observed variables")
        if len(y.shape) < 2:
            raise ValueError("Target matrix y should be 2-dimensional, where rows are observation samples and columns "
                             "are target variables.")
        if len(A.shape) < 2:
            raise ValueError("Transition matrix should be 2-dimensional, where rows and columns are coefficients of "
                             "observation variables.")
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == A.shape[0] == A.shape[1]

        # set basic parameters
        self.n_obs = X.shape[0]
        self.n_in = X.shape[1]
        self.n_out = y.shape[1]
        self.delta = delta
        self.A = A
        self.A_inv = torch.inverse(A)
        self.A_t = A.T
        self.P = delta * torch.eye(self.n_in, device=X.device, dtype=X.dtype)
        self.w = torch.zeros(size=self.n_in, device=X.device, dtype=X.dtype)

        # calculate initial weight vector
        X_t = X.T
        self.w.add(torch.matmul(torch.pinverse(X_t @ X), torch.matmul(X_t, y)))

    def forward(self, x: torch.Tensor, y: torch.tensor) -> float:

        # calculate variance of the error
        r = x.T @ self.P @ x

        # calculate the gain vector
        k = (self.A @ self.P @ x) / r

        # update the error correlation matrix
        self.P = self.A @ (self.P - self.A_inv @ k @ x.T @ self.P) @ self.A_t

        # update the weights
        loss = y - x @ self.w
        self.w.mul_(self.A)
        self.w.add_(k * loss)

        return loss.T @ loss
