"""
Contains classes and functions for model optimization/training
"""
import torch
from torch.nn import Module


class ExpRLS(Module):

    def __init__(self, X: torch.Tensor, y: torch.Tensor, w: torch.Tensor, beta: float, delta: float = 1.0):
        """General form of the extended recursive least-squares algorithm as described in [1]_.

        Parameters
        ----------
        X
            2D array with observations. Each row is an observation, each column is an observed variable.
        y
            2D array with targets. Each row is an observation, each column is a target variable.
            Number of observations must be the same for `X` and `y`.
        w
            2D array with initial set of weights. Each row corresponds to a observation variable and each column
            corresponds to a target variable.
        beta
            Forgetting rate with 0 < beta <= 1. The smaller beta is, the more importance is given to most recent
            observations over past observations.
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
        assert X.shape[0] == y.shape[0]

        # set basic parameters
        self.n_obs = X.shape[0]
        self.n_in = X.shape[1]
        self.n_out = y.shape[1]
        self.delta = delta
        self.beta_sq = beta**2
        self.beta_sq_inv = 1.0/self.beta_sq
        self.P = delta * torch.eye(self.n_in, device=X.device, dtype=X.dtype)
        self.w = w
        self.loss = 0

        # calculate initial weight vector
        if self.n_obs > 1:
            self.w.add(torch.matmul(torch.matmul(y.T, X), torch.pinverse(X.T @ X)))

    def forward(self, x: torch.Tensor, y: torch.tensor) -> torch.Tensor:

        # predict target vector y
        y_pred = self.w @ x

        # calculate current error
        err = y - y_pred

        # calculate the gain
        k = torch.matmul(self.P, x*self.beta_sq_inv)
        k /= (1.0 + torch.inner(x, k))

        # update the weights
        self.w.add_(torch.outer(err, k))

        # update the error correlation matrix
        self.P -= torch.outer(k, torch.inner(x, self.P))
        self.P *= self.beta_sq_inv

        # update loss
        self.loss = torch.inner(err, err)
        return y_pred
