"""
Contains classes and functions for model optimization/training
"""
import torch
import math
from typing import Optional, Callable


class RLS(torch.Module):

    def __init__(self, w: torch.Tensor, forget_rate: float = 1.0, delta: float = 1.0, **kwargs):

        super().__init__()
        self.tau = 1.0/forget_rate
        self.tau_sq = math.sqrt(self.tau)
        self.delta = delta
        self.n = len(w)
        self.w = w
        self.A = delta*torch.eye(self.n)
        self.loss = 0.0

    def forward(self, x: torch.Tensor, y: torch.tensor) -> float:

        z = self.tau * self.A * x
        x_t = x.T
        alpha = (1 + x_t * z)**(-1)
        self.w.add((y - alpha*x_t * (self.w + y*z)) * z)
        self.A.add(-alpha * z * z.T)
        self.loss = float(y - x * self.w)

        return self.loss
